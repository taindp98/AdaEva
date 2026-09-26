"""Reproduce LLaMEA on BBOB — a NATIVE reimplementation of the LLaMEA loop.

Instead of driving the vendored ``llamea.LLaMEA`` class, this file reimplements
the LLaMEA (mu,lambda)/(mu+lambda) Evolution Strategy from scratch as
``LLaMEA_BBOB``, reusing only LLaMEA *primitives*:
  - ``llamea.solution.Solution``            (the individual container)
  - ``llm.sample_solution``                 (LLM call + code/name/description extraction)
  - ``prepare_namespace`` / ``clean_local_namespace``  (safe exec of generated code)

The value of the rewrite: every part a *search-strategy variant* would change is
now a small, clearly-named **override seam** (see ``LLaMEA_BBOB`` docstring). The
racing variant (``src/racing/llamea_bbob.py``) subclasses this and overrides only
``_evaluate_population`` + ``_select_survivors`` (+ the instance pool) — instead of
over-inheriting the EoH racing scaffold and losing LLaMEA's structure.

Paper setting:
  - llm_model    = qwen/qwen3-coder-next
  - temperature  = 0.8
  - t_iter       = 100
  - n_parents    = 10  (mu)
  - n_offspring  = 10  (lambda)
  - elitism      = True
  - dim          = 5
  - budget_factor= 2000  -> per-instance eval budget = 2000 * dim function calls
  - n_runs       = 5

Evolution Strategy (ES) settings:
  - elitism = True :  (mu + lambda). Next parents chosen from parents AND offspring
                      combined; the best are never lost.
  - elitism = False:  (mu , lambda). Next parents chosen ONLY from offspring; the
                      current parents are discarded. lambda MUST be >= mu for
                      selection pressure.

BBOB evaluation (fixed paper settings):
  - fids   = 1..24  (all 24 noiseless functions)
  - iids   = (1, 2, 3)
  - reps   = configurable via --n-reps (default 1; original LLaMEA uses 3)
  - AOCC bounds: lower=1e-8, upper=1e2  (fitness = mean AOCC, higher = better)
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import datetime as _dt
import json
import math
import os
import re
import pathlib
import random
import sys
import textwrap
import threading

import numpy as np
from dotenv import load_dotenv

from ioh import get_problem, logger

# LLaMEA PRIMITIVES we reuse (not the LLaMEA driver class):
from llamea.solution import Solution


def _fixed_llamea_solutions(fixed_pop: list, n_parents: int) -> list:
    """Build ``n_parents`` UNEVALUATED ``Solution`` objects from the fixed initial
    population (list of ``{"source": ...}`` dicts). Fitness is assigned later by
    ``_evaluate_population`` on THIS run's instances, so the fixed heuristics are
    re-scored exactly like LLM-sampled ones — only the LLM sampling is skipped."""
    sols = []
    for h in fixed_pop[:n_parents]:
        src = h["source"]
        m = re.search(r"class\s+(\w+)", src) or re.search(r"def\s+(\w+)", src)
        name = m.group(1) if m else "GeneratedAlgorithm"
        sols.append(Solution(code=src, name=name,
                             description=h.get("description", "") or "",
                             generation=0, operator="fixed-init"))
    return sols
from llamea.llm import LLM as LLaMEA_LLM
from llamea.utils import prepare_namespace, clean_local_namespace
from misc import OverBudgetException, aoc_logger, correct_aoc

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.logger import mirror_stdout_to, make_log_dir
from utils.llm import CachedLLM, OpenRouterClient, OllamaClient, MistralClient, vLLMClient, GoogleClient
from utils import make_wandb_logger
from utils.fixed_init_pop import load_fixed_initial_population

# Fixed BBOB evaluation parameters
_FIDS = range(1, 25)
_IIDS = (1, 2, 3)
_AOC_LOWER = 1e-8
_AOC_UPPER = 1e2

# Unique BBOB instances = |fids| x |iids| — a (fid, iid) problem at fixed dim.
# The number of REPETITIONS per instance (repeated seeded runs, to average out a
# stochastic heuristic) is configurable via --n-reps (default 1; original LLaMEA
# uses 3). Total per-candidate evaluations = _N_UNIQUE_INSTANCES * n_reps.
_N_UNIQUE_INSTANCES = len(_FIDS) * len(_IIDS)  # 72

# Homo mode settings: single BBOB function (Rastrigin = fid 3) across 3 dimensions {5, 10, 20}
_HOMO_FID = 3
_HOMO_DIMS = (5, 10, 20)


def _auto_eval_timeout(budget_factor: int, dim: int) -> float:
    """Per-(candidate, instance) wall-clock cap that SCALES with the func-eval
    budget (= ``budget_factor * dim``). Generous enough not to false-timeout heavy
    surrogate/model-based heuristics, while still cutting infinite loops — and it
    grows with ``dim`` so it stays correct across the {5,10,20} dim sweep (a fixed
    cap would be too tight at high dim). Examples (budget_factor=2000): dim 5 ->
    160s, dim 10 -> 260s, dim 20 -> 460s."""
    return 60.0 + float(budget_factor) * float(dim) / 100.0

# --------------------------------------------------------------------------- #
# LLaMEA default prompt pieces (copied verbatim from llamea.llamea so the loop
# reproduces stock behaviour without instantiating the LLaMEA class). The
# per-task ``task_prompt`` is supplied separately by ``_make_task_prompt``.
# --------------------------------------------------------------------------- #
_ROLE_PROMPT = "You are a highly skilled computer scientist and Python expert."

_EXAMPLE_PROMPT = textwrap.dedent("""
    An example of such code (a simple random search), is as follows:
    ```
    import numpy as np

    class RandomSearch:
        def __init__(self, budget=10000, dim=10):
            self.budget = budget
            self.dim = dim
            self.f_opt = np.inf
            self.x_opt = None

        def __call__(self, func):
            for i in range(self.budget):
                x = np.random.uniform(func.bounds.lb, func.bounds.ub)

                f = func(x)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

            return self.f_opt, self.x_opt
    ```
    """)

_OUTPUT_FORMAT_PROMPT = textwrap.dedent("""
    Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
    # Description: <short-description>
    # Code:
    ```python
    <code>
    ```
    """)

# The single-parent mutation operator (LLaMEA default). This is the VARIATION
# operator; a crossover variant would extend/override ``construct_prompt``.
_MUTATION_PROMPTS = [
    "Refine the strategy of the selected solution to improve it.",
]


class LLaMEA_LLM_Adapter(LLaMEA_LLM):
    """Adapter for LLaMEA's LLM interface to use our CachedLLM."""

    def __init__(self, cached_llm: CachedLLM, model_name: str,
                 temperature: float = 0.9, **kwargs):
        super().__init__(api_key="", model=model_name, **kwargs)
        self._cached = cached_llm
        self._temperature = float(temperature)
        self._call_counter = 0
        # ``query`` runs concurrently when --num-threads > 1 (offspring are sampled
        # on a thread pool). The salt must be unique per call, so the counter
        # increment has to be atomic — otherwise two threads sampling the SAME
        # prompt (e.g. the identical initial-population prompt) could draw the same
        # salt, collide on the cache key, and return duplicate individuals
        # (killing population diversity).
        self._counter_lock = threading.Lock()

    def query(self, session_messages: list, max_retries: int = 5, default_delay: int = 10):
        with self._counter_lock:
            self._call_counter += 1
            salt = f"call-{self._call_counter}"
        # LLaMEA provides session_messages as [{"role": "user", "content": "..."}].
        return self._cached.chat(session_messages, temperature=self._temperature, cache_salt=salt)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class TrajectoryLogger:
    """Write a per-generation best-so-far trajectory to ``trajectory.json`` and
    ``heuristics.json``.

    ``used_budget`` counts cumulative per-instance-candidate evaluations:
        used_budget = (cumulative candidates evaluated) × evals_per_candidate
    where ``evals_per_candidate = |fids| × |iids| × n_reps``.
    """

    def __init__(self, log_dir: pathlib.Path, label: str, evals_per_candidate: int,
                 wandb_logger=None):
        self.log_dir = pathlib.Path(log_dir)
        self.label = label
        self._evals_per_candidate = int(evals_per_candidate)
        self.trajectory: list[dict] = []
        self.heuristics: list[dict] = []
        self._best_score = float("-inf")
        self._best_cand: str | None = None
        self._cum_candidates = 0
        self.wandb_logger = wandb_logger

    def record_generation(self, gen_id: int, population: list) -> None:
        """Call once per generation with the newly evaluated candidates."""
        self._cum_candidates += len(population)
        used_budget = self._cum_candidates * self._evals_per_candidate

        for i, sol in enumerate(population):
            fitness = sol.fitness if hasattr(sol, "fitness") else float("-inf")
            cand_id = f"gen{gen_id:02d}_cand{i:02d}"
            self.heuristics.append({
                "cand_id": cand_id,
                "gen_id": gen_id,
                "score": float(fitness) if math.isfinite(fitness) else float("-inf"),
                "source": sol.code,
            })
            if isinstance(fitness, float) and fitness > self._best_score:
                self._best_score = fitness
                self._best_cand = cand_id

        row = {
            "gen_id": gen_id,
            "cand_id": self._best_cand,
            "score": float(self._best_score),
            "n_instances": self._evals_per_candidate,
            "used_budget": used_budget,
        }
        self.trajectory.append(row)
        if self.wandb_logger:
            self.wandb_logger.log_trajectory_row(row)
        self._save()

    def _save(self) -> None:
        json.dump(
            {"label": self.label, "trajectory": self.trajectory},
            open(self.log_dir / "trajectory.json", "w"),
            indent=2,
        )
        json.dump(
            {"label": self.label, "total_sampled": len(self.heuristics), "heuristics": self.heuristics},
            open(self.log_dir / "heuristics.json", "w"),
            indent=2,
        )
        print(
            f"  [trajectory] gen={self.trajectory[-1]['gen_id']}  "
            f"best={self._best_score:.4f}  used_budget={self.trajectory[-1]['used_budget']}"
        )


# --------------------------------------------------------------------------- #
# BBOB per-(candidate, fid, iid, rep) evaluation — picklable, runs in workers.
# --------------------------------------------------------------------------- #

def _eval_single_task(args_tuple):
    """Evaluate one algorithm on one (fid, iid, rep) BBOB task; return AOCC.

    Top-level so ``joblib.Parallel`` can pickle it. Re-imports its deps inside the
    worker. Returns ``(ind_idx, aocc)`` or ``None`` on a compile/runtime failure.
    """
    ind_idx, fid, iid, rep, dim, budget, code, algorithm_name = args_tuple

    import numpy as np
    from ioh import get_problem, logger
    from misc import OverBudgetException, aoc_logger, correct_aoc
    from llamea.utils import prepare_namespace, clean_local_namespace

    local_ns = {}
    try:
        global_ns, _ = prepare_namespace(code, allowed=["numpy"], logger=None)
        exec(code, global_ns, local_ns)
        local_ns = clean_local_namespace(local_ns, global_ns)
        algo_cls = local_ns[algorithm_name]
    except Exception:
        return None

    l2 = aoc_logger(budget, lower=_AOC_LOWER, upper=_AOC_UPPER, triggers=[logger.trigger.ALWAYS])
    problem = get_problem(fid, iid, dim)
    problem.attach_logger(l2)

    np.random.seed(rep)
    try:
        algorithm = algo_cls(budget=budget, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass
    except Exception:
        return None

    auc = correct_aoc(problem, l2, budget)
    return (ind_idx, auc)


def _make_task_prompt() -> str:
    return textwrap.dedent(
        """
    Your task is to design novel metaheuristic algorithms to solve black box optimization problems.
    The optimization algorithm should handle a wide range of tasks, which is evaluated on a large test suite of noiseless functions.
    Your task is to write the optimization algorithm in Python code.
    The code should contain one function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows.
    Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

    An example of such code is as follows:
    ```python
    import numpy as np

    class RandomSearch:
        def __init__(self, budget=10000, dim=5):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            best_y = float('inf')
            best_x = None
            for _ in range(self.budget):
                x = np.random.uniform(-5.0, 5.0, self.dim)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y
    ```

    Give a novel heuristic algorithm to solve this task. Give it a one-line description with the main idea.
    You can only use the numpy (1.26) library, no other libraries are allowed.
    Give the response in the format:
    # Name: <name of the algorithm>
    # Code: <code>
    """
    )


# --------------------------------------------------------------------------- #
# LLaMEA_BBOB — native ES loop, with clearly-marked override seams
# --------------------------------------------------------------------------- #

class LLaMEA_BBOB:
    """Native (mu,lambda)/(mu+lambda) LLaMEA Evolution Strategy on BBOB.

    A from-scratch reimplementation of the LLaMEA loop (no ``from llamea import
    LLaMEA``), reusing only LLaMEA primitives (``Solution``,
    ``llm.sample_solution``, ``prepare_namespace``). This class alone reproduces
    stock LLaMEA-on-BBOB. Everything a search-strategy variant would change is a
    small OVERRIDE SEAM:

        _initialize_population   -> mu initial solutions (LLM, parallel)         [reuse]
        construct_prompt         -> the VARIATION operator (mutation / crossover)[override to add crossover]
        _select_parents          -> parent selection (random/roulette/tournament)[reuse]
        _sample_offspring        -> lambda offspring from selected parents        [reuse]
        _evaluate_population      -> assign .fitness                              [OVERRIDE: racing = F-race]
        _select_survivors         -> replacement / next parents                   [OVERRIDE: racing = race survivors]
        _stop                     -> stop condition                               [OVERRIDE: racing = max_gen + budget_cap]
        _record_generation        -> trajectory / heuristics logging             [reuse]

    Fitness is the mean AOCC over the BBOB grid (higher = better), so the ES is a
    MAXIMISER (``minimization=False``).
    """

    def __init__(
        self,
        *,
        llm,
        log_dir: pathlib.Path,
        label: str,
        n_parents: int,
        n_offspring: int,
        elitism: bool,
        budget: int,
        dim: int,
        budget_factor: int,
        n_reps: int,
        parent_selection: str = "random",
        tournament_size: int = 3,
        num_threads: int = 1,
        num_cores: int = 1,
        eval_timeout: float | None = 60.0,
        seed: int = 0,
        instance_pool_mode: str = "hetero",
        task_prompt: str | None = None,
        role_prompt: str = _ROLE_PROMPT,
        example_prompt: str = _EXAMPLE_PROMPT,
        output_format_prompt: str = _OUTPUT_FORMAT_PROMPT,
        mutation_prompts: list | None = None,
        minimization: bool = False,
        wandb_logger=None,
        fixed_init_population: list[dict] | None = None,
    ):
        self.llm = llm
        self.log_dir = pathlib.Path(log_dir)
        self.label = label
        self.n_parents = int(n_parents)          # mu
        self.n_offspring = int(n_offspring)       # lambda
        self.elitism = bool(elitism)
        self.budget = int(budget)                 # total CANDIDATE budget (LLM evals)
        self.dim = int(dim)
        self.budget_factor = int(budget_factor)
        self.n_reps = int(n_reps)
        self.parent_selection = parent_selection
        self.tournament_size = int(tournament_size)
        self.num_threads = max(1, int(num_threads))
        self.num_cores = max(1, int(num_cores))
        self.seed = int(seed)
        self.instance_pool_mode = str(instance_pool_mode).lower()
        self.minimization = bool(minimization)
        self._worst = float("inf") if self.minimization else float("-inf")

        # Per-(candidate, instance) wall-clock cap for the BBOB eval. CRUCIAL for
        # LLM-generated code: a heuristic that infinite-loops WITHOUT calling
        # ``func`` never trips ``OverBudgetException`` (that only fires on the
        # func-eval count), so without a cap it would hang a worker forever and
        # stall the run. Enforced with ``Future.result(timeout)`` + pool
        # kill/rebuild (mirrors ``utils.race.elitist_race``). None/<=0 disables it.
        self.eval_timeout = float(eval_timeout) if eval_timeout and float(eval_timeout) > 0 else None
        self._eval_pool = None
        if self.num_cores > 1:
            self._eval_pool = _cf.ProcessPoolExecutor(max_workers=self.num_cores)
            import atexit as _atexit
            _atexit.register(self._shutdown_pool)
        self._UNSET = object()   # sentinel: "result slot not yet filled"

        self.role_prompt = role_prompt
        self.task_prompt = task_prompt if task_prompt is not None else _make_task_prompt()
        self.example_prompt = example_prompt
        self.output_format_prompt = output_format_prompt
        self.mutation_prompts = list(mutation_prompts) if mutation_prompts else list(_MUTATION_PROMPTS)
        self.fixed_init_population = fixed_init_population

        # Evolutionary state
        self.population: list[Solution] = []
        self.run_history: list[Solution] = []
        self.generation = 0
        self.best: Solution | None = None
        self._sample_lock = threading.Lock()

        # Per-generation trajectory / heuristics logging (reprod output contract).
        self._traj = TrajectoryLogger(
            log_dir=self.log_dir, label=self.label,
            evals_per_candidate=(len(_HOMO_DIMS) * len(_IIDS) * self.n_reps if self.instance_pool_mode == "homo" else _N_UNIQUE_INSTANCES * self.n_reps),
            wandb_logger=wandb_logger,
        )

    # ---- Prompt construction (VARIATION operator) -----------------------

    def _init_prompt(self) -> list:
        """Session messages for a fresh initial solution (LLaMEA initialize_single)."""
        content = (
            self.role_prompt + self.task_prompt + self.example_prompt + self.output_format_prompt
        )
        return [{"role": "user", "content": content}]

    def construct_prompt(self, parent: Solution, population: list) -> list:
        """VARIATION operator (OVERRIDE seam). LLaMEA's population-context,
        SINGLE-parent mutation: show the whole population as context, then ask the
        LLM to refine the one selected parent. A crossover variant overrides this
        to fuse multiple parents (EoH E1/E2 style)."""
        population_summary = "\n".join(ind.get_summary() for ind in population)
        error_message = ""
        if parent.error:
            error_message = f"\n### Error Encountered\n{parent.error}\n\n"
        mutation_operator = random.choice(self.mutation_prompts)
        parent.set_operator(mutation_operator)
        final_prompt = (
            f"{self.task_prompt}\n"
            f"The current population of algorithms already evaluated (name, description, score) is:\n"
            f"{population_summary}\n\n"
            f"The selected solution to update is:\n{parent.description}\n\n"
            f"With code:\n\n```python\n{parent.code}\n```\n\n"
            f"Feedback:\n\n{parent.feedback}\n\n"
            f"{error_message}\n"
            f"{mutation_operator}\n\n"
            f"{self.output_format_prompt}\n"
        )
        return [{"role": "user", "content": self.role_prompt + final_prompt}]

    # ---- Sampling (LLM) -------------------------------------------------

    def _sample_one(self, session_messages: list, gen: int) -> Solution:
        """One LLM sample -> Solution. On any failure (incl. NoCodeException) return
        a placeholder Solution with worst fitness, so it still occupies a slot and
        counts toward the budget (matching LLaMEA's failure handling)."""
        try:
            sol = self.llm.sample_solution(session_messages, HPO=False)
            sol.generation = gen
            return sol
        except Exception as e:
            # This placeholder is indistinguishable from a real candidate in
            # heuristics.json apart from its empty `source`, so SAY SO -- an
            # unlogged failure here silently degrades a run (e.g. an LLM
            # timeout that only starts firing once prompts grow late in a run).
            print(f"    [sample] gen {gen}: LLM sample FAILED -> empty candidate "
                  f"({type(e).__name__}: {str(e)[:160]})", flush=True)
            sol = Solution(name="", code="", generation=gen)
            sol.set_scores(self._worst, feedback="", error=e)
            return sol

    def _sample_many(self, prompt_list: list, gen: int) -> list:
        """Sample a batch of prompts, in parallel across --num-threads samplers."""
        results: list = [None] * len(prompt_list)
        if self.num_threads <= 1 or len(prompt_list) <= 1:
            for i, pr in enumerate(prompt_list):
                results[i] = self._sample_one(pr, gen)
        else:
            n = min(self.num_threads, len(prompt_list))
            with _cf.ThreadPoolExecutor(max_workers=n, thread_name_prefix="llamea-sampler") as ex:
                fut_to_i = {ex.submit(self._sample_one, pr, gen): i
                            for i, pr in enumerate(prompt_list)}
                for fut in _cf.as_completed(fut_to_i):
                    results[fut_to_i[fut]] = fut.result()
        return results

    def _initialize_population(self) -> list:
        """mu initial solutions from the init prompt (parallel). Override seam."""
        if self.fixed_init_population is not None:
            return _fixed_llamea_solutions(self.fixed_init_population, self.n_parents)
        prompts = [self._init_prompt() for _ in range(self.n_parents)]
        return self._sample_many(prompts, gen=0)

    def _select_parents(self, population: list, k: int) -> list:
        """Return k parents by ``parent_selection`` (random/roulette/tournament).
        Reused by the racing variant unchanged."""
        method = (self.parent_selection or "random").lower()
        if not population:
            return []
        if method == "random":
            return list(np.random.choice(population, k, replace=True))
        if method == "roulette":
            return self._roulette(population, k)
        if method == "tournament":
            return self._tournament(population, k, self.tournament_size)
        raise ValueError(f"Unknown parent_selection: {self.parent_selection}")

    def _sorted_by_fitness(self, population: list) -> list:
        reverse = not self.minimization  # maximise -> best (highest) first
        return sorted(range(len(population)),
                      key=lambda i: population[i].fitness, reverse=reverse)

    def _roulette(self, population: list, k: int) -> list:
        n = len(population)
        order = self._sorted_by_fitness(population)     # best..worst
        weights = np.zeros(n, dtype=float)
        for pos, idx in enumerate(order):
            weights[idx] = n - pos                       # best -> n, worst -> 1
        s = weights.sum()
        probs = np.ones(n) / n if (s <= 0 or not np.isfinite(s)) else weights / s
        chosen = np.random.choice(range(n), size=k, replace=True, p=probs)
        return [population[i] for i in chosen]

    def _tournament(self, population: list, k: int, tournament_size: int) -> list:
        n = len(population)
        ts = max(1, min(tournament_size, n))
        picked = []
        for _ in range(k):
            cand = random.sample(range(n), ts) if n >= ts else list(
                np.random.choice(range(n), size=ts, replace=True))
            if self.minimization:
                best_i = min(cand, key=lambda i: population[i].fitness)
            else:
                best_i = max(cand, key=lambda i: population[i].fitness)
            picked.append(population[best_i])
        return picked

    def _sample_offspring(self, population: list, gen: int) -> list:
        """lambda offspring: select parents, build a prompt per parent (variation
        operator), sample in parallel. Override seam (racing keeps the same)."""
        parents = self._select_parents(population, self.n_offspring)
        prompts = [self.construct_prompt(p, population) for p in parents]
        return self._sample_many(prompts, gen=gen)

    # ---- Timeout-guarded process pool (mirrors elitist_race dispatch) ---

    def _shutdown_pool(self) -> None:
        """Shut a pool down AND hard-kill its worker processes. ``shutdown(wait=
        False)`` alone does NOT stop a worker running an infinite loop — it would
        survive and hang the interpreter's atexit join. So we terminate() then
        kill() every worker explicitly (mirrors RacingBase._shutdown_eval_pool)."""
        pool = self._eval_pool
        self._eval_pool = None
        if pool is None:
            return
        procs = list((getattr(pool, "_processes", None) or {}).values())
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()

    def _recreate_pool(self, old_pool):
        """Kill a pool holding a runaway/hung worker (incl. its processes); start
        a fresh one."""
        self._eval_pool = old_pool
        self._shutdown_pool()
        self._eval_pool = _cf.ProcessPoolExecutor(max_workers=self.num_cores)
        return self._eval_pool

    def _map_with_timeout(self, task_list: list) -> list:
        """Run ``_eval_single_task`` over ``task_list`` on the process pool with a
        per-task wall-clock cap. A task that hangs past ``eval_timeout`` (or breaks
        the pool) is penalised (``None``) and the pool is killed + rebuilt so a
        runaway worker cannot stall the run; already-finished siblings are salvaged
        and unfinished ones re-submitted. Ported from ``utils.race.elitist_race``.
        A candidate that times out on ONE instance is assumed to hang on ALL of
        them (a runaway loop is instance-independent), so its remaining tasks are
        SHORT-CIRCUITED to ``None`` — otherwise a single bad heuristic would burn
        ``n_instances x eval_timeout`` before being fully penalised.

        Returns results aligned with ``task_list`` (``(ind_idx, aocc)`` or ``None``)."""
        from concurrent.futures import TimeoutError as _CFTimeout
        from concurrent.futures.process import BrokenProcessPool as _BrokenPool
        UNSET = self._UNSET
        n = len(task_list)
        out: list = [UNSET] * n
        dead: set = set()          # ind_idx that already timed out -> skip its rest
        pool = self._eval_pool
        futures = [pool.submit(_eval_single_task, t) for t in task_list]
        i = 0
        while i < n:
            if out[i] is not UNSET:
                i += 1
                continue
            if task_list[i][0] in dead:            # this candidate already hung
                out[i] = None
                i += 1
                continue
            try:
                out[i] = futures[i].result(timeout=self.eval_timeout)
            except (_CFTimeout, _BrokenPool):
                out[i] = None                       # timed out / broken -> penalise
                cand = task_list[i][0]
                dead.add(cand)
                t = task_list[i]
                print(f"  [WARNING] eval TIMEOUT: heuristic '{t[7]}' (pop_idx={cand}) on instance "
                      f"(fid={t[1]}, iid={t[2]}, rep={t[3]}) exceeded {self.eval_timeout:.0f}s "
                      f"-> whole candidate penalised; rebuilding pool", flush=True)
                if self.eval_timeout is None:       # no cap -> cannot recover
                    raise
                # Salvage siblings (of OTHER candidates) that already finished.
                done_now: dict = {}
                for j in range(i + 1, n):
                    if out[j] is UNSET and task_list[j][0] not in dead and futures[j].done():
                        try:
                            done_now[j] = futures[j].result(timeout=0)
                        except Exception:
                            done_now[j] = None
                pool = self._recreate_pool(pool)
                for j in range(i + 1, n):
                    if out[j] is not UNSET:
                        continue
                    if task_list[j][0] in dead:     # skip the dead candidate's rest
                        out[j] = None
                    elif j in done_now:
                        out[j] = done_now[j]
                    else:
                        futures[j] = pool.submit(_eval_single_task, task_list[j])
            except Exception:
                out[i] = None                        # worker crashed -> penalise
            i += 1
        return out

    # ---- Evaluation (OVERRIDE seam: racing = F-race) --------------------

    def _evaluate_population(self, population: list) -> None:
        """Assign ``.fitness`` (mean AOCC) to every solution by running it on the
        full BBOB grid (|fids| x |iids| x n_reps) in a --num-cores process pool.

        This is the KEY override seam: the racing variant replaces the full grid
        with an adaptive F-race over a seeded (instance, seed) pool.
        """
        tasks: list = []
        failed: set = set()
        for idx, sol in enumerate(population):
            issue = None
            try:
                global_ns, issue = prepare_namespace(sol.code, allowed=["numpy"], logger=None)
                ns: dict = {}
                exec(sol.code, global_ns, ns)
                ns = clean_local_namespace(ns, global_ns)
                if self.instance_pool_mode == "homo":
                    for d in _HOMO_DIMS:
                        bgt = self.budget_factor * d
                        for iid in _IIDS:
                            for rep in range(self.n_reps):
                                tasks.append((idx, _HOMO_FID, iid, rep, d, bgt, sol.code, sol.name))
                else:
                    budget = self.budget_factor * self.dim
                    for fid in _FIDS:
                        for iid in _IIDS:
                            for rep in range(self.n_reps):
                                tasks.append((idx, fid, iid, rep, self.dim, budget, sol.code, sol.name))
            except Exception as e:
                sol.set_scores(self._worst, feedback=f" {issue}." if issue else "", error=e)
                failed.add(idx)

        if not tasks:
            return
        import collections
        # Pool path (num_cores > 1): per-task wall-clock cap via _map_with_timeout.
        # Sequential fallback (num_cores == 1): single process, no external cap.
        if self._eval_pool is not None:
            results = self._map_with_timeout(tasks)
        else:
            results = [_eval_single_task(t) for t in tasks]
        expected_tasks_per_cand = (len(_HOMO_DIMS) * len(_IIDS) * self.n_reps if self.instance_pool_mode == "homo" else len(_FIDS) * len(_IIDS) * self.n_reps)
        by_idx = collections.defaultdict(list)
        for r in results:
            if r is not None:
                # Assign 0.0 (worst AOCC) to crashed tasks so generalization failure is penalized
                val = r[1] if (r[1] is not None and math.isfinite(r[1])) else 0.0
                by_idx[r[0]].append(val)

        for idx, sol in enumerate(population):
            if idx in failed:
                continue
            aucs = by_idx[idx]
            # Pad any unreturned/timed-out tasks with 0.0 (worst AOCC)
            if len(aucs) < expected_tasks_per_cand:
                aucs.extend([0.0] * (expected_tasks_per_cand - len(aucs)))

            if not aucs or all(v == 0.0 for v in aucs):
                sol.set_scores(self._worst, "Algorithm failed to evaluate on BBOB instances.")
            else:
                m, s = float(np.mean(aucs)), float(np.std(aucs))
                sol.add_metadata("aucs", aucs)
                sol.set_scores(
                    m,
                    f"The algorithm {sol.name} got an average Area over the convergence "
                    f"curve (AOCC, 1.0 is the best) score of {m:0.4f} with standard "
                    f"deviation {s:0.4f}.",
                )

    # ---- Survivor selection / replacement (OVERRIDE seam) ---------------

    def _select_survivors(self, parents: list, offspring: list) -> list:
        """Next-generation parents. (mu+lambda) if elitism else (mu,lambda),
        truncated by fitness. Racing overrides this to keep the F-race survivors."""
        reverse = not self.minimization
        pool = (parents + offspring) if self.elitism else list(offspring)
        pool.sort(key=lambda s: (s.fitness if math.isfinite(s.fitness) else self._worst),
                  reverse=reverse)
        return pool[: self.n_parents]

    # ---- Stop condition (OVERRIDE seam) ---------------------------------

    def _stop(self) -> bool:
        """LLaMEA budget rule: stop once ``len(run_history) >= budget`` (candidate
        count). Racing overrides with max_generations + budget_cap."""
        return len(self.run_history) >= self.budget

    # ---- Best tracking + logging ----------------------------------------

    def _update_best(self, population: list) -> None:
        for s in population:
            if not math.isfinite(s.fitness):
                continue
            if self.best is None or (
                s.fitness < self.best.fitness if self.minimization else s.fitness > self.best.fitness
            ):
                self.best = s

    def _record_generation(self, population: list) -> None:
        """Log the newly-evaluated batch to trajectory.json / heuristics.json."""
        # nan can only appear if a solution slipped through unevaluated; pin to worst.
        for s in population:
            if not math.isfinite(s.fitness) and math.isnan(s.fitness):
                s.fitness = self._worst
        self._traj.record_generation(gen_id=self.generation, population=population)

    # ---- Main loop ------------------------------------------------------

    def run(self) -> Solution:
        # Generation 1 = the initial population (matches LLaMEA's generation count).
        self.population = self._initialize_population()
        self._evaluate_population(self.population)
        self.run_history += list(self.population)
        self.generation = 1
        self._update_best(self.population)
        self._record_generation(self.population)
        print(f"  gen 01: initialised {len(self.population)} parents; "
              f"best AOCC={self.best.fitness if self.best else float('nan'):.4f}", flush=True)

        while not self._stop():
            offspring = self._sample_offspring(self.population, gen=self.generation + 1)
            if not offspring:
                print(f"  gen {self.generation + 1:02d}: 0 offspring — stopping", flush=True)
                break
            self._evaluate_population(offspring)
            self.run_history += list(offspring)
            self.generation += 1
            self.population = self._select_survivors(self.population, offspring)
            self._update_best(offspring)
            self._record_generation(offspring)
            print(f"  gen {self.generation:02d}: {len(offspring)} offspring; "
                  f"pop={len(self.population)}; used_candidates={len(self.run_history)}/{self.budget}; "
                  f"best AOCC={self.best.fitness if self.best else float('nan'):.4f}", flush=True)

        return self.best


_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "llm_model": "mistral-large-latest",
    "llm_backend": "mistral",
    "temperature": 0.8,
    "t_iter": 100,
    "max_generations": None,
    "n_parents": 10,
    "n_offspring": 10,
    "dim": 5,
    "budget_factor": 2000,
    "n_reps": 1,
    "n_runs": 5,
    "elitism": True,
    "evolution_mode": "population",
    "parent_selection": "random",
    "tournament_size": 3,
    "num_threads": 4,
    "num_cores": 4,
    "eval_timeout": -1.0,
    "instance_pool_mode": "hetero",
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "llm_model": "model",
    "llm_backend": "llm",
    "temperature": "temp",
    "t_iter": "t",
    "max_generations": "mg",
    "n_parents": "np",
    "n_offspring": "no",
    "dim": "d",
    "budget_factor": "bf",
    "n_reps": "nrep",
    "n_runs": "nr",
    "elitism": "eli",
    "evolution_mode": "em",
    "parent_selection": "ps",
    "tournament_size": "ts",
    "num_threads": "nt",
    "num_cores": "nc",
    "eval_timeout": "et",
    "instance_pool_mode": "ipm",
}


def _run_tag(args: argparse.Namespace) -> str:
    overrides = []
    for key, default_val in _BASH_DEFAULTS.items():
        actual_val = getattr(args, key, None)
        if actual_val == default_val:
            continue
        short = _ABBREV.get(key, key)
        if key == "llm_model" and isinstance(actual_val, str) and "/" in actual_val:
            actual_val = actual_val.split("/")[-1]
        if isinstance(actual_val, bool):
            overrides.append(short if actual_val else f"no{short}")
        else:
            overrides.append(f"{short}{actual_val}")
    return "_".join(overrides) if overrides else "default"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce LLaMEA on BBOB (native ES loop).")
    p.add_argument("--llm-model", type=str, default="mistralai/Devstral-Small-2-24B-Instruct-2512")
    p.add_argument("--llm-backend", type=str, default="vllm",
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"])
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--t-iter", type=int, default=100,
                   help="LLaMEA total CANDIDATE budget (LLM evaluations), incl. the "
                        "initial population. NOT the number of generations. Ignored "
                        "when --max-generations is set.")
    p.add_argument("--max-generations", type=int, default=None,
                   help="Number of EVOLUTION generations (repo convention). When set, "
                        "overrides --t-iter: budget = n_parents + max_generations x "
                        "n_offspring. E.g. population(10,10) with --max-generations 10 "
                        "-> 110 candidates = 1 init gen + 10 evolution gens.")
    p.add_argument("--n-parents", type=int, default=10)
    p.add_argument("--n-offspring", type=int, default=10)
    p.add_argument("--elitism", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evolution-mode", type=str, default="population",
                   choices=["population", "single"],
                   help="'population' (default): (mu,lambda)/(mu+lambda) ES using --n-parents / "
                        "--n-offspring / --elitism. 'single': classic (1+1) LLaMEA — forces "
                        "n_parents=n_offspring=1 with elitism (the two population knobs are ignored).")
    p.add_argument("--instance-pool-mode", type=str, default=_BASH_DEFAULTS["instance_pool_mode"], choices=["hetero", "homo"], help="Instance pool mode: 'hetero' (default: 24 BBOB functions of 5 dim, 3 iids) or 'homo' (single BBOB function Rastrigin fid 3 across dims {5, 10, 20}, 3 iids).")
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--budget-factor", type=int, default=2000)
    p.add_argument("--n-reps", type=int, default=1,
                   help="Repeated seeded runs per (fid, iid) instance, averaged to reduce a "
                        "stochastic heuristic's noise (seeds 0..n_reps-1). Total evals per "
                        "candidate = 24 fids x 3 iids x n_reps. Default 1; the original LLaMEA "
                        "BBOB setup uses 3.")
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fix-init-pop", action="store_true",
                   help="Use src/init_pop/llamea_24_bbob.json for generation 0 instead of querying the LLM.")
    p.add_argument("--parent-selection", type=str, default="random",
                   choices=["random", "tournament", "roulette"])
    p.add_argument("--tournament-size", type=int, default=3)
    p.add_argument("--label", type=str, default="llamea_bbob")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=4,
                   help="Parallel samplers for the LLM API (offspring sampling thread pool).")
    p.add_argument("--num-cores", type=int, default=4, help="Parallel evaluators (process pool).")
    p.add_argument("--eval-timeout", type=float, default=-1.0,
                   help="Per-(candidate, instance) wall-clock cap (s) for one BBOB run, enforced when "
                        "--num-cores > 1 (Future.result(timeout) + pool kill/rebuild). Guards against "
                        "LLM heuristics that infinite-loop without calling func. -1 (default) = "
                        "AUTO-SCALE with budget = 60 + budget_factor*dim/100 (e.g. dim 5 -> 160s, "
                        "dim 20 -> 460s); 0 = disabled; >0 = fixed seconds.")
    p.add_argument("--use-wandb", action="store_true", default=False)
    return p.parse_args(argv)


def _build_llm(args: argparse.Namespace, cache_dir: pathlib.Path,
               log_dir: pathlib.Path) -> LLaMEA_LLM_Adapter:
    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            raise RuntimeError("MISTRAL_API_KEY not set in environment / .env")
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
    elif args.llm_backend == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set in environment / .env")
        client = GoogleClient(timeout=args.llm_timeout, model=args.llm_model)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            raise RuntimeError("OPENAI_API_KEY not set in environment / .env")
        client = OpenRouterClient(timeout=args.llm_timeout, x_title="reprod_llamea", model=args.llm_model)
    cached = CachedLLM(client, cache_dir=cache_dir,
                       prompt_log=log_dir / "llm_prompts.jsonl")
    return LLaMEA_LLM_Adapter(cached, model_name=cached.client.model, temperature=args.temperature)


def run_one(args: argparse.Namespace, seed: int, tag: str, log_dir: pathlib.Path,
            cache_dir: pathlib.Path, wandb_logger=None) -> Solution:
    random.seed(seed)
    np.random.seed(seed)

    # Evolution mode: 'single' is classic (1+1) LLaMEA; 'population' is the
    # (mu,lambda)/(mu+lambda) ES driven by --n-parents/--n-offspring/--elitism.
    if args.evolution_mode == "single":
        n_parents, n_offspring, elitism = 1, 1, True
    else:
        n_parents, n_offspring, elitism = args.n_parents, args.n_offspring, args.elitism
    print(f"  instance_pool_mode={args.instance_pool_mode}  evolution_mode={args.evolution_mode}  n_parents={n_parents}  "
          f"n_offspring={n_offspring}  elitism={elitism}")

    # LLaMEA's ``budget`` counts CANDIDATES (LLM evaluations), not generations. The
    # init population already spends n_parents candidates, so a small --t-iter can
    # leave ZERO evolution generations. --max-generations expresses the intent
    # directly and overrides --t-iter: budget = n_parents + G * n_offspring.
    if args.max_generations is not None and args.max_generations >= 0:
        llamea_budget = n_parents + args.max_generations * n_offspring
        print(f"  [budget] --max-generations={args.max_generations} -> budget "
              f"= n_parents({n_parents}) + {args.max_generations} x n_offspring({n_offspring}) "
              f"= {llamea_budget} candidates  (overrides --t-iter={args.t_iter})")
    else:
        llamea_budget = args.t_iter
        approx_gens = max(0, (llamea_budget - n_parents)) // max(1, n_offspring)
        print(f"  [budget] --t-iter={llamea_budget} candidates  "
              f"(= n_parents({n_parents}) init + ~{approx_gens} evolution generations "
              f"of n_offspring({n_offspring}); pass --max-generations to set generations directly)")

    # Resolve --eval-timeout: -1 => auto-scale with budget; 0 => disabled; >0 => fixed.
    if args.eval_timeout is not None and args.eval_timeout < 0:
        eval_timeout = _auto_eval_timeout(args.budget_factor, args.dim)
        print(f"  [eval-timeout] auto = 60 + budget_factor({args.budget_factor}) x dim({args.dim})/100 "
              f"= {eval_timeout:.0f}s per (candidate, instance)")
    else:
        eval_timeout = args.eval_timeout

    llm = _build_llm(args, cache_dir, log_dir)
    fixed_init_population = None
    if args.fix_init_pop:
        fixed_path = ROOT / "src" / "init_pop" / "llamea_24_bbob.json"
        fixed_init_population = load_fixed_initial_population(fixed_path, n_parents)
        print(f"  [fixed-init-pop] loaded {len(fixed_init_population)} heuristics from {fixed_path}")
    es = LLaMEA_BBOB(
        llm=llm,
        log_dir=log_dir,
        label=f"{args.llm_model.split('/')[-1]}_{tag}_seed{seed}",
        n_parents=n_parents,
        n_offspring=n_offspring,
        elitism=elitism,
        budget=llamea_budget,
        dim=args.dim,
        budget_factor=args.budget_factor,
        n_reps=args.n_reps,
        parent_selection=args.parent_selection,
        tournament_size=args.tournament_size,
        num_threads=args.num_threads,
        num_cores=args.num_cores,
        eval_timeout=eval_timeout,
        seed=seed,
        instance_pool_mode=args.instance_pool_mode,
        wandb_logger=wandb_logger,
        fixed_init_population=fixed_init_population,
    )
    return es.run()


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")

    tag = _run_tag(args)
    now = _dt.datetime.now()
    dt_stamp = args.run_stamp or f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"

    print("=== Reproducing LLaMEA (native ES loop) ===")
    print(f"Model       : {args.llm_model}")
    print(f"Temperature : {args.temperature}")
    print(f"Inst Pool Mode: {args.instance_pool_mode}")
    print(f"Dim         : {args.dim}")
    print(f"Per-inst bgt: {args.budget_factor} * {args.dim} = {args.budget_factor * args.dim}")
    print(f"Runs        : {args.n_runs}  (seeds {args.seed} .. {args.seed + args.n_runs - 1})")
    print(f"Tag         : {tag}")

    results = []
    for i in range(args.n_runs):
        seed = args.seed + i
        print(f"\n--- Run {i + 1}/{args.n_runs}  (seed={seed}) ---")

        log_dir = make_log_dir(args.log_root, args.label, dt_stamp, seed, tag=tag)
        cache_dir = args.cache_root / dt_stamp / str(seed)
        cache_dir.mkdir(parents=True, exist_ok=True)

        import yaml
        args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
        with open(log_dir / "args.yaml", "w") as _f:
            yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

        mirror_stdout_to(log_dir / "terminal.txt")
        print(f"terminal output mirrored -> {log_dir / 'terminal.txt'}")

        run_name = f"llamea_{dt_stamp.replace('/', '_')}_{seed}_{tag}"
        wandb_logger = make_wandb_logger(
            enabled=args.use_wandb, project="llm4ad", name=run_name, config=args_dict)

        best = run_one(args, seed, tag, log_dir, cache_dir, wandb_logger)
        results.append(best)
        if best is not None:
            print(f"Best: {best.name}  AOCC={best.fitness}")
        else:
            print("Best: (none — all candidates failed)")

        wandb_logger.finish()

    print("\n=== All Runs Completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
