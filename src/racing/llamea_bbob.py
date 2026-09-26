"""Racing LLaMEA on BBOB — the LLaMEA framework (from full) + the F-race.

This inverts the previous design. Instead of ``RacingLLaMEA(RacingBase)`` (the
EoH racing scaffold as skeleton, with LLaMEA bolted on as a fragile dummy
``LLaMEA(f=None)`` — which crashed every init candidate via
``initialize_single`` self-evaluating through a null evaluator), the skeleton is
now the NATIVE LLaMEA ES loop from ``reprod/llamea_bbob.LLaMEA_BBOB``, and the
irace elitist F-race (``utils.elitist_race``) is plugged into the one seam it
belongs in: it replaces full evaluation + (mu+lambda) truncation with adaptive,
partial, statistically-tested evaluation + survivor selection.

Inherited unchanged from ``LLaMEA_BBOB`` (the LLaMEA core):
    _initialize_population, construct_prompt (population-context mutation),
    _select_parents (random/roulette/tournament), _sample_offspring,
    _sample_many/_sample_one, the LLM adapter, the prompts.
    -> ``_initialize_population`` is native and does NOT self-evaluate, so the old
       ``'NoneType' object is not callable`` crash cannot recur.

Plugged in here (the racing seam):
    - a seeded ``(fid, iid)`` instance pool (72 base problems) that grows fresh
      per-task seeds on demand (irace ``ntimes`` repetition, rep <-> seed);
    - a BBOB per-(instance, seed) scorer returning ``cost = -AOCC`` (racing
      minimises cost; AOCC is maximised) — the SAME evaluation reprod uses, so it
      actually works;
    - ``_race`` = one ``elitist_race`` call over ``elites + offspring`` that
      evaluates partially, runs the Friedman/t-test elimination, and returns the
      coverage-tiered survivors;
    - ``run`` overridden as the race loop; ``_stop`` = max_generations + budget_cap.
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures as _cf
import datetime as _dt
import json
import math
import os
import pathlib
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import collections
import numpy as np
import yaml
from dotenv import load_dotenv

from ioh import get_problem, logger as ioh_logger

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from full.llamea_bbob import LLaMEA_BBOB, LLaMEA_LLM_Adapter, _make_task_prompt, _auto_eval_timeout
from utils import ConfigAS, elitist_race, make_wandb_logger
from utils.race import _config_scores
from utils.logger import make_log_dir, _Tee
from racing.base import select_incumbent, print_instance_order
from utils.llm import CachedLLM, OllamaClient, MistralClient, vLLMClient, OpenRouterClient, GoogleClient
from misc import OverBudgetException, aoc_logger, correct_aoc


def _usage_fields(obj) -> dict:
    """Token counts for the calling thread's most recent LLM call.

    Reads the per-thread usage off the CachedLLM behind ``obj.llm``, so threaded
    sampling cannot attribute another thread's tokens to this record. Returns
    zeros for backends that report no usage (and never raises: logging must not
    be able to abort a run).
    """
    try:
        llm = getattr(obj, "llm", None)
        cached = getattr(llm, "_cached", None)
        u = cached.take_last_usage() if cached is not None else {}
    except Exception:
        u = {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "cached_tokens": int(u.get("cached_tokens", 0) or 0),
        "reasoning_tokens": int(u.get("reasoning_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
    }


# Fixed BBOB evaluation parameters (match reprod)
_FIDS = range(1, 25)
_IIDS = (1, 2, 3)
_AOC_LOWER = 1e-8
_AOC_UPPER = 1e2
_N_UNIQUE_INSTANCES = len(_FIDS) * len(_IIDS)  # 72
_HOMO_FID = 3
_HOMO_DIMS = (5, 10, 20)

BIG_PENALTY = 1e6            # finite TIMEOUT penalty (slow-but-valid heuristic killed)
REJECT_COST = float("inf")   # crash / invalid result -> Inf. The race applies --deal-with-crashed:
#                              'rejection' (default) drops it; 'penalty' converts it to BIG_PENALTY
#                              (finite) and keeps it in the statistical test.
_SEED_MAX = 2 ** 31 - 1


@dataclass(frozen=True)


class SeededInstance:
    """One irace-style task: a (base BBOB instance, seed) pair. ``instance`` is a
    dict ``{fid, iid, dim, budget_factor}``; ``seed`` seeds the stochastic
    heuristic so all configs are compared under identical randomness."""

    instance: Any
    seed: int
    base_idx: int
    rep: int


def _unwrap(task_or_inst) -> tuple:
    if isinstance(task_or_inst, SeededInstance):
        return task_or_inst.instance, task_or_inst.seed
    return task_or_inst, None


# --------------------------------------------------------------------------- #
# BBOB per-(instance, seed) scoring — cost = -AOCC (lower = better).
# Same evaluation as reprod's _eval_single_task, one instance at a time.
# --------------------------------------------------------------------------- #

def _score_bbob_inst(fn: callable, inst) -> float:
    """Run algorithm ``fn`` on one seeded BBOB task; return ``-AOCC`` as cost.

    A crash / non-finite result returns ``REJECT_COST`` (Inf) so the race rejects
    the configuration (irace's Inf-means-rejection convention)."""
    instance, seed = _unwrap(inst)
    fid = int(instance["fid"]); iid = int(instance["iid"])
    dim = int(instance["dim"]); budget = int(instance["budget_factor"]) * dim
    try:
        problem = get_problem(fid, iid, dim)
        l2 = aoc_logger(budget, lower=_AOC_LOWER, upper=_AOC_UPPER,
                        triggers=[ioh_logger.trigger.ALWAYS])
        problem.attach_logger(l2)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        algorithm = fn(budget=budget, dim=dim)
        algorithm(problem)
        aocc = correct_aoc(problem, l2, budget)
    except OverBudgetException:
        aocc = correct_aoc(problem, l2, budget)
    except Exception:
        return REJECT_COST
    aocc = float(aocc)
    return -aocc if math.isfinite(aocc) else REJECT_COST


def score_bbob_config(cfg: "ConfigAS", inst) -> tuple:
    """Picklable process-pool worker. Returns ``(cost, cpu_seconds)``."""
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    try:
        cost = _score_bbob_inst(cfg.callable, inst)
    except Exception:
        cost = REJECT_COST
    cpu = time.process_time() - cpu0
    if os.environ.get("LLAMEA_WORKER_TRACE"):
        dt = time.perf_counter() - t0
        instance, seed = _unwrap(inst)
        seed_str = f"/seed={seed}" if seed is not None else ""
        print(f"    [eval-worker pid={os.getpid()}] {cfg.id}:f{instance['fid']}i{instance['iid']}{seed_str} "
              f"-> cost={cost:.4f} aocc={-cost:.4f}; cpu={cpu:.2f}s wall={dt:.2f}s", flush=True)
    return cost, cpu


# --------------------------------------------------------------------------- #
# RacingLLaMEA
# --------------------------------------------------------------------------- #

class RacingLLaMEA(LLaMEA_BBOB):
    """LLaMEA ES with the irace elitist F-race as evaluation + survivor selection."""

    def __init__(
        self,
        *,
        llm,
        log_dir: pathlib.Path,
        label: str,
        n_parents: int,
        n_offspring: int,
        dim: int,
        n_reps: int = 1,
        budget_factor: int,
        seed: int,
        # racing
        max_generations,
        budget_cap,
        t_first: int = 24,
        t_each: int = 2,
        alpha: float = 0.05,
        test_type: str = "friedman",
        posthoc_test_type: str = "conover",
        race_elitist: bool = True,
        elitist_new_instances: int = 1,
        elitist_limit: int = 12,
        save_pop: bool = False,
        deal_with_crashed: str = "rejection",
        crash_penalty: float = BIG_PENALTY,
        llamea_elitism: bool = True,
        eval_timeout=None,
        timeout_cost: float = BIG_PENALTY,
        deterministic: bool = False,
        # LLaMEA core
        parent_selection: str = "random",
        tournament_size: int = 3,
        num_threads: int = 1,
        num_cores: int = 1,
        instance_pool_mode: str = "hetero",
        prompt_mode: str = "partial_eval",
        fixed_init_population: list | None = None,
        wandb_logger=None,
    ):
        super().__init__(
            llm=llm, log_dir=log_dir, label=label,
            n_parents=n_parents, n_offspring=n_offspring,
            elitism=llamea_elitism, budget=10 ** 9,         # budget unused: _stop overridden
            dim=dim, budget_factor=budget_factor, n_reps=n_reps,
            parent_selection=parent_selection, tournament_size=tournament_size,
            num_threads=num_threads, num_cores=num_cores, seed=seed,
            minimization=False, fixed_init_population=fixed_init_population,
            wandb_logger=wandb_logger,
        )
        # Racing config
        self.wandb_logger = wandb_logger   # LLaMEA_BBOB stashes it in _traj; racing logs directly
        self.n_reps = int(n_reps)
        self.max_generations = max_generations
        self.budget_cap = budget_cap
        self.t_first = int(t_first)
        self.t_each = int(t_each)
        self.alpha = float(alpha)
        self.test_type = test_type
        self.posthoc_test_type = str(posthoc_test_type)
        self.race_elitist = bool(race_elitist)
        self.save_pop = bool(save_pop)
        if deal_with_crashed not in ("rejection", "penalty"):
            raise ValueError(
                f"deal_with_crashed must be 'rejection' or 'penalty', got {deal_with_crashed!r}"
            )
        self.deal_with_crashed = str(deal_with_crashed)
        self._crash_penalty = float(crash_penalty)
        self.elitist_new_instances = int(elitist_new_instances)
        self.elitist_limit = int(elitist_limit)
        self.llamea_elitism = bool(llamea_elitism)
        self._eval_timeout = float(eval_timeout) if eval_timeout and float(eval_timeout) > 0 else None
        self._timeout_cost = float(timeout_cost)
        self._deterministic = bool(deterministic)

        self.instance_pool_mode = str(instance_pool_mode).lower()
        # Seeded instance pool — 72 base problems (hetero) or 9 base problems (homo); grows on demand.
        if self.instance_pool_mode == "homo":
            self._base_instances = [
                {"fid": _HOMO_FID, "iid": iid, "dim": d, "budget_factor": budget_factor}
                for d in _HOMO_DIMS for iid in _IIDS
            ]
        else:
            self._base_instances = [
                {"fid": fid, "iid": iid, "dim": dim, "budget_factor": budget_factor}
                for fid in _FIDS for iid in _IIDS
            ]
        self._seed_rng = np.random.RandomState(int(seed) + 1234)
        self._n_reps_built = 0
        self.instances = self._base_instances if deterministic else self._build_rep(self._base_instances)

        # Evaluation process pool (num_cores workers), recreated on a broken pool.
        self._eval_pool = None
        if self.num_cores > 1:
            self._eval_pool = _cf.ProcessPoolExecutor(max_workers=self.num_cores)
            atexit.register(self._shutdown_pool)

        # Race / generation bookkeeping
        self.budget_used = 0
        self._n_races = 0
        self._evo_gens = 0
        self._elite_cfgs: List[ConfigAS] = []
        self._shared_insts: list = []    # 1-based idx of the instances all current
        #                                  elites share (the fair-score set); set by
        #                                  _rescore_elites_shared, read by the prompt.
        self._cfg_by_id: dict = {}
        self._sol_by_id: dict = {}
        self._cand_counter: dict = {}

        # Own trajectory/heuristics logging (racing budget != full-grid budget).
        self._traj_rows: list = []       # best-so-far series (monotone)
        self.incumbents_mean_rank: list = []   # per-generation mean_rank incumbent
        self.incumbents_mean_cost: list = []   # per-generation mean_cost incumbent
        self._heur_rows: list = []
        self._best_aocc = float("-inf")
        self._best_cand_id = None
        self._best_n_instances = 0       # coverage frozen with the best-so-far score
        self._cpu_seconds_used = 0.0     # summed worker CPU seconds (from score_bbob_config)
        self._timeout_warned: set = set()  # (cfg.id, inst_idx) already WARNed for a timeout
        self._perf_log_path = self.log_dir / "instance_seed_perf.jsonl"
        # Structured JSONL logs (see utils/manuscript_log.py).
        from utils.manuscript_log import ManuscriptLogger
        self._mlog = ManuscriptLogger(self.log_dir, len(self._base_instances))
        self._topup_evals_gen = 0        # fair-scoring top-up evals in the current gen
        self._last_race_experiments = 0
        self._last_race_cpu = 0.0
        self._last_race_step_trace: list = []
        self._last_race_seen: list = []
        self._perf_logged: set = set()
        # Initial-population pre-evaluation (mirror racing/eoh_obp): scores the mu
        # parents (no race) to seed offspring sampling; the first mu+lambda race
        # then resets these histories. Its own RNG keeps instance-seed streams
        # reproducible independent of the pre-eval instance sample.
        self._needs_pre_eval_cleanup = False
        self._pre_eval_rng = np.random.RandomState(int(seed) + 4321)
        # Prompt variant + per-LLM-call prompt logging (debugging the rendered
        # population block). ``partial_eval`` = the rich block (score + N instances
        # + BBOB fids); ``original`` = plain LLaMEA (bare average score).
        self.prompt_mode = str(prompt_mode)
        self._llm_call_idx = 0
        # Prompt record built on the sampling thread, written after that thread's
        # LLM call returns (so it can carry the call's token usage).
        # Prompt records awaiting their LLM call, keyed by id() of the messages
        # list the record describes. Prompts are BUILT on the main thread
        # (_sample_offspring) but SAMPLED on pool workers when --num-threads > 1,
        # so a thread-local hand-off loses them; the prompt object itself travels
        # with the call and is therefore the reliable key.
        self._pending_prompt = {}
        # Offspring sampling runs in a ThreadPoolExecutor (reprod LLaMEA), so the
        # call_idx increment AND the append must be serialised or prompt rows
        # collide / interleave.
        self._prompt_log_lock = threading.Lock()
        self._sampling_gen: Optional[int] = None
        self._prompt_log_path = self.log_dir / "llm_prompts.jsonl"

    # ---- Eval pool management -------------------------------------------

    def _shutdown_pool(self) -> None:
        """Shut down AND hard-kill the worker processes. ``shutdown(wait=False)``
        alone does NOT stop a worker running an infinite loop (it would survive
        and hang the interpreter's atexit join), so terminate()+kill() each worker
        (mirrors RacingBase._shutdown_eval_pool)."""
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
        self._eval_pool = old_pool
        self._shutdown_pool()
        self._eval_pool = _cf.ProcessPoolExecutor(max_workers=self.num_cores)
        return self._eval_pool

    # ---- Seeded (instance, seed) pool -----------------------------------

    def _build_rep(self, base_instances) -> list:
        """One repetition of the base instances as SeededInstance tasks (fresh
        seeds), shuffled so the race isn't biased by task order."""
        rep = self._n_reps_built
        self._n_reps_built += 1
        tasks = [SeededInstance(instance=inst, seed=int(self._seed_rng.randint(0, _SEED_MAX)),
                                base_idx=base_idx, rep=rep)
                 for base_idx, inst in enumerate(base_instances)]
        self._seed_rng.shuffle(tasks)
        return tasks

    def _ensure_task_pool(self, next_instance: int) -> None:
        if self._deterministic:
            return
        need = next_instance + max(1, self.elitist_new_instances) - 1
        while len(self.instances) < need:
            self.instances = self.instances + self._build_rep(self._base_instances)
            print(f"  [seed-pool] extended to {len(self.instances)} tasks "
                  f"({self._n_reps_built} reps of {len(self._base_instances)} base instances)", flush=True)

    # ---- Solution -> ConfigAS -------------------------------------------

    def _materialize(self, sol, gen: int) -> ConfigAS:
        within = self._cand_counter.get(gen, 0)
        self._cand_counter[gen] = within + 1
        cand_id = f"g{gen}_{'p' if gen == 0 else 'c'}{within}"
        src = sol.code or ""
        # Strip an accidental markdown fence if the parser left one.
        if src.lstrip().startswith("```"):
            src = src.split("```", 2)[1] if src.count("```") >= 2 else src
            if src.startswith("python"):
                src = src[len("python"):]
        cfg = ConfigAS(id=cand_id, source=src, entry_point=(sol.name or "x"),
                       name=sol.name or cand_id, global_ns={"np": np})
        self._cfg_by_id[cand_id] = cfg
        self._sol_by_id[cand_id] = sol
        self._heur_rows.append({"cand_id": cand_id, "gen_id": gen, "source": src})
        return cfg

    # ---- Variation operator (OVERRIDE: partial-eval-aware prompt + logging) -

    def _member_eval_info(self, population: list) -> list:
        """Per population member: ``{cand_id, name, description, score,
        n_instances, fids, class_scores}``, where ``class_scores`` lists average
        AOCC scores grouped by the 5 BBOB problem classes.

        Computed over the SHARED instances (``self._shared_insts`` — the identical
        set every elite was fair-scored on), so the per-class AOCCs are comparable
        like-for-like across the population, NOT each config's own disjoint instance
        history. Falls back to a config's own instances only before the first race
        has established a shared set (gen 1)."""
        bbob_classes = collections.OrderedDict([
            ("Separable (f1-f5)", range(1, 6)),
            ("Low/Mod Cond (f6-f9)", range(6, 10)),
            ("High Cond (f10-f14)", range(10, 15)),
            ("Multi-modal Struct (f15-f19)", range(15, 20)),
            ("Multi-modal Weak (f20-f24)", range(20, 25)),
        ])
        cand_by_sol = {id(s): cid for cid, s in self._sol_by_id.items()}
        shared = getattr(self, "_shared_insts", None) or []
        info = []
        for ind in population:
            cid = cand_by_sol.get(id(ind))
            cfg = self._cfg_by_id.get(cid) if cid is not None else None
            if cfg is None:
                insts = []
            elif shared:
                insts = [i for i in shared if i in cfg.costs_by_inst]
            else:                                   # gen-1 fallback: no shared set yet
                insts = list(cfg.costs_by_inst.keys())
            n = len(insts)
            fids = set()
            class_aoccs = collections.defaultdict(list)
            for i in insts:
                cost = cfg.costs_by_inst[i]
                if 0 < i <= len(self.instances):
                    d = _unwrap(self.instances[i - 1])[0]
                    fid = int(d["fid"])
                    fids.add(fid)
                    aocc = -cost if (math.isfinite(cost) and cost < 1e5) else 0.0
                    for c_name, f_range in bbob_classes.items():
                        if fid in f_range:
                            class_aoccs[c_name].append(aocc)
            sorted_fids = sorted(fids)
            class_scores = []
            for c_name in bbob_classes.keys():
                aoccs = class_aoccs.get(c_name, [])
                if aoccs:
                    class_scores.append(f"{c_name}: {np.mean(aoccs):.4f}")
            # Status for the prompt: alive survivor vs a DEAD candidate carried
            # only for diversity (--save-pop) — distinguished into a
            # test-eliminated (valid but worse) vs a crashed (invalid) heuristic.
            if cfg is not None and not cfg.alive:
                status = "crashed" if self._is_crashed(cfg) else "eliminated"
            else:
                status = "alive"
            info.append({"cand_id": cid, "name": ind.name,
                         "description": ind.description,
                         "score": ind.fitness, "n_instances": n,
                         "fids": sorted_fids, "class_scores": class_scores,
                         "status": status})
        return info

    def construct_prompt(self, parent, population: list) -> list:
        """LLaMEA mutation prompt with two selectable variants (``--prompt-mode``),
        logged to ``llm_prompts.jsonl`` on every call for later debugging.

        - ``original``: plain LLaMEA — the population is shown as bare
          ``name: description (Score: X)`` (X = average AOCC over whatever
          instances were tested), identical to ``LLaMEA_BBOB.construct_prompt``.
        - ``partial_eval`` (default): all population members are evaluated on the
          SAME shared (instance, seed) tasks (``_rescore_elites_shared``), stated
          once in a header, and each line exposes that member's average AOCC
          aggregated by the 5 BBOB PROBLEM CLASSES over that shared set — comparable
          like-for-like, and compact (no per-instance float-spam)."""
        info = self._member_eval_info(population)
        if self.prompt_mode == "original":
            messages = super().construct_prompt(parent, population)
        else:
            error_message = ""
            if parent.error:
                error_message = f"\n### Error Encountered\n{parent.error}\n\n"
            mutation_operator = random.choice(self.mutation_prompts)
            parent.set_operator(mutation_operator)
            # The shared (fid, iid) tasks every member was scored on — stated once.
            shared = getattr(self, "_shared_insts", None) or []
            shared_pretty = []
            for i in shared:
                if 0 < i <= len(self.instances):
                    d, _s = _unwrap(self.instances[i - 1])
                    shared_pretty.append(f"f{int(d['fid'])}i{int(d['iid'])}")
            lines = []
            for ind, meta in zip(population, info):
                fit = ind.fitness if ind.fitness is not None else float("-inf")
                status = meta.get("status", "alive")
                if status == "eliminated":
                    lines.append(f"{ind.name}: {ind.description} (Score: -inf, ELIMINATED by the race "
                                 f"— kept only for diversity, not comparable)")
                elif status == "crashed":
                    lines.append(f"{ind.name}: {ind.description} (Score: -inf, FAILED/crashed and "
                                 f"eliminated — kept only for diversity)")
                elif not math.isfinite(fit) or fit <= -1e5 or not meta.get("class_scores"):
                    lines.append(f"{ind.name}: {ind.description} (Score: -inf, failed on evaluation)")
                else:
                    detail = f"per-class AOCC [{', '.join(meta['class_scores'])}]"
                    lines.append(f"{ind.name}: {ind.description} (Score: {fit:.4f}, {detail})")
            population_summary = "\n".join(lines)
            shared_hdr = (
                f"The non-eliminated algorithms below were evaluated on the SAME "
                f"{len(shared_pretty)} shared (instance, seed) tasks "
                f"[{', '.join(shared_pretty)}], so their scores are directly comparable "
                f"(eliminated candidates are marked and score -inf).\n"
                if shared_pretty else "")
            final_prompt = (
                f"{self.task_prompt}\n"
                f"{shared_hdr}"
                f"The current population of algorithms already evaluated "
                f"(name, description, score, per-BBOB-class AOCC over the shared tasks) is:\n"
                f"{population_summary}\n\n"
                f"The selected solution to update is:\n{parent.description}\n\n"
                f"With code:\n\n```python\n{parent.code}\n```\n\n"
                f"Feedback:\n\n{parent.feedback}\n\n"
                f"{error_message}\n"
                f"{mutation_operator}\n\n"
                f"{self.output_format_prompt}\n"
            )
            messages = [{"role": "user", "content": self.role_prompt + final_prompt}]
        self._stash_llm_prompt(parent, info, messages)
        return messages

    def _init_prompt(self) -> list:
        """Initial-population prompt (generation 0), stashed for the prompt log.

        ``_initialize_population`` builds its prompts here rather than through
        ``construct_prompt``, so without this override gen-0 calls would flush an
        empty stash and vanish from ``llm_prompts.jsonl`` (losing mu calls' worth
        of token accounting per run). There is no parent or population context to
        show at this point, hence the empty/None provenance fields."""
        messages = super()._init_prompt()
        rec = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "call_idx": None,   # assigned under the lock below
            "gen": 0,
            "prompt_mode": self.prompt_mode,
            "scores_exposed": False,   # nothing has been evaluated yet at gen 0
            "shared_instance_ids": list(getattr(self, "_shared_insts", []) or []),
            "shared_instance_count": len(getattr(self, "_shared_insts", []) or []),
            "parent_cand_id": None,
            "parent_name": None,
            "population_shown": [],
            "prompt": messages,
        }
        with self._prompt_log_lock:
            rec["call_idx"] = self._llm_call_idx
            self._llm_call_idx += 1
            self._pending_prompt[id(messages)] = rec
        return messages

    def _stash_llm_prompt(self, parent, info: list, messages: list) -> None:
        """Build the ``llm_prompts.jsonl`` record for one LLM (mutation) call and
        stash it on this thread.

        The record is WRITTEN later by ``_sample_one`` (see ``_flush_llm_prompt``),
        once the call has returned and its token usage is known. Prompts are built
        on the calling thread and sampled on that same thread, so a thread-local
        hand-off keeps prompt and usage paired under --num-threads > 1.
        Non-finite scores are stringified (JSON-safe)."""
        def _num(x):
            return x if isinstance(x, (int, float)) and math.isfinite(x) else str(x)
        cand_by_sol = {id(s): cid for cid, s in self._sol_by_id.items()}
        # population_shown: what the LLM actually conditioned on this call — each
        # member's DISPLAYED score, coverage and status (Dims 6,7).
        pop = [{"cand_id": m["cand_id"], "name": m["name"],
                "displayed_score": _num(m["score"]),
                "coverage": m["n_instances"], "fids": m["fids"],
                "status": m.get("status", "alive")} for m in info]
        rec = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "call_idx": None,   # assigned under the lock below (atomic read-and-increment)
            "gen": self._sampling_gen,
            "prompt_mode": self.prompt_mode,
            "scores_exposed": True,      # LLaMEA's prompt shows per-candidate scores (both modes)
            "shared_instance_ids": list(getattr(self, "_shared_insts", []) or []),
            "shared_instance_count": len(getattr(self, "_shared_insts", []) or []),
            "parent_cand_id": cand_by_sol.get(id(parent)),
            "parent_name": parent.name,
            "population_shown": pop,
            "prompt": messages,
        }
        with self._prompt_log_lock:
            rec["call_idx"] = self._llm_call_idx
            self._llm_call_idx += 1
            self._pending_prompt[id(messages)] = rec

    def _flush_llm_prompt(self, messages, status: str) -> None:
        """Write the prompt record for ``messages``, tagged with the call's token
        usage and outcome. Called on every ``_sample_one`` exit path so a failed
        sample still leaves its rendered prompt on record."""
        with self._prompt_log_lock:
            rec = self._pending_prompt.pop(id(messages), None)
        if rec is None:
            return
        rec["status"] = status
        # Token usage for this call, read per-thread so threaded sampling
        # (num_threads>1) cannot misattribute another thread's counts.
        # Cost is derived from these in post-processing.
        rec.update(_usage_fields(self))
        with self._prompt_log_lock:
            with open(self._prompt_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _sample_one(self, session_messages: list, gen: int):
        """Sample one offspring, then flush this thread's prompt record with the
        token usage the call just consumed."""
        try:
            sol = super()._sample_one(session_messages, gen)
        except Exception:
            self._flush_llm_prompt(session_messages, "exception")
            raise
        self._flush_llm_prompt(
            session_messages, "ok" if getattr(sol, "code", "") else "sample_fail"
        )
        return sol

    def _sample_offspring(self, population: list, gen: int) -> list:
        """Stash the generation being sampled so ``_log_llm_prompt`` can tag each
        prompt record with it, then defer to the LLaMEA implementation."""
        self._sampling_gen = int(gen)
        return super()._sample_offspring(population, gen)

    # ---- Timeout warning (shared by pre-eval + race) --------------------

    def _warn_timeout(self, cfg_id: str, inst_idx: int, phase: str = "race") -> None:
        """Print a one-time WARNING (deduped by (cfg, instance)) that ``cfg_id``
        hit the per-eval wall-clock cap on instance ``inst_idx``."""
        key = (cfg_id, inst_idx)
        if key in self._timeout_warned:
            return
        self._timeout_warned.add(key)
        task = self.instances[inst_idx - 1] if 0 < inst_idx <= len(self.instances) else None
        if isinstance(task, SeededInstance):
            desc = (f"(fid={task.instance.get('fid')}, iid={task.instance.get('iid')}, "
                    f"seed={task.seed}, inst_idx={inst_idx})")
        else:
            desc = f"(inst_idx={inst_idx})"
        print(f"  [WARNING] eval TIMEOUT [{phase}]: heuristic '{cfg_id}' on instance {desc} "
              f"exceeded {self._eval_timeout}s -> penalised with timeout_cost", flush=True)

    # ---- Initial-population pre-evaluation (NO race) --------------------

    def _pre_evaluate(self, init_cfgs: List[ConfigAS]) -> None:
        """PLAIN pre-scoring of the initial mu population on ``t_first`` sampled
        (instance, seed) tasks — NO race, NO elimination (mirrors
        ``racing/eoh_obp._pre_evaluate``). Sets each parent's ``mean_cost`` +
        solution fitness so the FIRST offspring sampling and the LLM prompt have
        meaningful scores; the first mu+lambda race then resets these histories
        (``_needs_pre_eval_cleanup``) and re-evaluates everyone fresh. Each eval
        is timeout-guarded via the pool (BBOB heuristics can hang — a deliberate
        divergence from eoh_obp's serial pre-eval) and counts toward
        ``budget_used``."""
        self._needs_pre_eval_cleanup = True
        self._ensure_task_pool(1)
        n_sample = min(self.t_first, len(self.instances))
        if n_sample == 0 or not init_cfgs:
            return
        idxs = sorted(int(i) + 1 for i in
                      self._pre_eval_rng.choice(len(self.instances), size=n_sample, replace=False))
        sampled = [(i, self.instances[i - 1]) for i in idxs]
        print(f"  [pre-eval] scoring {len(init_cfgs)} initial candidates on {n_sample} "
              f"sampled (instance, seed) tasks (1-based idx {idxs}) — no race", flush=True)
        t0 = time.time()
        tasks = [(ci, cfg, i, inst)
                 for ci, cfg in enumerate(init_cfgs) for (i, inst) in sampled]
        results = self._map_scores_with_timeout(tasks)   # aligned: (cost, cpu)
        for (ci, cfg, inst_idx, _inst), (cost, cpu) in zip(tasks, results):
            cfg.record(inst_idx, cost)
            self.budget_used += 1
            self._cpu_seconds_used += cpu
        # Flow pre-eval scores to the Solutions so offspring sampling + prompt
        # have fitness (= -mean_cost = AOCC).
        for cfg in init_cfgs:
            sol = self._sol_by_id.get(cfg.id)
            if sol is None:
                continue
            mc = cfg.mean_cost
            if math.isfinite(mc):
                sol.set_scores(
                    -mc,
                    feedback=(f"The algorithm {sol.name} scored AOCC {-mc:0.4f} in pre-evaluation "
                              f"(partial: {len(cfg.costs_by_inst)} (instance, seed) tasks, no race)."),
                )
            else:
                err = self._capture_error(cfg)
                sol.set_scores(
                    self._worst,
                    feedback=f"The algorithm {sol.name} failed / was rejected during pre-evaluation.",
                    error=err,
                )
        dt = time.time() - t0
        n_valid = sum(1 for c in init_cfgs if math.isfinite(c.mean_cost))
        print(f"  [pre-eval] done in {dt:.1f}s  {n_valid}/{len(init_cfgs)} valid  "
              f"budget_used={self.budget_used}", flush=True)

    def _map_scores_with_timeout(self, tasks: list) -> list:
        """Score ``(ci, cfg, inst_idx, inst)`` tasks via the eval pool with a
        per-eval wall-clock cap; returns ``(cost, cpu)`` aligned with ``tasks``.
        A hung / broken eval is penalised (``timeout_cost``) and the pool rebuilt;
        a candidate that hangs on one instance is assumed to hang on all, so its
        remaining tasks are short-circuited (mirrors
        ``reprod.LLaMEA_BBOB._map_with_timeout``). No race semantics."""
        n = len(tasks)
        PENALTY = (self._timeout_cost, 0.0)
        REJECT = (REJECT_COST, 0.0)
        if self._eval_pool is None:                 # single-core / diagnostic: serial
            res = []
            for _ci, cfg, _ii, inst in tasks:
                try:
                    res.append(score_bbob_config(cfg, inst))
                except Exception:
                    res.append(REJECT)
            return res
        from concurrent.futures import TimeoutError as _CFTimeout
        from concurrent.futures.process import BrokenProcessPool as _BrokenPool
        out: list = [None] * n
        dead: set = set()                            # ci that already hung -> skip rest
        pool = self._eval_pool
        futures = [pool.submit(score_bbob_config, t[1], t[3]) for t in tasks]
        i = 0
        while i < n:
            if out[i] is not None:
                i += 1
                continue
            ci, cfg, inst_idx, _inst = tasks[i]
            if ci in dead:                           # candidate already timed out
                cfg.timed_out_insts.add(inst_idx)
                out[i] = PENALTY
                i += 1
                continue
            try:
                out[i] = futures[i].result(timeout=self._eval_timeout)
            except (_CFTimeout, _BrokenPool):
                out[i] = PENALTY
                dead.add(ci)
                cfg.timed_out_insts.add(inst_idx)
                self._warn_timeout(cfg.id, inst_idx, phase="pre-eval")
                if self._eval_timeout is None:       # no cap -> cannot recover
                    raise
                done_now: dict = {}
                for j in range(i + 1, n):
                    if out[j] is None and tasks[j][0] not in dead and futures[j].done():
                        try:
                            done_now[j] = futures[j].result(timeout=0)
                        except Exception:
                            done_now[j] = REJECT
                pool = self._recreate_pool(pool)
                for j in range(i + 1, n):
                    if out[j] is not None:
                        continue
                    if tasks[j][0] in dead:
                        out[j] = PENALTY
                    elif j in done_now:
                        out[j] = done_now[j]
                    else:
                        futures[j] = pool.submit(score_bbob_config, tasks[j][1], tasks[j][3])
            except Exception:
                out[i] = REJECT
            i += 1
        return out

    # ---- Failure diagnosis for the LLM repair loop ----------------------

    def _capture_error(self, cfg: ConfigAS) -> Optional[Exception]:
        """Reproduce a crashed heuristic's exception IN THE MAIN PROCESS so the LLM
        receives the real error (type + offending source line) via
        ``set_scores(error=)`` — restoring LLaMEA's error-driven repair loop that
        racing otherwise lost (the pool workers swallow the exception as REJECT_COST).
        Unlike reprod (which only catches exec-time errors), running the heuristic
        also catches RUNTIME crashes (e.g. a name used but never defined in the
        function body). Returns the live exception, or ``None`` if it doesn't
        reproduce. Hang-safe: it reproduces only on a penalty instance that is NOT a
        recorded timeout (``timed_out_insts``), so a non-terminating heuristic — whose
        penalty may be a FINITE BIG_PENALTY under ``--deal-with-crashed penalty`` — is
        never re-run here (that would infinite-loop). A genuine crash raises at once."""
        timed_out = getattr(cfg, "timed_out_insts", None) or set()
        failed = next((k for k, v in cfg.costs_by_inst.items()
                       if v >= BIG_PENALTY and k not in timed_out), None)
        if failed is None or not (0 < failed <= len(self.instances)):
            return None
        instance, seed = _unwrap(self.instances[failed - 1])
        try:
            fn = cfg.callable                               # exec of the heuristic (may raise)
            fid = int(instance["fid"]); iid = int(instance["iid"])
            dim = int(instance["dim"]); budget = int(instance["budget_factor"]) * dim
            problem = get_problem(fid, iid, dim)
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
            algorithm = fn(budget=budget, dim=dim)          # LLaMEA class-based convention
            algorithm(problem)                              # run -> reproduces the crash
            return None
        except Exception as e:
            return e

    # ---- The F-race (plugged into the eval + select seam) ---------------

    def _race(self, cfgs: List[ConfigAS], next_instance: int) -> dict:
        """One elitist race over ``cfgs`` (elites + offspring). Evaluates partially,
        eliminates by the Friedman/t-test, returns coverage-tiered survivors and
        writes each config's ``costs_by_inst`` (cost = -AOCC)."""
        if not cfgs:
            return {"survivors": [], "next_instance": next_instance, "break_reason": "no candidates"}
        # First race after the initial-population pre-eval: discard the pre-eval
        # histories so elites and offspring compete fresh on the race's own
        # instance sample (mirror racing/eoh_obp._race cleanup).
        if self._needs_pre_eval_cleanup:
            for c in cfgs:
                c.reset_history()
            self._needs_pre_eval_cleanup = False
            print(f"  [pre-eval cleanup] reset histories for {len(cfgs)} configs "
                  f"before first race", flush=True)
        if not self.race_elitist:
            for c in cfgs:
                c.reset_history()
        self._ensure_task_pool(next_instance)

        def runner(params, inst):
            return _score_bbob_inst(params, inst)

        seed = self.seed + self._n_races
        race_idx = self._n_races
        self._n_races += 1

        # race_log.jsonl phase-1: roster ENTERING the race (after any reset above).
        from utils.race_log import snapshot_candidates, append_race_record
        _race_budget_before = self.budget_used
        _race_before_roster = snapshot_candidates(cfgs)

        out = elitist_race(
            configurations=cfgs,
            target_runner=runner,
            instances_log=self.instances,
            max_experiments=len(cfgs) * len(self.instances) + 1,
            next_instance=next_instance,
            elitist=self.race_elitist,
            elitist_new_instances=self.elitist_new_instances,
            elitist_limit=self.elitist_limit,
            early_stopping_non_elitist=False,
            first_test=self.t_first,
            each_test=self.t_each,
            test_type=self.test_type,
            posthoc_test_type=self.posthoc_test_type,
            alpha=self.alpha,
            metric="sum_ranks",
            min_survival=self.n_parents,
            sample_instances=True,
            seed=seed,
            verbose=2,
            eval_pool=self._eval_pool,
            pool_runner=score_bbob_config,
            eval_timeout=self._eval_timeout,
            timeout_cost=self._timeout_cost,
            crash_penalty=(None if self.deal_with_crashed == "rejection" else self._crash_penalty),
            pool_recreate=self._recreate_pool,
        )
        self.budget_used += int(out["experiments_used"])
        self._cpu_seconds_used += float(out.get("cpu_seconds", 0.0))
        self._last_race_experiments = int(out["experiments_used"])
        self._last_race_cpu = float(out.get("cpu_seconds", 0.0))
        self._last_race_step_trace = out.get("step_trace", [])
        self._last_race_seen = out.get("seen_instances", [])

        # WARNING for any (heuristic, instance) that hit the eval timeout this race.
        # elitist_race marks the timed-out instance on the config (``timed_out_insts``);
        # we surface each new one once (de-duped, since elites carry the set forward).
        for c in cfgs:
            for inst_idx in sorted(getattr(c, "timed_out_insts", ())):
                self._warn_timeout(c.id, inst_idx, phase="race")

        seen = out.get("seen_instances", [])
        scores = _config_scores(list(cfgs), seen, "sum_ranks")

        # Flow race costs back to the Solutions as fitness (= -mean_cost = AOCC)
        # + feedback, so construct_prompt's population summary is meaningful.
        for c in cfgs:
            sol = self._sol_by_id.get(c.id)
            if sol is None:
                continue
            mc = c.mean_cost
            if math.isfinite(mc):
                if self.prompt_mode == "original":
                    # Pure plain-LLaMEA baseline: EXACT wording of reprod.LLaMEA_BBOB
                    # (mean AOCC over ALL evaluated instances + std) — no mention of
                    # racing / partial evaluation, so `original` is a true baseline.
                    _aoccs = [-v for v in c.costs_by_inst.values()]
                    _std = float(np.std(_aoccs)) if _aoccs else 0.0
                    sol.set_scores(
                        -mc,
                        f"The algorithm {sol.name} got an average Area over the convergence "
                        f"curve (AOCC, 1.0 is the best) score of {-mc:0.4f} with standard "
                        f"deviation {_std:0.4f}.",
                    )
                else:
                    sol.set_scores(
                        -mc,
                        feedback=(f"The algorithm {sol.name} scored AOCC {-mc:0.4f} in the race "
                                  f"(partial evaluation over {len(c.costs_by_inst)} (instance, seed) tasks)."),
                    )
            else:
                # Failed / rejected: reproduce the crash in-process and feed the real
                # error (type + offending line) back to the LLM — restoring LLaMEA's
                # error-driven repair loop (construct_prompt's `### Error Encountered`).
                # reprod captures only exec-time errors; here we also catch RUNTIME
                # crashes (e.g. an undefined name used in the body — the dominant one).
                err = self._capture_error(c)
                if self.prompt_mode == "original":
                    # Pure plain-LLaMEA baseline crash feedback (matches reprod).
                    sol.set_scores(self._worst,
                                   "Algorithm failed to evaluate on BBOB instances.",
                                   error=err)
                else:
                    sol.set_scores(
                        self._worst,
                        feedback=f"The algorithm {sol.name} failed / was rejected during racing evaluation.",
                        error=err,
                    )

        survivor_cfgs = [c for c in cfgs if c.alive]
        survivors = sorted(survivor_cfgs,
                           key=lambda c: (scores.get(id(c), float("inf")), c.mean_cost))
        # For --save-pop: survivors first, then the eliminated configs ranked by
        # sum_ranks, so run() can top the elite pool back up to mu with the best-ranked
        # eliminated candidates when a race over-prunes (mirrors RacingBase._race).
        eliminated = sorted([c for c in cfgs if not c.alive],
                            key=lambda c: (scores.get(id(c), float("inf")), c.mean_cost))
        self._append_perf_log(cfgs, seen, gen_id=self.generation + 1)

        # race_log.jsonl phase-2: settled roster (restricted to this race's instances)
        # + budget bracket. llamea runs one race per generation (no per-operator
        # sub-races), so operator is always null here. Best-effort.
        try:
            append_race_record(self.log_dir, {
                "race_idx": int(race_idx),
                "gen_id": int(self.generation + 1),
                "operator": None,
                "seed": int(seed),
                "phase_before": {
                    "used_budget": int(_race_budget_before),
                    "n_candidates": len(_race_before_roster),
                    "candidates": _race_before_roster,
                },
                "phase_after": {
                    "used_budget": int(self.budget_used),
                    "experiments_used": int(out.get("experiments_used", 0)),
                    "break_reason": out.get("break_reason", ""),
                    "n_survivors": len(survivors),
                    # Pass the race's computed sum_ranks dict (id(cfg)->score): llamea's
                    # ConfigAS never populate ranks_by_inst, so their sum_ranks property
                    # would log 0; this is the actual rank key the survivors were sorted by.
                    "candidates": snapshot_candidates(survivors + eliminated, set(seen), scores=scores),
                },
            })
        except Exception as e:
            print(f"  [race-log] WARN: llamea._race logging failed: {e}", flush=True)

        return {"survivors": survivors, "all_records_sorted": survivors + eliminated,
                "next_instance": out["next_instance"],
                "break_reason": out.get("break_reason", ""),
                "step_trace": out.get("step_trace", []),
                "experiments_used": int(out.get("experiments_used", 0)),
                "cpu_seconds": float(out.get("cpu_seconds", 0.0)),
                "seen_instances": out.get("seen_instances", [])}

    # ---- Comparable LLM-facing fitness (shared-instance re-scoring) ------

    @staticmethod
    def _is_crashed(c: ConfigAS) -> bool:
        """A dead candidate is CRASHED (invalid code) if any recorded cost is a
        crash/penalty (non-finite or >= 1e5); otherwise it was merely
        test-eliminated (a valid heuristic that lost the Friedman/Conover test)."""
        return any((not math.isfinite(v)) or v >= 1e5 for v in c.costs_by_inst.values())

    def _rescore_elites_shared(self) -> None:
        """Re-score the current elite Solutions on the instances ALL elites SHARE,
        so the fitness fed to the NEXT generation's LLaMEA prompt + parent selection
        is COMPARABLE across elites (identical instances), instead of each elite's
        mean over its own DISJOINT accumulated instance set — the non-comparable
        number (``mean_cost``) that misled the LLM and kept racing-LLaMEA from
        climbing (unlike EoH, whose prompt never exposes scores).

        MODE-DEPENDENT (the fair-scoring on/off ablation): the shared rescoring of
        ALIVE elites runs ONLY in ``prompt_mode == "partial_eval"``. In ``original``
        mode the alive elites keep their plain-LLaMEA fitness (``-mean_cost`` over
        ALL their own evaluated instances, set in ``_race``). Dead (save-diversity)
        elites are ``-inf``'d in BOTH modes.

        The shared set is the intersection of the elites' ``costs_by_inst`` and so
        REUSES race evaluations already paid for (free when the elites are mature).
        Floor: if that intersection is smaller than ``t_first`` — a fresh,
        low-coverage elite just entered the pool — top up the under-covered elites
        (``_topup_shared_instances``) so the comparable score never degrades to
        noise. Only the ELITES are re-scored (the lineage the LLM refines); the
        F-race survivor test + logged incumbents keep using each config's FULL
        ``costs_by_inst``, so racing's budget efficiency and selection are intact."""
        self._topup_evals_gen = 0
        elites = self._elite_cfgs
        if not elites:
            return
        # Under --save-pop the pool is topped up to mu with DEAD (eliminated
        # or crashed) candidates, kept ONLY for offspring diversity. They must not
        # enter the shared-instance measurement: an early-killed candidate carries
        # few instances and would shrink the shared set, thinning the reliable
        # LLM-facing score. Only ALIVE survivors define + receive the fair shared
        # score; dead candidates are scored -inf and labeled for the prompt (with
        # the crash error preserved so the repair loop still works if one is later
        # picked as a parent).
        alive_elites = [c for c in elites if c.alive]
        dead_elites = [c for c in elites if not c.alive]
        for c in dead_elites:
            sol = self._sol_by_id.get(c.id)
            if sol is None:
                continue
            if self._is_crashed(c):
                # Re-capture a LIVE exception for the repair loop. Do NOT read back
                # sol.error — set_scores stores it as a STRING, and passing a str to
                # set_scores(error=) throws (it expects an Exception with
                # __traceback__). _capture_error returns a live Exception or None and
                # is hang-safe (it skips timed-out instances).
                sol.set_scores(
                    self._worst,
                    feedback=(f"The algorithm {sol.name} FAILED / crashed during racing and was "
                              f"eliminated (kept only for offspring diversity)."),
                    error=self._capture_error(c),
                )
            else:
                sol.set_scores(
                    self._worst,
                    feedback=(f"The algorithm {sol.name} was ELIMINATED by the race (a valid but "
                              f"worse heuristic; kept only for offspring diversity)."),
                )
        # 'original' prompt mode = plain-LLaMEA baseline: the LLM prompt + parent
        # selection use each ALIVE config's fitness as its mean over ALL its OWN
        # evaluated instances (set in _race as -mean_cost), NOT the shared-instance
        # fair score. So skip the shared rescoring (and its top-up evals) here; only
        # 'partial_eval' uses the comparable shared score. Dead elites are still
        # -inf'd above so save-diversity carry-overs don't mislead the prompt.
        if self.prompt_mode == "original":
            self._shared_insts = []
            return
        if not alive_elites:
            self._shared_insts = []
            return
        inst_sets = [set(c.costs_by_inst.keys()) for c in alive_elites]
        shared = set.intersection(*inst_sets) if inst_sets else set()
        floor = min(self.t_first, len(self.instances))
        if len(shared) < floor:
            shared = self._topup_shared_instances(alive_elites, floor)
        if not shared:
            self._shared_insts = []
            return
        shared_list = sorted(shared)
        self._shared_insts = shared_list   # exposed to the partial_eval prompt
        # Debug: the actual BBOB (fid, iid) of the shared instances the fair,
        # comparable score is computed over this race (with rep for reps > 1).
        pretty = []
        for i in shared_list:
            if 0 < i <= len(self.instances):
                d, _seed = _unwrap(self.instances[i - 1])
                rep = getattr(self.instances[i - 1], "rep", 0)
                pretty.append(f"f{int(d['fid'])}i{int(d['iid'])}"
                              + (f"r{rep}" if rep else ""))
        print(f"  [elite-rescore] fair score over {len(shared_list)} shared instances "
              f"(1-based idx {shared_list}) for {len(alive_elites)} alive elites "
              f"({len(dead_elites)} dead kept for diversity): {', '.join(pretty)}", flush=True)
        for c in alive_elites:
            sol = self._sol_by_id.get(c.id)
            if sol is None:
                continue
            # Convert costs to AOCC: valid tasks (cost < 1e5) -> -cost; crashed tasks (cost >= 1e5) -> 0.0
            aoccs = [(-v if (math.isfinite(v) and v < 1e5) else 0.0)
                     for i in shared_list if i in c.costs_by_inst for v in [c.costs_by_inst[i]]]
            if len(aoccs) < len(shared_list):
                aoccs.extend([0.0] * (len(shared_list) - len(aoccs)))
            if not aoccs or all(v == 0.0 for v in aoccs):
                sol.set_scores(self._worst, feedback=f"The algorithm {sol.name} failed on all shared evaluation tasks.")
            else:
                mean_aocc = float(np.mean(aoccs))
                sol.set_scores(
                    mean_aocc,
                    feedback=(f"The algorithm {sol.name} scored average AOCC {mean_aocc:0.4f} over the "
                              f"{len(shared_list)} (instance, seed) tasks shared by all current elites "
                              f"(comparable, like-for-like score; crashed tasks scored 0.0)."),
                )

    def _topup_shared_instances(self, elites: List[ConfigAS], floor: int) -> set:
        """Grow the elite-shared instance set to ``floor`` by evaluating
        under-covered elites on the most-commonly-covered instances (so the fewest
        new evals are needed). Because instances are recorded by id, after the
        top-up every elite holds all ``targets``; returns that shared set. Counts
        the extra evals against ``budget_used`` (bounded: only fires when a fresh
        elite joins, at most mu x t_first evals)."""
        cover = collections.Counter()
        for c in elites:
            cover.update(c.costs_by_inst.keys())
        targets = [i for i, _ in sorted(cover.items(), key=lambda kv: (-kv[1], kv[0]))][:floor]
        if len(targets) < floor:                         # too few distinct instances seen
            for i in range(1, len(self.instances) + 1):
                if i not in targets:
                    targets.append(i)
                if len(targets) >= floor:
                    break
        targets = targets[:floor]
        tasks = [(ci, c, i, self.instances[i - 1])
                 for ci, c in enumerate(elites) for i in targets
                 if i not in c.costs_by_inst and 0 < i <= len(self.instances)]
        if tasks:
            print(f"  [elite-rescore] shared set < {floor}; topping up {len(tasks)} "
                  f"(elite, instance) evals to align {len(elites)} elites on {floor} "
                  f"shared instances", flush=True)
            results = self._map_scores_with_timeout(tasks)
            for (_ci, c, inst_idx, _inst), (cost, cpu) in zip(tasks, results):
                c.record(inst_idx, cost)
                self.budget_used += 1
                self._cpu_seconds_used += cpu
            self._topup_evals_gen += len(tasks)
            self._append_perf_log(elites, targets, gen_id=self.generation + 1)
        return set(targets)

    # ---- Manuscript structured logging ----------------------------------

    def _instance_meta(self, inst_idx) -> dict:
        """Resolve a 1-based race inst_idx to BBOB instance metadata for the trace."""
        if not inst_idx or not (0 < inst_idx <= len(self.instances)):
            return {}
        d, seed = _unwrap(self.instances[inst_idx - 1])
        rep = getattr(self.instances[inst_idx - 1], "rep", 0)
        return {"instance_id": f"f{int(d['fid'])}i{int(d['iid'])}r{rep}",
                "fid": int(d["fid"]), "iid": int(d["iid"]),
                "seed": (int(seed) if seed is not None else None), "rep": int(rep)}

    def _log_manuscript(self, gen_id, combined, offspring_cfgs, out, t_race) -> None:
        """Emit race_step_trace / race_summary / candidate_log / diversity_log for
        this generation. Best-effort: never break the search loop."""
        try:
            from utils.manuscript_log import code_hash
            ml = self._mlog
            elite_ids = {c.id for c in self._elite_cfgs}

            def _timed(c):
                return bool(getattr(c, "timed_out_insts", None))

            def _status(c):
                if c.alive:
                    return "survived_elite" if c.id in elite_ids else "survived"
                if c.id in elite_ids:
                    return "carried_diversity"
                if _timed(c):
                    return "timeout"
                return "crashed" if self._is_crashed(c) else "eliminated"

            crashed = sum(1 for c in combined if (not c.alive) and self._is_crashed(c) and not _timed(c))
            timeout = sum(1 for c in combined if _timed(c))
            eliminated = sum(1 for c in combined if (not c.alive) and not self._is_crashed(c) and not _timed(c))

            ml.log_race_steps(gen_id, out.get("step_trace", []), instance_meta=self._instance_meta)

            inc = self.incumbents_mean_rank[-1] if self.incumbents_mean_rank else {}
            # Refill (elites kept that didn't survive), cumulative budget/LLM,
            # both incumbent rules. self._elite_cfgs is already the post-refill elite set.
            _surv_ids = {c.id for c in out.get("survivors", [])}
            _refilled = [c for c in self._elite_cfgs if c.id not in _surv_ids]
            _refill_keys = {c.id: (c.mean_cost if math.isfinite(c.mean_cost) else float("inf"))
                            for c in _refilled}
            _alive_elites = [c for c in self._elite_cfgs if c.alive]
            try:
                _ir = select_incumbent(_alive_elites, "mean_rank")
                _ic = select_incumbent(_alive_elites, "mean_cost")
            except Exception:
                _ir = _ic = None
            _cached = getattr(self.llm, "_cached", None) or self.llm
            ml.log_race_summary(ml.race_summary_record(
                gen_id=gen_id, n_candidates_init=len(combined),
                n_survivors=len(out.get("survivors", [])),
                n_refilled=len(_refilled),
                refilled_ids=[c.id for c in _refilled], refill_keys=_refill_keys,
                instances_evaluated_max=len(out.get("seen_instances", [])),
                total_evaluations_spent=int(out.get("experiments_used", 0)),
                budget_consumed_cum=int(getattr(self, "budget_used", 0)),
                topup_evaluations=int(self._topup_evals_gen),
                stop_reason=out.get("break_reason", ""),
                race_wall_seconds=time.time() - t_race,
                cpu_seconds=float(out.get("cpu_seconds", 0.0)),
                llm_calls_cum=int(getattr(_cached, "total_calls", 0) or 0),
                llm_prompt_tokens_cum=int(getattr(_cached, "total_prompt_tokens", 0) or 0),
                llm_completion_tokens_cum=int(getattr(_cached, "total_completion_tokens", 0) or 0),
                incumbent_cand_id=inc.get("cand_id"), incumbent_mode="both",
                incumbent_rank_id=(getattr(_ir, "cfg", _ir).id if _ir is not None else None),
                incumbent_cost_id=(getattr(_ic, "cfg", _ic).id if _ic is not None else None),
                incumbent_partial_score=inc.get("score"),
                incumbent_coverage=int(inc.get("n_instances", 0)),
                crashed_count=crashed, timeout_count=timeout, eliminated_count=eliminated))

            # LLaMEA parent_ids are Solution UUIDs; translate to cand_id space so
            # the genealogy joins to candidate_log.cand_id (lineage trees, Dim 4).
            uuid2cand = {getattr(s, "id", None): cid for cid, s in self._sol_by_id.items()}
            for c in offspring_cfgs:
                sol = self._sol_by_id.get(c.id)
                _pids = getattr(sol, "parent_ids", None) if sol else None
                _pids = [uuid2cand.get(p, p) for p in _pids] if _pids else _pids
                ml.log_candidate({
                    "cand_id": c.id, "gen_born": gen_id,
                    "parent_ids": _pids,
                    "operator": getattr(sol, "operator", None) if sol else None,
                    "status": _status(c),
                    "n_instances_evaluated": len(c.costs_by_inst),
                    "lifetime_mean_cost": (c.mean_cost if math.isfinite(c.mean_cost) else float("inf")),
                    "is_elite_carried": c.id in elite_ids,
                    "elite_credit_span": int(getattr(c, "is_elite_credit", 0)),
                    "code_hash": code_hash(c.source),
                })

            alive_elites = [c for c in self._elite_cfgs if c.alive]
            dead_elites = [c for c in self._elite_cfgs if not c.alive]
            srcs = [c.source for c in alive_elites]
            fits = [(self._sol_by_id.get(c.id).fitness if self._sol_by_id.get(c.id) else float("nan"))
                    for c in alive_elites]
            ml.log_diversity(ml.diversity_record(
                gen_id=gen_id, survivor_sources=srcs, survivor_fitnesses=fits,
                crashed_fraction=(crashed + timeout) / max(1, len(combined)),
                eliminated_carried_count=len(dead_elites)))
        except Exception as e:
            print(f"  [manuscript-log] WARN: gen {gen_id} logging failed: {e}", flush=True)

    # ---- Stop condition (OVERRIDE) --------------------------------------

    def _stop(self) -> bool:
        if self.budget_cap is not None and self.budget_used >= self.budget_cap:
            print(f"  [budget cap reached] {self.budget_used}/{self.budget_cap} — stopping", flush=True)
            return True
        if self.max_generations is not None and self._evo_gens >= self.max_generations:
            return True
        return False

    # ---- Best tracking + logging (OVERRIDE) -----------------------------

    def _update_best(self, cfgs: List[ConfigAS]) -> None:
        """Pre-race placeholder for ``self.best`` (used ONLY on the pre-evaluated
        initial population, where every parent shares the same t_first coverage so a
        raw-fitness argmax is a valid ranking). After generation 1, ``self.best`` is
        advanced coverage-aware from the sum_ranks incumbent in ``_record_generation``
        — NOT by raw argmax over offspring — so a lucky under-evaluated candidate can
        never be reported as the best (matches eoh_tsp_gls._finalize_best)."""
        for c in cfgs:
            sol = self._sol_by_id.get(c.id)
            if sol is None or not math.isfinite(sol.fitness):
                continue
            if self.best is None or sol.fitness > self.best.fitness:
                self.best = sol

    def _record_generation(self, gen_id: int, batch_cfgs: List[ConfigAS]) -> None:
        """Log the newly-evaluated batch (heuristics.json) + best-so-far
        incumbent (trajectory.json). used_budget is the ACTUAL race evaluation
        count (partial), not a full-grid count."""
        # This generation's incumbent = rank-best surviving elite (coverage-tiered
        # sum_ranks). ``mean_cost`` is over its LIFETIME costs_by_inst (elites carry
        # results across races), so score = -mean_cost = lifetime mean AOCC.
        # Incumbent identification (switchable via --incumbent-mode): "mean_rank"
        # (default) = _elite_cfgs[0] (irace overall_ranks best); "mean_cost" = best
        # lifetime mean objective. Shared with RacingBase via select_incumbent.
        # DEAD candidates (carried only for --save-pop offspring sampling) are
        # barred from being the incumbent — the incumbent must be an alive survivor
        # (matters for mean_cost, where an early-eliminated candidate could otherwise
        # win on a stale low mean cost; mean_rank already sorts survivors first).
        _alive = [c for c in self._elite_cfgs if c.alive]

        def _inc_row(ic):
            aocc = (-ic.mean_cost) if (ic is not None and math.isfinite(ic.mean_cost)) else float("-inf")
            return {
                "gen_id": int(gen_id),
                "cand_id": ic.id if ic is not None else None,
                "used_budget": int(self.budget_used),
                "cpu_seconds": round(self._cpu_seconds_used, 3),
                "score": float(aocc),        # lifetime mean AOCC (higher = better)
                "n_instances": int(len(ic.costs_by_inst) if ic is not None else 0),
            }
        # Record BOTH incumbent rules every generation (--incumbent-mode was dropped);
        # eval scripts read incumbents_mean_rank / incumbents_mean_cost under --has-incumbent.
        self.incumbents_mean_rank.append(_inc_row(select_incumbent(_alive, "mean_rank")))
        self.incumbents_mean_cost.append(_inc_row(select_incumbent(_alive, "mean_cost", announce_gen=gen_id)))
        # The trajectory row is driven by the mean_rank incumbent (historical default).
        inc = select_incumbent(_alive, "mean_rank")
        inc_aocc = (-inc.mean_cost) if (inc is not None and math.isfinite(inc.mean_cost)) else float("-inf")
        inc_n = len(inc.costs_by_inst) if inc is not None else 0

        # Trajectory row. NO across-generation comparison for EITHER mode: report
        # THIS generation's incumbent directly (regress-allowed), overriding the
        # old best. Early incumbents are evaluated on FEWER instances, so comparing
        # their mean score against later, better-evaluated incumbents is unreliable
        # — a low-coverage early fluke could otherwise freeze the best-so-far. So
        # the trajectory just mirrors the per-generation `incumbents` series.
        # ``incumbent_mode`` only selects WHICH survivor is the incumbent
        # (mean_rank = irace rank-best; mean_cost = lifetime-cost-best); both are
        # coverage-protected within their generation (rank tiering / validity filter).
        self._best_aocc = inc_aocc
        self._best_cand_id = inc.id if inc is not None else self._best_cand_id
        self._best_n_instances = inc_n
        if inc is not None:
            self.best = self._sol_by_id.get(inc.id, self.best)
        row = {
            "gen_id": int(gen_id),
            "cand_id": self._best_cand_id,
            "score": float(self._best_aocc),
            "n_instances": int(self._best_n_instances),
            "used_budget": int(self.budget_used),
            "cpu_seconds": round(self._cpu_seconds_used, 3),
        }
        self._traj_rows.append(row)
        if self.wandb_logger:
            self.wandb_logger.log_trajectory_row(row)

        # heuristics.json: score each candidate by its (partial) AOCC.
        score_by_id = {}
        for h in self._heur_rows:
            c = self._cfg_by_id.get(h["cand_id"])
            score_by_id[h["cand_id"]] = (-c.mean_cost if (c is not None and math.isfinite(c.mean_cost) and not any(v >= 1e5 for v in c.costs_by_inst.values()))
                                         else float("-inf"))
        heur_out = [{"cand_id": h["cand_id"], "gen_id": int(h["gen_id"]),
                     "score": score_by_id.get(h["cand_id"]), "source": h["source"]}
                    for h in self._heur_rows]

        json.dump({"label": self.label, "trajectory": self._traj_rows,
                   "incumbents_mean_rank": self.incumbents_mean_rank,
                   "incumbents_mean_cost": self.incumbents_mean_cost},
                  open(self.log_dir / "trajectory.json", "w"), indent=2)
        json.dump({"label": self.label, "total_sampled": len(heur_out), "heuristics": heur_out},
                  open(self.log_dir / "heuristics.json", "w"), indent=2)
        print(f"  [trajectory] gen={gen_id}  best_AOCC={self._best_aocc:.4f}  "
              f"used_budget={self.budget_used}", flush=True)

    def _append_perf_log(self, cfgs: List[ConfigAS], seen, gen_id: Optional[int] = None) -> None:
        from utils.manuscript_log import perf_row_extras
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        idx_to_task = {i + 1: t for i, t in enumerate(self.instances)}
        big = float(getattr(self, "_big_penalty", 1e5))
        lines = []
        for c in cfgs:
            for task_idx, cost in c.costs_by_inst.items():
                key = (c.id, task_idx)
                if key in self._perf_logged:
                    continue
                self._perf_logged.add(key)
                task = idx_to_task.get(task_idx)
                base_idx, seed, rep = (task.base_idx, task.seed, task.rep) if isinstance(task, SeededInstance) else (task_idx - 1, None, 0)
                cc = float(cost)
                # Extras: status/race_step/wall_s (from meta_by_inst). LLaMEA does
                # not track the race prefix order here, so instance_idx_k stays None.
                extras = perf_row_extras(c, task_idx, cost, [], big)
                lines.append(json.dumps({
                    # `gen` mirrors the EoH runners' schema; the analysis replay needs
                    # it to scope candidates to a generation (task_idx is reused across
                    # generations, so instance identity alone cannot recover it).
                    "gen": int(gen_id if gen_id is not None else self.generation + 1),
                    "cand_id": c.id, "task_idx": int(task_idx), "base_idx": int(base_idx),
                    "instance_idx_k": extras["instance_idx_k"], "race_step": extras["race_step"],
                    "rep": int(rep), "seed": (int(seed) if seed is not None else None),
                    "cost": (cc if math.isfinite(cc) else str(cc)), "aocc": (-cc if math.isfinite(cc) else None),
                    "status": extras["status"], "wall_s": extras["wall_s"],
                    "ts": ts,
                }))
        if lines:
            with open(self._perf_log_path, "a") as f:
                f.write("\n".join(lines) + "\n")

    # ---- Main loop (OVERRIDE): the race-based ES ------------------------

    def run(self):
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        print(f"[{self.label}] racing LLaMEA (budget_cap={cap_str}, max_generations={gen_str}, "
              f"T_first={self.t_first}, T_each={self.t_each}, mu={self.n_parents}, "
              f"lambda={self.n_offspring}, race_elitist={self.race_elitist}, "
              f"llamea_elitism={self.llamea_elitism})", flush=True)
        print_instance_order(self.instances, self._deterministic,
                             getattr(self, "seed", None))

        # Initial population: sampled (gen 0) without pre-evaluation.
        # Initial mu parents become the starting elites for Gen 1 offspring sampling.
        # Gen 1's race then evaluates the 2*mu combined candidates (mu parents + mu offspring)
        # directly from Instance 1.
        init_sols = self._initialize_population()
        init_cfgs = [self._materialize(s, gen=0) for s in init_sols]
        self._elite_cfgs = init_cfgs[: self.n_parents]
        self._update_best(init_cfgs)
        next_instance = 1
        self.generation = 0
        print(f"  gen 00: sampled {len(init_cfgs)} initial parents; "
              f"used_budget={self.budget_used}", flush=True)

        while not self._stop():
            gen_id = self.generation + 1
            elite_sols = [self._sol_by_id[c.id] for c in self._elite_cfgs]
            self.population = elite_sols  # construct_prompt reads this + solution.fitness
            offspring_sols = self._sample_offspring(elite_sols, gen=gen_id)
            offspring_cfgs = [self._materialize(s, gen=gen_id) for s in offspring_sols]
            if not offspring_cfgs:
                print(f"  gen {gen_id:02d}: 0 offspring — stopping", flush=True)
                break

            # 2*pop_size candidates entering the race (mu parents + lambda offspring)
            combined = (self._elite_cfgs + offspring_cfgs) if self.llamea_elitism else list(offspring_cfgs)
            t_race = time.time()
            out = self._race(combined, next_instance)
            next_instance = out["next_instance"]
            survivors = out["survivors"]
            if self.save_pop:
                # Maintain fixed elite pool size of mu (n_parents) across all generations.
                # Top up valid survivors with best-ranked eliminated/crashed candidates.
                self._elite_cfgs = (out["all_records_sorted"][: self.n_parents]
                                    or self._elite_cfgs[: self.n_parents])
                if len(survivors) < self.n_parents:
                    n_topup = len(self._elite_cfgs) - len(survivors)
                    print(f"  [save-diversity] WARNING: alive candidates ({len(survivors)}) < mu ({self.n_parents}) "
                          f"-> topped up {n_topup} slots with best-ranked eliminated/crashed candidates", flush=True)
                elite_ids = [c.id for c in self._elite_cfgs]
                print(f"  [save-diversity] elite pool ({len(self._elite_cfgs)} candidates) carried to gen {gen_id + 1:02d}: {elite_ids}", flush=True)
            else:
                self._elite_cfgs = survivors[: self.n_parents] or self._elite_cfgs[: self.n_parents]
            # Comparable LLM-facing fitness: re-score the new elite pool on the
            # instances all elites SHARE, so the next generation's prompt + parent
            # selection compare like-for-like — NOT each elite's mean over its own
            # disjoint instance history (the signal that misled the LLM).
            self._rescore_elites_shared()
            self.generation = gen_id
            self._evo_gens += 1
            # self.best is advanced coverage-aware inside _record_generation (from the
            # sum_ranks incumbent), NOT by a raw argmax over `combined` here.
            self._record_generation(gen_id, offspring_cfgs)
            self._log_manuscript(gen_id, combined, offspring_cfgs, out, t_race)
            print(f"  gen {gen_id:02d}/{gen_str}: pool={len(combined)} "
                  f"survivors={len(out['survivors'])} elites={len(self._elite_cfgs)}; "
                  f"budget={self.budget_used}/{cap_str}; race_wall={time.time() - t_race:.1f}s "
                  f"({out['break_reason']})", flush=True)

        self._shutdown_pool()
        print(f"[{self.label}] done: {self._evo_gens} evolution generations, "
              f"final budget={self.budget_used}, best AOCC={self._best_aocc:.4f}", flush=True)
        valid = self._final_eval()   # {"mean_rank": ..., "mean_cost": ...}
        # Manuscript reliability pairing per incumbent rule (partial vs. full held-out).
        for rule in ("mean_rank", "mean_cost"):
            try:
                self._mlog.finalize_reliability(getattr(self, f"incumbents_{rule}"),
                                                (valid or {}).get(rule), rule=rule)
            except Exception as e:
                print(f"  [manuscript-log/{rule}] WARN: reliability finalize failed: {e}", flush=True)
        return self.best

    def _final_eval(self) -> dict:
        """Validate BOTH incumbent rules via eval_bbob/eval_homo_bbob ->
        valid_trajectory_mean_rank/mean_cost.json."""
        ipm = getattr(self, "instance_pool_mode", "hetero").lower()
        if ipm == "homo":
            from analyses.eval_homo_bbob import evaluate_single as eval_fn, _to_serialisable
            mode_name = "eval_homo_bbob.py"
        else:
            from analyses.eval_bbob import evaluate_single as eval_fn, _to_serialisable
            mode_name = "eval_bbob.py"
        print(f"\n=== Post-Evolution Final Evaluation ({mode_name}) ===", flush=True)
        dim = getattr(self, "dim", 5)
        budget_factor = getattr(self, "budget_factor", 2000)
        n_rep = getattr(self, "n_reps", getattr(self, "n_rep", 1))
        eval_timeout = getattr(self, "_eval_timeout", -1.0)
        if eval_timeout is None:
            eval_timeout = -1.0
        results: dict = {}
        for rule in ("mean_rank", "mean_cost"):
            try:
                res = eval_fn(
                    exp_path=self.log_dir, dim=dim, budget_factor=budget_factor,
                    n_rep=n_rep, eval_timeout=eval_timeout, n_cores=self.num_cores,
                    mode="valid", has_incumbent=True, incumbent_key=f"incumbents_{rule}")
            except Exception as e:
                print(f"  [_final_eval/{rule}] skipped: {type(e).__name__}: {e}", flush=True)
                continue
            out_path = self.log_dir / f"valid_trajectory_{rule}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2, default=_to_serialisable)
            print(f"Saved {rule} validation -> {out_path}", flush=True)
            results[rule] = res
        return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "llm_model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "llm_backend": "vllm",
    "temperature": 0.8,
    "n_parents": 10,
    "n_offspring": 10,
    "max_generations": -1,
    "ref_max_generations": 20,
    "dim": 5,
    "budget_factor": 2000,
    "n_reps": 1,
    "t_first": 24,
    "t_each": 2,
    "alpha": 0.05,
    "test_type": "friedman",
    "posthoc_test_type": "conover",
    "race_elitist": True,
    "llamea_elitism": True,
    "save_pop": False,
    "deal_with_crashed": "rejection",
    "crash_penalty": 1e6,
    "elitist_new_instances": 1,
    "elitist_limit": 12,
    "eval_timeout": -1.0,
    "parent_selection": "random",
    "tournament_size": 3,
    "num_threads": 4,
    "num_cores": 4,
    "prompt_mode": "partial_eval",
    "instance_pool_mode": "hetero",
    "deterministic": False
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "llm_model": "model", "llm_backend": "llm", "temperature": "temp",
    "n_parents": "mu", "n_offspring": "lam", "max_generations": "mg",
    "ref_max_generations": "rmg", "dim": "d", "budget_factor": "bf", "n_reps": "nrep",
    "t_first": "tf", "t_each": "te", "alpha": "a", "test_type": "tt",
    "posthoc_test_type": "phtt",
    "race_elitist": "relit", "llamea_elitism": "eselit", "save_pop": "savpop",
    "deal_with_crashed": "dwc",
    "crash_penalty": "cp",
    "elitist_new_instances": "tni", "elitist_limit": "elimit", "eval_timeout": "et", "parent_selection": "ps",
    "tournament_size": "ts", "num_threads": "nt", "num_cores": "nc",
    "prompt_mode": "pm",
    "instance_pool_mode": "ipm",
    "deterministic": "det"
}


def _run_tag(args: argparse.Namespace) -> str:
    overrides = []
    for key, default_val in _BASH_DEFAULTS.items():
        actual = getattr(args, key, None)
        if actual == default_val:
            continue
        short = _ABBREV.get(key, key)
        if key == "llm_model" and isinstance(actual, str) and "/" in actual:
            actual = actual.split("/")[-1]
        overrides.append((short if actual else f"no{short}") if isinstance(actual, bool)
                         else f"{short}{actual}")
    return "_".join(overrides) if overrides else "default"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Racing LLaMEA on BBOB (LLaMEA framework + F-race).")
    p.add_argument("--n-parents", type=int, default=10, help="mu: elites carried between generations.")
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed generation 0 from src/init_pop/llamea_24_bbob.json (re-raced "
                        "this run) instead of sampling the initial population from the LLM.")
    p.add_argument("--n-offspring", type=int, default=10, help="lambda: offspring sampled per generation.")
    p.add_argument("--max-generations", type=int, default=-1, help="Evolution-generation cap. -1 = disable.")
    p.add_argument("--ref-max-generations", type=int, default=20,
                   help="Reference generations for the default budget cap (mu * ref_gens * n_instances).")
    p.add_argument("--n-reps", "--n-rep", type=int, default=_BASH_DEFAULTS["n_reps"], help="Repeated seeded runs per instance (default 1).")
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--budget-factor", type=int, default=2000)
    p.add_argument("--t-first", type=int, default=_BASH_DEFAULTS["t_first"])
    p.add_argument("--t-each", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--test-type", type=str, default="friedman", choices=["friedman", "ttest"])
    p.add_argument("--posthoc-test-type", type=str, default="conover", choices=["conover", "nemenyi"], help="Friedman post-hoc test variant. 'conover' (default): original R irace formula (Conover 1999) vs best_idx only. 'nemenyi': scikit-posthocs all-pairs Nemenyi test.")
    p.add_argument("--no-race-elitist", dest="race_elitist", action="store_false", default=True,
                   help="Disable the irace elitist F-race (carry-over of elite costs).")
    p.add_argument("--no-llamea-elitism", dest="llamea_elitism", action="store_false", default=True,
                   help="(mu, lambda) ES instead of (mu + lambda): race only the offspring.")
    p.add_argument("--save-pop", action="store_true", default=False,
                   help="When a race yields fewer survivors than mu, refill the elite pool to mu "
                        "with the best-ranked ELIMINATED candidates instead of shrinking it "
                        "(preserves parent-pool diversity).")
    p.add_argument("--elitist-new-instances", type=int, default=1)
    p.add_argument("--elitist-limit", type=int, default=12, help="In elitist irace, maximum number per race of elimination tests that do not eliminate a configuration. 0 = no limit.")
    p.add_argument("--eval-timeout", type=float, default=-1.0,
                   help="Per-(config, instance) wall-clock cap (s), enforced by elitist_race when "
                        "--num-cores > 1 (Future.result(timeout) + pool kill/rebuild). Guards against "
                        "LLM heuristics that infinite-loop without calling func; a timed-out eval is "
                        "penalised with --timeout-cost (finite, not rejected). -1 (default) = AUTO-SCALE "
                        "with budget = 60 + budget_factor*dim/100 (dim 5 -> 160s, dim 20 -> 460s); "
                        "0 = disabled; >0 = fixed seconds.")
    p.add_argument("--timeout-cost", type=float, default=BIG_PENALTY)
    p.add_argument("--parent-selection", type=str, default="random",
                   choices=["random", "tournament", "roulette"])
    p.add_argument("--tournament-size", type=int, default=3)
    p.add_argument("--instance-pool-mode", type=str, default=_BASH_DEFAULTS["instance_pool_mode"], choices=["hetero", "homo"], help="Instance pool mode: 'hetero' (default: 24 BBOB functions of 5 dim, 3 iids) or 'homo' (single BBOB function Rastrigin fid 3 across dims {5, 10, 20}, 3 iids).")
    p.add_argument("--prompt-mode", type=str, default="partial_eval",
                   choices=["original", "partial_eval"],
                   help="LLM population-context prompt variant. 'original' = plain LLaMEA "
                        "(bare average score per candidate). 'partial_eval' (default) = also "
                        "expose each candidate's instance count + real BBOB fids, so the LLM "
                        "can weigh partial-evaluation reliability. Every prompt is logged to "
                        "llm_prompts.jsonl regardless of mode.")
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-cap", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--label", type=str, default="racing_llamea_bbob")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=1024)
    p.add_argument("--llm-backend", type=str, default="vllm",
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"])
    p.add_argument("--llm-model", type=str, default="mistralai/Devstral-Small-2-24B-Instruct-2512")
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--num-cores", type=int, default=4)
    p.add_argument("--crash-penalty", type=float, default=BIG_PENALTY,
                   help="Penalty cost assigned to crashed candidates when --deal-with-crashed penalty is active.")
    p.add_argument("--deal-with-crashed", type=str, default="rejection", choices=["rejection", "penalty"],
                   help="How the race treats a crashed candidate (non-finite/inf cost). "
                        "'rejection' (default): faithful irace — drop it immediately "
                        "(strip elite protection, remove from the alive set). 'penalty': keep "
                        "it alive with a finite worst-case cost so it undergoes the statistical "
                        "test — preserves a fragile population from collapsing when heuristics "
                        "crash often.")
    p.add_argument("--use-wandb", action="store_true", default=False)
    return p.parse_args(argv)


def _build_llm(args, cache_dir):
    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "mistral":
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
    elif args.llm_backend == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set in environment / .env")
        client = GoogleClient(timeout=args.llm_timeout, model=args.llm_model)
    else:
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
    cached = CachedLLM(client, cache_dir=cache_dir)
    return LLaMEA_LLM_Adapter(cached, model_name=cached.client.model, temperature=args.temperature)


def main(argv=None) -> int:
    main_t0 = time.time()
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)

    now = _dt.datetime.now()
    dt_stamp = args.run_stamp or f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"
    log_dir = make_log_dir(args.log_root, args.label, dt_stamp, args.seed, tag=_run_tag(args))
    cache_dir = args.cache_root / dt_stamp / str(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    term_log_path = log_dir / "terminal.txt"
    term_log_file = open(term_log_path, "w", buffering=1)
    _orig_out, _orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_out, term_log_file)
    sys.stderr = _Tee(_orig_err, term_log_file)

    def _restore():
        sys.stdout, sys.stderr = _orig_out, _orig_err
        term_log_file.close()
    atexit.register(_restore)
    print(f"terminal output mirrored -> {term_log_path}")

    max_generations = None if args.max_generations is not None and args.max_generations < 0 else args.max_generations
    n_unique = (len(_HOMO_DIMS) * len(_IIDS)) if args.instance_pool_mode == "homo" else _N_UNIQUE_INSTANCES
    if args.budget_cap is None:
        budget_cap = args.n_parents * args.ref_max_generations * n_unique
    else:
        budget_cap = None if args.budget_cap < 0 else args.budget_cap
    if max_generations is None and budget_cap is None:
        print("error: at least one of --max-generations or --budget-cap must be >= 0", file=sys.stderr)
        return 2

    # Resolve --eval-timeout: -1 => auto-scale with budget; 0 => disabled; >0 => fixed.
    if args.eval_timeout is not None and args.eval_timeout < 0:
        eval_timeout = _auto_eval_timeout(args.budget_factor, args.dim)
        print(f"  [eval-timeout] auto = 60 + budget_factor({args.budget_factor}) x dim({args.dim})/100 "
              f"= {eval_timeout:.0f}s per (config, instance)")
    else:
        eval_timeout = args.eval_timeout

    llm = _build_llm(args, cache_dir)

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    args_dict["llm_model"] = llm._cached.client.model
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args)}"
    from utils.manuscript_log import write_run_meta as _wrm
    _wrm(log_dir, {
        "run_id": run_name, "framework": "LLaMEA", "domain": "BBOB",
        "policy": ("raceAD" if getattr(args, "elitist", True) else "race-nonelitist"),
        "seed": args.seed,
        "M": getattr(args, "n_parents", getattr(args, "pop_size", None)),
        "N": getattr(args, "n_instances", None), "K": getattr(args, "n_instances", None),
        "alpha": getattr(args, "alpha", None), "test_type": getattr(args, "test_type", None),
        "posthoc": getattr(args, "posthoc_test_type", None),
        "T_first": getattr(args, "t_first", None), "T_each": getattr(args, "t_each", None),
        "elimit": getattr(args, "elitist_limit", None),
        "refill": bool(getattr(args, "save_pop", getattr(args, "save_diversity", False))),
        "incumbent_rule": "both",
        "B": budget_cap, "llm_model": args_dict.get("llm_model"),
    })
    wandb_logger = make_wandb_logger(enabled=args.use_wandb, project="llm4ad", name=run_name, config=args_dict)

    print(f"instance_pool_mode={args.instance_pool_mode}  pop: mu={args.n_parents} lambda={args.n_offspring}  dim={args.dim}  "
          f"budget_cap={'off' if budget_cap is None else budget_cap}  "
          f"max_generations={'off' if max_generations is None else max_generations}")

    fixed_init_population = None
    if args.fix_init_pop:
        from utils.fixed_init_pop import load_fixed_initial_population
        _fp = ROOT / "src" / "init_pop" / "llamea_24_bbob.json"
        fixed_init_population = load_fixed_initial_population(_fp, args.n_parents)
        print(f"  [fixed-init-pop] loaded {len(fixed_init_population)} heuristics from {_fp}",
              flush=True)

    RacingLLaMEA(
        llm=llm, log_dir=log_dir, label=args.label,
        n_parents=args.n_parents, n_offspring=args.n_offspring,
        dim=args.dim, budget_factor=args.budget_factor, n_reps=args.n_reps, seed=args.seed,
        max_generations=max_generations, budget_cap=budget_cap,
        t_first=args.t_first, t_each=args.t_each, alpha=args.alpha, test_type=args.test_type, posthoc_test_type=args.posthoc_test_type,
        race_elitist=args.race_elitist, elitist_new_instances=args.elitist_new_instances, elitist_limit=args.elitist_limit,
        save_pop=args.save_pop,
        deal_with_crashed=args.deal_with_crashed,
        crash_penalty=args.crash_penalty,
        llamea_elitism=args.llamea_elitism, eval_timeout=eval_timeout,
        timeout_cost=args.timeout_cost, deterministic=args.deterministic,
        parent_selection=args.parent_selection, tournament_size=args.tournament_size,
        num_threads=args.num_threads, num_cores=args.num_cores,
        prompt_mode=args.prompt_mode, instance_pool_mode=args.instance_pool_mode,
        fixed_init_population=fixed_init_population, wandb_logger=wandb_logger,
    ).run()

    wandb_logger.finish()
    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
