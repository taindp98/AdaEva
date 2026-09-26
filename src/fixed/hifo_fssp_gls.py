"""Random Policy (Fixed-K) evaluation for HiFo-Prompt on FSSP-GLS.

The FSSP-GLS counterpart of ``tiny/hifo_obp.py`` / ``tiny/hifo_tsp_gls.py`` (and the HiFo
counterpart of ``tiny/eoh_fssp_gls.py``). Like ``tiny/eoh_fssp_gls.py`` it evaluates every
newly generated heuristic on exactly K instances (a fixed-allocation baseline vs the
UCB-based MAB / racing), but it drives the **HiFo-Prompt package's native engine**
(InsightPool hindsight + EvolutionaryNavigator foresight + the ``m3`` operator)
instead of LLM4AD's EoH.

It is ``reprod/hifo_fssp_gls.py`` with the two ``tiny`` changes (mirroring
``tiny/hifo_tsp_gls.py``):

  * Fixed-K evaluation: ``_FSSPAdapter`` restricts each candidate's GLS scoring to a
    K-instance subset of the full ``--n-instances`` pool. ``--instance-mode fixed``
    (default) samples one K-subset once (seeded by ``--seed``) and reuses it for every
    heuristic; ``--instance-mode random`` draws a fresh K-subset per heuristic (seeded
    reproducibly by the heuristic source). The subset is selected per candidate in the
    adapter, which submits only those K instances to the per-(heuristic, instance)
    evaluator pool (see below).

  * Budget-matched generations: reprod/hifo spends ``(2 + max_gen*n_op)*pop*n_instances``
    instance-evals (HiFo produces ``n_op*pop`` offspring per generation plus 2 init
    batches). To spend the SAME budget while evaluating each candidate on only K
    instances, HiFo's ``ec_n_pop`` is scaled up to ``eff_gen`` where
    ``(2 + eff_gen*n_op)*pop*K == (2 + max_gen*n_op)*pop*n_instances`` (so ``eff_gen ==
    max_gen`` when ``K == n_instances``). Same budget-match as ``tiny/hifo_obp.py``.

Everything else — the CachedLLM bridge, Layer-B HiFo internals, timeout decoupling, the
per-(heuristic, instance) ``ProcessPoolExecutor(num_cores)`` GLS evaluator (each of a
candidate's K subset instances is a task capped by a PER-INSTANCE ``--eval-timeout`` ~65s,
with pool rebuild on a hang; num_cores parallelises the K instances), real per-generation
``cpu_seconds`` (from eval_cpu.jsonl), and the eoh-style ``trajectory.json`` /
``heuristics.json`` log watcher — mirrors ``reprod/hifo_fssp_gls.py``. FSSP has NO known
optimum, so — like ``tiny/eoh_fssp_gls.py`` — NO gap is reported (raw mean makespan). A
post-evolution ``_final_eval`` re-scores the incumbent on the full pool via
``analyses/eval_fssp.py`` (as in ``tiny/eoh_fssp_gls.py``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import concurrent.futures
import pathlib
import random
import sys
import threading
import time
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "packages" / "HiFo-Prompt" / "hifo" / "src"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, make_wandb_logger, OpenRouterClient, OllamaClient,
                   MistralClient, vLLMClient)
from utils.logger import make_log_dir, mirror_stdout_to

from llm4ad.task.optimization.fssp_gls.evaluation import FSSP_GLS_Evaluation
from llm4ad.task.optimization.fssp_gls.get_instance import GetData
import llm4ad.task.optimization.fssp_gls.evaluation as _fssp_eval_mod

# FSSP has NO known optimum (unlike TSP's Concorde optimum), so — matching
# tiny/eoh_fssp_gls.py — no gap is reported; the trajectory tracks raw mean makespan.
# GLS caps overridden to match tiny/eoh_fssp_gls.py: 60s/instance, 1000 iters. These are
# module globals read by ``solve_without_time`` at call time, so they MUST be re-set
# inside each worker process (see ``_fssp_worker_init``).
_FSSP_TIME_MAX = 60.0
_FSSP_ITER_MAX = 1000

# HiFo-Prompt package (native engine)
from hifo.hifo import EVOL
from hifo.utils.getParas import Paras
import hifo.methods.hifo.hifo_evolution as _hifo_evolution
import hifo.methods.hifo.hifo_hp as _hifo_hp


# Bash-script defaults (mirror tiny/eoh_tsp_gls.py) — used for the folder-name tag.
_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "pop_size": 10,
    "max_generations": 20,
    "n_instances": 64,
    "K": 1,
    "instance_mode": "fixed",
    "n_jobs": 50,
    "selection_num": 5,
    "temperature": 0.9,
    "num_threads": 4,
    "num_cores": 4,
    "llm_model": "qwen/qwen3-coder-next",
    "llm_backend": "openrouter",
    # --- Layer B: HiFo method internals (paper defaults; see hifo_hp.py) ----- #
    # Insight Pool (Hindsight)
    "pool_capacity": 30,
    "novelty_threshold": 0.7,
    "selection_count": 3,
    "usage_penalty_weight": 0.1,
    "recency_bonus": 0.2,
    "recency_window": 2,
    "ema_alpha": 0.3,
    "decay_rate": 0.01,
    "probation_usage": 3,
    # Credit-assignment tiers (Eq. 3 intercepts)
    "credit_best": 0.8,
    "credit_inc": 0.2,
    "credit_pen": -0.3,
    # Evolutionary Navigator (Foresight)
    "progress_eps": 1e-4,
    "stagnation_threshold": 3,
    "progress_threshold": 2,
    "diversity_threshold": 0.3,
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "pop_size": "ps",
    "max_generations": "mg",
    "n_instances": "ni",
    "K": "K",
    "instance_mode": "im",
    "n_jobs": "nj",
    "selection_num": "sn",
    "temperature": "temp",
    "num_threads": "nt",
    "num_cores": "nc",
    "llm_model": "model",
    "llm_backend": "llm",
    # Layer B — HiFo internals
    "pool_capacity": "pcap",
    "novelty_threshold": "nov",
    "selection_count": "sc",
    "usage_penalty_weight": "wu",
    "recency_bonus": "rb",
    "recency_window": "rw",
    "ema_alpha": "ema",
    "decay_rate": "dr",
    "probation_usage": "pu",
    "credit_best": "cb",
    "credit_inc": "cinc",
    "credit_pen": "cpen",
    "progress_eps": "peps",
    "stagnation_threshold": "stag",
    "progress_threshold": "prog",
    "diversity_threshold": "divt",
}


def _run_tag(args: argparse.Namespace) -> str:
    """Compact folder-name tag encoding args that differ from the defaults."""
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


# ----------------------------------------------------------------------------- #
# K-instance subset selection (fixed | random) — from fixed/eoh_tsp_gls.py.
# ----------------------------------------------------------------------------- #

def _sample_subset_idx(n_total: int, k: int, seed: int) -> list:
    """K distinct instance indices sampled without replacement from range(n_total),
    reproducible given ``seed`` (sorted for stable, readable logging)."""
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    return sorted(rng.choice(n_total, size=min(int(k), n_total), replace=False).tolist())


def _program_subset_seed(program_str: str, base_seed: int) -> int:
    """Stable per-heuristic seed for 'random' mode. Uses hashlib (NOT Python's
    salted ``hash``) so the same heuristic source yields the same subset across
    worker processes and reruns with the same ``--seed``."""
    h = int(hashlib.md5(program_str.encode("utf-8")).hexdigest(), 16)
    return (h ^ (int(base_seed) & 0xFFFFFFFF)) % (2 ** 32)


# --------------------------------------------------------------------------- #
# Per-(heuristic, instance) evaluation workers (run in the process pool).
#
# The eval is flattened to ONE (candidate, instance) task per pool submission so the
# pool parallelises across INSTANCES (each of a candidate's K subset instances runs in
# its own worker). Mirrors reprod/hifo_tsp_gls / reprod/reevo_tsp_gls. Workers cache the
# FULL instance pool once (initializer); the per-candidate K-subset is selected in the
# adapter, so only ``(code, inst_idx)`` crosses the pickle boundary per task.
# --------------------------------------------------------------------------- #
_W_INSTANCES = None   # per-worker cache of the FULL FSSP instance list (set by initializer)


def _fssp_worker_init(instances):
    global _W_INSTANCES
    _W_INSTANCES = instances
    # ``solve_without_time`` reads the GLS caps from module globals at call time; the
    # worker imports fssp_gls.evaluation fresh (defaults 10s), so re-set them to the
    # tiny/eoh_fssp_gls.py values (60s/1000 iters) IN THE WORKER PROCESS.
    import llm4ad.task.optimization.fssp_gls.evaluation as _m
    _m.time_max = _FSSP_TIME_MAX
    _m.iter_max = _FSSP_ITER_MAX


def _fssp_compile_update(code: str):
    """Compile ``code`` and return the ``get_matrix_and_jobs`` callable (raises if
    absent/unparseable — the aggregator turns that into an invalid candidate)."""
    ns = {"np": np}
    exec(code, ns)
    fn = ns.get("get_matrix_and_jobs")
    if not callable(fn):
        raise ValueError("no 'get_matrix_and_jobs' defined in code")
    return fn


def _eval_one_fssp(args):
    """Evaluate ONE candidate on ONE instance -> ``(makespan, cpu_seconds)`` (cost is
    +inf on a heuristic crash). ``args`` is ``(code, inst_idx)``; the instance comes from
    the worker global. GLS is internally capped at 60s/instance; the OUTER per-instance
    ``result(timeout=...)`` in the parent reclaims a heuristic that infinite-loops inside
    ``get_matrix_and_jobs`` before the 60s check."""
    from llm4ad.task.optimization.fssp_gls.evaluation import solve_without_time
    code, inst_idx = args
    fn = _fssp_compile_update(code)  # exec is ~µs
    t0 = time.process_time()
    c = solve_without_time(_W_INSTANCES[inst_idx], fn)   # inf on internal crash
    cpu = time.process_time() - t0                       # worker CPU for this instance's GLS
    return float(c), float(cpu)


# --------------------------------------------------------------------------- #
# LLM bridge: route HiFo's InterfaceLLM.get_response() through our CachedLLM.
# --------------------------------------------------------------------------- #
class _HiFoLLMBridge:
    """Exposes HiFo's ``get_response(prompt_str) -> str`` over a ``CachedLLM``.

    A per-call incrementing cache salt (like ``utils.llm._draw_sample``) ensures
    identical prompts — notably HiFo's ``i1`` initialisation, where every sample
    shares one prompt — still yield diverse (and separately-cached) completions."""

    def __init__(self, cached: CachedLLM, temperature: float, max_tokens: int,
                 verbose: bool = True):
        self._cached = cached
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._n = 0
        self._inflight = 0
        self._ok = 0
        self._fail = 0
        self._verbose = verbose
        self._lock = threading.Lock()

    def get_response(self, prompt_content) -> str:
        # Every HiFo LLM call funnels through here, so this is the one place to make
        # the otherwise-silent generation loop observable: log start/finish, wall
        # time, response size, running success/fail tally, and in-flight concurrency
        # for each individual query (calls run concurrently across --num-threads, so
        # each line is tagged with its call number #N to stay traceable).
        with self._lock:
            self._n += 1
            n = self._n
            self._inflight += 1
            inflight = self._inflight
        salt = f"hifo-call-{n}"
        prompt_str = str(prompt_content)
        if self._verbose:
            print(f"[llm] #{n} start  (inflight={inflight}, prompt={len(prompt_str)} chars)",
                  flush=True)
        messages = [{"role": "user", "content": prompt_str}]
        t0 = time.time()
        try:
            resp = self._cached.chat(
                messages, temperature=self._temperature,
                max_tokens=self._max_tokens, cache_salt=salt,
            )
        except Exception as e:
            dt = time.time() - t0
            with self._lock:
                self._fail += 1
                self._inflight -= 1
                ok, fail = self._ok, self._fail
            print(f"[llm] #{n} FAILED in {dt:6.1f}s: {type(e).__name__}: {e}  "
                  f"(ok={ok} fail={fail})", flush=True)
            raise
        dt = time.time() - t0
        n_chars = len(resp) if isinstance(resp, str) else 0
        with self._lock:
            self._ok += 1
            self._inflight -= 1
            ok, fail = self._ok, self._fail
        if self._verbose:
            empty = "  [EMPTY RESPONSE]" if n_chars == 0 else ""
            print(f"[llm] #{n} ok     in {dt:6.1f}s ({n_chars} chars){empty}  "
                  f"(ok={ok} fail={fail})", flush=True)
        return resp


# --------------------------------------------------------------------------- #
# Problem adapter: HiFo's problem interface (.prompts + .evaluate) with per-candidate
# Fixed-K subset selection over the per-(heuristic, instance) evaluator pool.
# --------------------------------------------------------------------------- #
class _FSSPPrompts:
    """GetPrompts-compatible prompt spec for FSSP-GLS. The generated function is
    ``get_matrix_and_jobs`` (the GLS perturbation: modify the processing-time matrix and
    select jobs to perturb). Mirrors the LLM4AD ``fssp_gls`` template so HiFo generates
    the same function signature EoH does."""

    def __init__(self):
        self.prompt_task = (
            "Given a flow-shop scheduling problem with n jobs and m machines, design a "
            "novel guided local search perturbation strategy. At each iteration the "
            "strategy modifies the processing-time matrix to expose bottleneck jobs and "
            "returns a short list of jobs to perturb via targeted local search. The goal "
            "is to minimise the final makespan."
        )
        self.prompt_func_name = "get_matrix_and_jobs"
        self.prompt_func_inputs = ["current_sequence", "time_matrix", "m", "n"]
        self.prompt_func_outputs = ["new_matrix", "perturb_jobs"]
        self.prompt_inout_inf = (
            "'current_sequence' is a list of n ints giving the current permutation of job "
            "indices; 'time_matrix' is an n*m Numpy matrix of processing times; 'm' is the "
            "number of machines; 'n' is the number of jobs. Return 'new_matrix', a modified "
            "n*m Numpy processing-time matrix, and 'perturb_jobs', a list of 2-5 job indices "
            "to apply targeted local search on."
        )
        self.prompt_other_inf = (
            "Keep the function signature, inputs and outputs unchanged. 'new_matrix' must "
            "keep the same n*m shape; 'perturb_jobs' must contain valid job indices in "
            "[0, n). Include 'import numpy as np' at the top of the code."
        )

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf


class _FSSPAdapter:
    """HiFo problem object over the FSSP-GLS instance pool with Fixed-K evaluation.

    HiFo calls ``.evaluate(code_string)`` and minimises the returned objective; we return
    the **mean makespan (lower = better)** over the candidate's K-instance subset — or
    ``None`` if any subset instance crashes or times out.

    Parallel evaluation is per-(heuristic, instance), mirroring reprod/hifo_tsp_gls: the
    workers cache the FULL pool (initializer), and ``evaluate()`` submits ONLY the
    candidate's K subset instances as independent tasks to a shared
    ``ProcessPoolExecutor(max_workers=num_cores)``, each capped by a PER-INSTANCE
    ``result(timeout=eval_timeout)``. ``--instance-mode fixed`` uses one K-subset for every
    candidate; ``random`` picks a fresh K-subset per candidate from the heuristic source
    (reproducible). A hung instance is reclaimed in ``eval_timeout`` seconds (pool rebuilt).
    ``evaluate()`` is serialised with a lock (pool-rebuild safety under HiFo's concurrent
    sampler threads); per-candidate CPU (summed across the parallel instance workers) is
    accumulated and appended in candidate order to eval_cpu.jsonl for the trajectory's
    cumulative ``cpu_seconds``."""

    def __init__(self, full_instances, k: int, instance_mode: str, subset_seed: int,
                 fixed_idx=None, num_cores: int = 1, eval_timeout: float = 65.0,
                 log_dir: pathlib.Path = None):
        self.prompts = _FSSPPrompts()
        self._instances = list(full_instances)
        self.n_full = len(self._instances)
        self.k = max(1, int(k))
        self.instance_mode = str(instance_mode)
        self.subset_seed = int(subset_seed)
        self._fixed_idx = list(fixed_idx) if fixed_idx is not None else None
        self.num_cores = max(1, int(num_cores))
        self.eval_timeout = (float(eval_timeout)
                             if eval_timeout and float(eval_timeout) > 0 else None)
        self._pool_init = (_fssp_worker_init, (self._instances,))
        self._eval_pool = None
        self._eval_lock = threading.Lock()   # serialise candidates (pool-rebuild safety)
        self.n_evals = 0
        self._cpu_seconds = 0.0
        self._cpu_log_fh = (open(log_dir / "eval_cpu.jsonl", "a", buffering=1)
                            if log_dir is not None else None)
        self._start_pool()

    def _subset_idx(self, code: str) -> list:
        """The K instance indices this candidate is scored on. ``fixed``: the one subset
        sampled once; ``random``: a fresh subset seeded by the heuristic source."""
        if self.instance_mode == "random":
            seed = _program_subset_seed(code, self.subset_seed)
            return _sample_subset_idx(self.n_full, self.k, seed)
        return self._fixed_idx if self._fixed_idx is not None else list(range(self.k))

    def _record_cpu(self, cand_cpu: float) -> None:
        self._cpu_seconds += float(cand_cpu)
        if self._cpu_log_fh is not None:
            try:
                self._cpu_log_fh.write(json.dumps({"cpu": round(float(cand_cpu), 4)}) + "\n")
            except Exception:
                pass

    def _start_pool(self):
        # Workers cache the FULL instance list once (initializer); tasks ship only
        # (code, inst_idx). Pre-warm in the single-threaded main (before HiFo's sampler
        # threads start) so workers fork from a single-threaded parent.
        self._eval_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_cores,
            initializer=self._pool_init[0], initargs=self._pool_init[1])
        try:
            list(self._eval_pool.map(int, range(self.num_cores)))
        except Exception:
            pass

    def _recreate_pool(self):
        """Terminate the current workers (some may be stuck in an infinite-loop
        heuristic) and start a fresh pool. Mirrors reprod/reevo_tsp_gls._recreate_pool."""
        pool = self._eval_pool
        self._eval_pool = None
        procs = list((getattr(pool, "_processes", None) or {}).values())
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=3)
            if p.is_alive():
                p.kill()
        self._start_pool()
        return self._eval_pool

    def evaluate(self, code_string):
        with self._eval_lock:
            idxs = self._subset_idx(code_string)
            results = self._map_instances_with_timeout(code_string, idxs)
            if results is None:                   # pool broke mid-eval
                self._record_cpu(0.0)
                return None
            costs = [c for (c, _cpu) in results]
            cand_cpu = float(sum(cpu for (_c, cpu) in results))
            self._record_cpu(cand_cpu)            # count CPU even for an invalid candidate
            if any(c is None or not math.isfinite(c) for c in costs):
                return None
            self.n_evals += 1
            return float(np.mean(costs))          # mean makespan over the K subset

    def _map_instances_with_timeout(self, code, idxs):
        """Submit the candidate's K subset GLS tasks; collect each as ``(cost, cpu)`` with
        a per-instance timeout. On a genuine hang, salvage finished siblings, rebuild the
        pool and short-circuit (candidate already invalid). Returns a list of
        ``(cost_or_None, cpu_seconds)`` aligned to ``idxs``, or ``None`` if the pool broke."""
        pool = self._eval_pool
        to = self.eval_timeout
        order = list(idxs)
        fut = {ji: pool.submit(_eval_one_fssp, (code, ji)) for ji in order}
        out: dict = {}
        pos = 0
        while pos < len(order):
            ji = order[pos]
            if ji in out:
                pos += 1
                continue
            try:
                cost, cpu = fut[ji].result(timeout=to)
                out[ji] = (float(cost), float(cpu))
            except concurrent.futures.TimeoutError:
                print(f"    [eval] instance {ji} exceeded {to}s "
                      f"(heuristic likely infinite-loops in get_matrix_and_jobs) -> "
                      f"candidate invalid; rebuilding pool", flush=True)
                out[ji] = (None, float(to or 0.0))
                for jk in order[pos + 1:]:
                    if jk not in out and fut[jk].done():
                        try:
                            c2, cpu2 = fut[jk].result(timeout=0)
                            out[jk] = (float(c2), float(cpu2))
                        except Exception:
                            out[jk] = (None, 0.0)
                self._recreate_pool()
                for jk in order[pos + 1:]:
                    out.setdefault(jk, (None, 0.0))
                break
            except concurrent.futures.process.BrokenProcessPool:
                print("    [eval] evaluator pool broke (worker died) — rebuilding, "
                      "candidate invalid.", flush=True)
                self._recreate_pool()
                return None
            except Exception:
                out[ji] = (None, 0.0)             # heuristic crash on this instance (fast)
            pos += 1
        return [out.get(ji, (None, 0.0)) for ji in order]

    def shutdown(self):
        """Tear down the evaluator process pool (call once the HiFo run finishes)."""
        try:
            self._eval_pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:                     # py<3.9 has no cancel_futures
            self._eval_pool.shutdown(wait=True)
        except Exception:
            pass
        if self._cpu_log_fh is not None:
            try:
                self._cpu_log_fh.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Logging: HiFo population dumps -> eoh-style trajectory.json + heuristics.json
# --------------------------------------------------------------------------- #
_ARTIFACT_REFRESH_SECONDS = 30  # how often the background watcher rebuilds the logs


def _build_eoh_style_logs(log_dir: pathlib.Path, label: str,
                          n_instances: int, n_operators: int, pop_size: int,
                          n_init_batches: int = 2, wandb_logger=None,
                          verbose: bool = True) -> None:
    """Post-process HiFo's per-generation population dumps
    (``results/pops/population_generation_*.json``) into the same log surface
    ``tiny/eoh_fssp_gls.py`` writes: a per-generation best-so-far ``trajectory.json`` and
    a ``heuristics.json`` of the (deduplicated) candidates seen.

    Each dumped individual carries ``objective`` = mean makespan (lower better, from
    ``_FSSPAdapter.evaluate``). ``score`` = incumbent mean makespan. FSSP has NO known
    optimum, so — like ``tiny/eoh_fssp_gls.py`` — NO gap is reported.

    ``used_budget`` = cumulative candidate EVALUATIONS x n_instances. In the ``tiny``
    variant each candidate is scored on K instances, so pass ``n_instances=K``. It is
    derived from HiFo's generation STRUCTURE, not from the dump sizes (the dumps hold
    only the survivors after ``population_management``): HiFo evaluates
    ``n_init_batches * pop_size`` offspring for the initial population (gid 0) and
    ``n_operators * pop_size`` per later generation (one ``pop_size`` batch for each of
    the e1/e2/m1/m2/m3 operators)."""
    pops_dir = log_dir / "results" / "pops"
    dumps = sorted(
        pops_dir.glob("population_generation_*.json"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    code_to_cand: dict = {}
    heuristics: list = []
    per_gen: dict = {}
    for path in dumps:
        gid = int(path.stem.rsplit("_", 1)[-1])
        try:
            pop = json.load(open(path))
        except Exception:
            continue
        per_gen[gid] = []
        for i, ind in enumerate(pop):
            code = ind.get("code") or ""
            obj = ind.get("objective")
            mean_cost = float(obj) if isinstance(obj, (int, float)) else float("inf")
            if code not in code_to_cand:
                cand_id = f"gen{gid:02d}_cand{i:02d}"
                code_to_cand[code] = cand_id
                heuristics.append({"cand_id": cand_id, "gen_id": gid,
                                   "score": mean_cost, "source": code})
            per_gen[gid].append((code, mean_cost))

    # Real per-generation CPU seconds: the adapter appends one candidate's total CPU
    # (summed across its parallel instance workers) to eval_cpu.jsonl IN CANDIDATE ORDER,
    # so cumulative CPU through gen gid is the prefix-sum of the first cand_evaluated(gid)
    # entries — consistent with used_budget's structural attempt count.
    cpu_prefix: list = []
    _acc = 0.0
    cpu_path = log_dir / "eval_cpu.jsonl"
    if cpu_path.exists():
        try:
            for line in open(cpu_path):
                line = line.strip()
                if not line:
                    continue
                try:
                    _acc += float(json.loads(line).get("cpu", 0.0))
                except Exception:
                    continue
                cpu_prefix.append(_acc)
        except Exception:
            pass

    def _cpu_through(k: int) -> float:
        if k <= 0 or not cpu_prefix:
            return 0.0
        return cpu_prefix[min(k, len(cpu_prefix)) - 1]

    # heuristics.json: prefer the FULL offspring record (all n_operators*pop_size
    # candidates/gen) from all_candidates.jsonl, matching EoH; trajectory below still uses
    # the survivor dumps for the best-so-far incumbent.
    from utils.hifo_candidate_log import build_heuristics_from_candidate_log
    _full = build_heuristics_from_candidate_log(log_dir, n_init_batches, n_operators, pop_size)
    if _full is not None:
        heuristics = _full

    # Map incumbent CODE -> the cand_id it carries in heuristics.json, so the trajectory's
    # incumbent resolves to the SAME source _final_eval (eval_*) re-evaluates. When the full
    # offspring record is used, heuristics.json's cand_ids come from all_candidates.jsonl — a
    # DIFFERENT namespace than the survivor-dump `code_to_cand`; using the dump cand_id here
    # would point the incumbent at a different heuristic in heuristics.json (exploded valid
    # scores). Falls back to the dump map for old runs (no all_candidates.jsonl).
    if _full is not None:
        _inc_cand_by_code: dict = {}
        for _h in _full:
            _inc_cand_by_code.setdefault(_h["source"], _h["cand_id"])
    else:
        _inc_cand_by_code = code_to_cand

    trajectory: list = []
    best = float("inf")
    best_code = None
    for gid in sorted(per_gen):
        for code, mean_cost in per_gen[gid]:
            if math.isfinite(mean_cost) and mean_cost < best:
                best, best_code = mean_cost, code
        best_cand = _inc_cand_by_code.get(best_code) if best_code is not None else None
        score = best if best_code is not None else float("inf")
        # FSSP has no known optimum -> no gap (cf. tiny/eoh_fssp_gls.py).
        # Candidates EVALUATED through this generation (attempts, incl. failed/dupes),
        # from HiFo's structure -- NOT the survivor count in the dumps.
        cand_evaluated = n_init_batches * pop_size + gid * n_operators * pop_size
        trajectory.append({
            "gen_id": gid,
            "cand_id": best_cand,
            "score": float(score),
            "n_instances": n_instances,
            "used_budget": cand_evaluated * n_instances,
            # Cumulative CPU seconds (summed across cores) burned by ALL candidate
            # evaluations through this generation — from eval_cpu.jsonl (see above).
            "cpu_seconds": round(_cpu_through(cand_evaluated), 3),
        })

    # survivors.json: the environmental-selection SURVIVOR set per generation. For HiFo the
    # population_generation_<gid> dumps ARE the survivors kept after population_management, so
    # per_gen[gid] already holds them; map each survivor CODE -> its heuristics.json cand_id
    # (the same all_candidates namespace the incumbent uses). Matches the EoH survivors.json
    # schema so the ranking analysis can form the parent+offspring selection pool uniformly.
    survivors_by_gen: dict = {}
    for gid in sorted(per_gen):
        survivors_by_gen[str(gid)] = [
            {"cand_id": _inc_cand_by_code.get(code), "score": float(mean_cost)}
            for code, mean_cost in per_gen[gid]
        ]

    # Atomic-ish writes (write to a temp then replace) so a reader/analysis never
    # sees a half-written file while the watcher rebuilds mid-run.
    for name, payload in (("trajectory.json", {"label": label, "trajectory": trajectory}),
                          ("heuristics.json", {"label": label,
                                               "total_sampled": len(heuristics),
                                               "heuristics": heuristics}),
                          ("survivors.json", {"label": label, "pop_size": pop_size,
                                              "survivors_by_gen": survivors_by_gen})):
        tmp = log_dir / (name + ".tmp")
        json.dump(payload, open(tmp, "w"), indent=2)
        os.replace(tmp, log_dir / name)
    if verbose:
        print(f"  logged -> {log_dir / 'trajectory.json'}  ({len(trajectory)} generations)")
        print(f"  logged -> {log_dir / 'heuristics.json'}  ({len(heuristics)} candidates)")
    if wandb_logger is not None:
        for row in trajectory:
            wandb_logger.log_trajectory_row(row)


# --------------------------------------------------------------------------- #
# Post-evolution final evaluation (mirror tiny/eoh_fssp_gls.py)
# --------------------------------------------------------------------------- #
def _final_eval(log_dir: pathlib.Path, num_cores: int, n_jobs: int = 50,
                m_low: int = 2, m_high: int = 20,
                instance_source: str = "pregenerated") -> dict:
    """Post-evolution evaluation using src/analyses/eval_fssp.py with mode='valid'.
    Re-scores the incumbent on a FIXED held-out set of 64 FSSP instances (not the
    K-subset) and saves to valid_trajectory.json. Mirrors tiny/eoh_fssp_gls.py's
    _final_eval (validation ALWAYS on 64 instances)."""
    from analyses.eval_fssp import evaluate_single, _to_serialisable
    print("=== Post-Evolution Final Evaluation (eval_fssp.py) ===", flush=True)
    res = evaluate_single(
        exp_path=log_dir,
        n_instances=64,
        n_jobs=n_jobs,
        m_low=m_low,
        m_high=m_high,
        instance_source=instance_source,
        n_cores=num_cores,
        mode="valid",
        has_incumbent=False,
    )
    out_path = log_dir / "valid_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=_to_serialisable)
    print(f"Saved post-evolution validation trajectory -> {out_path}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-K random policy evaluation for HiFo-Prompt on FSSP-GLS.")
    p.add_argument("--pop-size", type=int, default=10, help="HiFo ec_pop_size (population size).")
    p.add_argument("--max-generations", type=int, default=20,
                   help="Reference number of HiFo generations at K=n_instances. The "
                        "actual ec_n_pop is budget-matched up from this (see main).")
    p.add_argument("--n-instances", type=int, default=64,
                   help="Number of FSSP instances in the pool.")
    p.add_argument("--K", type=int, default=1,
                   help="Number of instances to evaluate each heuristic on (default: 1).")
    p.add_argument("--instance-mode", type=str, default="fixed", choices=["fixed", "random"],
                   help="K-subset selection from the --n-instances pool. 'fixed' "
                        "(default): sample K instances once (seeded by --seed) before "
                        "evolution and reuse for every heuristic. 'random': sample a "
                        "fresh K-subset per heuristic evaluation (seeded reproducibly "
                        "by the heuristic source). Both evaluate each heuristic on all "
                        "K instances.")
    p.add_argument("--n-jobs", type=int, default=50, help="Jobs per FSSP instance.")
    p.add_argument("--m-low", type=int, default=2, help="Min machines per instance (inclusive).")
    p.add_argument("--m-high", type=int, default=20, help="Max machines per instance (inclusive).")
    p.add_argument("--instance-source", type=str, default="pregenerated",
                   choices=["pregenerated", "generate"],
                   help="'pregenerated' (default) loads the fixed EoH-S FSSP dataset (same "
                        "as tiny/eoh_fssp_gls.py); 'generate' samples fresh instances by --seed.")
    p.add_argument("--selection-num", type=int, default=5,
                   help="HiFo ec_m: number of parents for e1/e2 (>=2).")
    p.add_argument("--eval-timeout", type=float, default=65.0,
                   help="PER-(heuristic, instance) wall-clock cap (s). Each of a "
                        "candidate's K subset GLS runs is capped here (~65s = GLS "
                        "60s/instance + slack); a heuristic that infinite-loops inside "
                        "get_matrix_and_jobs is reclaimed in this time.")
    p.add_argument("--timeout-seconds", type=int, default=None,
                   help="DEPRECATED / ignored — the eval is now per-(heuristic, instance); "
                        "use --eval-timeout. Accepted for wrapper back-compat only.")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="LLM sampling temperature for the HiFo prompts.")

    # --- Layer B: HiFo method internals (default == paper config; injected
    #     into hifo.methods.hifo.hifo_hp before the engine runs) -------------- #
    g_ip = p.add_argument_group("HiFo Insight Pool (Hindsight)")
    g_ip.add_argument("--pool-capacity", type=int, default=30,
                      help="C_pool: max insights retained (paper 30).")
    g_ip.add_argument("--novelty-threshold", type=float, default=0.7,
                      help="theta_novelty: Jaccard admission cutoff (paper 0.7).")
    g_ip.add_argument("--selection-count", type=int, default=3,
                      help="s: top-s insights injected per generation (paper 3).")
    g_ip.add_argument("--usage-penalty-weight", type=float, default=0.1,
                      help="w_u: usage-penalty coefficient (paper 0.1).")
    g_ip.add_argument("--recency-bonus", type=float, default=0.2,
                      help="tau_r: recency bonus magnitude (paper 0.2).")
    g_ip.add_argument("--recency-window", type=int, default=2,
                      help="T_w: recency bonus window in generations (paper 2).")
    g_ip.add_argument("--ema-alpha", type=float, default=0.3,
                      help="alpha: EMA rate for effectiveness (paper 0.3).")
    g_ip.add_argument("--decay-rate", type=float, default=0.01,
                      help="R_decay: eviction-score time decay (paper 0.01).")
    g_ip.add_argument("--probation-usage", type=int, default=3,
                      help="T_usage: uses below which an insight is eviction-immune (paper 3).")

    g_cr = p.add_argument_group("HiFo credit-assignment tiers (Eq. 3 intercepts)")
    g_cr.add_argument("--credit-best", type=float, default=0.8,
                      help="beta_best: intercept when offspring beats the best (paper 0.8).")
    g_cr.add_argument("--credit-inc", type=float, default=0.2,
                      help="beta_inc: intercept when offspring beats the average (paper 0.2).")
    g_cr.add_argument("--credit-pen", type=float, default=-0.3,
                      help="beta_pen: intercept when offspring is below average (paper -0.3).")

    g_nav = p.add_argument_group("HiFo Evolutionary Navigator (Foresight)")
    g_nav.add_argument("--progress-eps", type=float, default=1e-4,
                       help="delta g: min best-fitness gain counted as progress (paper 1e-4).")
    g_nav.add_argument("--stagnation-threshold", type=int, default=3,
                       help="tau_stag: stagnant gens before forcing Explore (paper 3).")
    g_nav.add_argument("--progress-threshold", type=int, default=2,
                       help="tau_prog: progress gens before forcing Exploit (paper 2).")
    g_nav.add_argument("--diversity-threshold", type=float, default=0.3,
                       help="delta_p: phenotypic-diversity floor for Explore (paper 0.3).")

    p.add_argument("--label", type=str, default="hifo/tiny_fssp_gls")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048, help="Max tokens for LLM generation.")
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm"],
                   help="LLM backend (same options as tiny/eoh_tsp_gls.py).")
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next",
                   help="Model id passed to the selected backend.")
    p.add_argument("--ollama-host", type=str, default=None,
                   help="Ollama host:port (only for --llm-backend=ollama).")
    p.add_argument("--num-threads", type=int, default=4,
                   help="HiFo exp_n_proc: parallel offspring SAMPLER workers (joblib "
                        "threading backend). Mirrors tiny/eoh_tsp_gls's num_samplers.")
    p.add_argument("--num-cores", type=int, default=4,
                   help="Parallel EVALUATORS (process pool), mirroring tiny/eoh_tsp_gls's "
                        "num_evaluators — the GLS evaluation runs in this many worker "
                        "processes, decoupled from the --num-threads samplers.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-wandb", action="store_true", default=False,
                   help="Log the trajectory to Weights & Biases.")
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed HiFo gen-0 from the paired fixed population in src/init_pop/ "
                        "(2*pop_size heuristics, re-evaluated this run) then run normal "
                        "population_management, instead of sampling from the LLM.")
    return p.parse_args(argv)


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

    mirror_stdout_to(log_dir / "terminal.txt")
    print(f"terminal output mirrored -> {log_dir / 'terminal.txt'}")

    n_op = 5              # HiFo default operators: e1, e2, m1, m2, m3
    n_init_batches = 2    # HiFo population_generation n_create=2 (initial pop batches)

    # Budget-match reprod/hifo. reprod spends (n_init + max_gen*n_op)*pop*n_instances
    # instance-evals; evaluating each candidate on only K instances, scale ec_n_pop up
    # to eff_gen so (n_init + eff_gen*n_op)*pop*K == that target (eff_gen == max_gen
    # when K == n_instances).
    target_budget = (n_init_batches + args.max_generations * n_op) * args.pop_size * args.n_instances
    target_candidates = target_budget / (args.pop_size * args.K)  # (n_init + eff_gen*n_op)
    eff_gen = max(1, math.ceil((target_candidates - n_init_batches) / n_op))
    tiny_candidates = (n_init_batches + eff_gen * n_op) * args.pop_size
    tiny_budget = tiny_candidates * args.K
    print(f"pop_size={args.pop_size}  reprod max_generations={args.max_generations}  "
          f"pool_instances={args.n_instances}  K={args.K}  instance_mode={args.instance_mode}  "
          f"n_jobs={args.n_jobs}")
    print(f"HiFo operators/gen = {n_op} (e1,e2,m1,m2,m3) + {n_init_batches} init batches; "
          f"budget-matched effective generations = {eff_gen} "
          f"(reprod HiFo budget = {target_budget} instance-evals)")
    print(f"tiny candidates = {tiny_candidates}, function evaluations = {tiny_budget}")

    # ---- LLM backend (identical to tiny/eoh_tsp_gls.py) -> CachedLLM ------ #
    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
    cached = CachedLLM(client, cache_dir=cache_dir,
                       prompt_log=log_dir / "llm_prompts.jsonl")
    print(f"Backend: {args.llm_backend}  model: {cached.client.model}  cache: {cache_dir}")

    # Bridge HiFo's LLM interface to our CachedLLM (monkeypatch the factory used
    # inside hifo_evolution.Evolution.__init__).
    bridge = _HiFoLLMBridge(cached, temperature=args.temperature, max_tokens=args.llm_max_tokens)
    _hifo_evolution.InterfaceLLM = lambda *a, **k: bridge

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    args_dict["llm_model"] = cached.client.model
    args_dict["effective_generations"] = eff_gen
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args)}"
    wandb_logger = make_wandb_logger(enabled=args.use_wandb, project="llm4ad",
                                     name=run_name, config=args_dict)

    # ---- FSSP-GLS evaluator (same instances/scoring as tiny/eoh_fssp_gls) - #
    # Override the GLS caps to the tiny/eoh_fssp_gls.py values (module defaults are 10s);
    # also re-set per worker (see _fssp_worker_init).
    _fssp_eval_mod.time_max = _FSSP_TIME_MAX
    _fssp_eval_mod.iter_max = _FSSP_ITER_MAX
    # GetData seeds np.random internally; re-seed before/after for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    full_datasets = list(GetData(
        args.n_instances, n_jobs=args.n_jobs, m_low=args.m_low, m_high=args.m_high,
        use_pregenerated=(args.instance_source == "pregenerated")).generate_instances())
    n_full = len(full_datasets)
    # The K-instance subset each candidate is scored on. 'fixed': one subset sampled once
    # (seeded by --seed), reused for every candidate. 'random': a fresh subset per
    # candidate (seeded by the heuristic source) -- selected in the adapter.
    if args.instance_mode == "fixed":
        fixed_idx = _sample_subset_idx(n_full, args.K, args.seed)
        print(f"  [instance-mode=fixed] sampled K={len(fixed_idx)} of {n_full} instances "
              f"once (seed={args.seed}): idx={fixed_idx}")
    else:
        fixed_idx = None
        print(f"  [instance-mode=random] K={args.K} of {n_full} instances resampled per "
              f"heuristic (seeded by source ^ {args.seed})")
    eval_timeout = args.eval_timeout if args.eval_timeout and args.eval_timeout > 0 else 65.0
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"  instances: evaluated K={args.K}  n_jobs={args.n_jobs}  machines in "
          f"[{args.m_low},{args.m_high}]  instance_mode={args.instance_mode}  "
          f"per-(heuristic,instance) eval timeout={eval_timeout}s")
    print("  (FSSP has no known optimum -> no gap; trajectory tracks raw mean makespan)")

    # Per-(heuristic, instance) evaluator process pool: --num-cores parallelises a
    # candidate's K subset GLS runs (each capped at eval_timeout); --num-threads stays
    # HiFo's sampler (LLM) concurrency.
    adapter = _FSSPAdapter(full_datasets, k=args.K, instance_mode=args.instance_mode,
                           subset_seed=args.seed, fixed_idx=fixed_idx,
                           num_cores=args.num_cores, eval_timeout=eval_timeout,
                           log_dir=log_dir)
    # Log EVERY evaluated candidate (all n_operators*pop_size offspring/gen + seeds) to
    # all_candidates.jsonl, so heuristics.json lists them all like EoH's (HiFo dumps only
    # the pop_size survivors).
    from utils.hifo_candidate_log import wrap_adapter_logging
    wrap_adapter_logging(adapter, log_dir)

    # ---- HiFo method internals (Layer B) -> hifo_hp ---------------------- #
    # Override the package's paper-default constants from the CLI. Done BEFORE
    # EVOL.run() builds the InsightPool / Navigator; the call sites read
    # hifo_hp.<NAME> at call time, so these take effect for the whole run.
    _hifo_hp.POOL_CAPACITY = args.pool_capacity
    _hifo_hp.NOVELTY_THRESHOLD = args.novelty_threshold
    _hifo_hp.SELECTION_COUNT = args.selection_count
    _hifo_hp.USAGE_PENALTY_WEIGHT = args.usage_penalty_weight
    _hifo_hp.RECENCY_BONUS = args.recency_bonus
    _hifo_hp.RECENCY_WINDOW = args.recency_window
    _hifo_hp.EMA_ALPHA = args.ema_alpha
    _hifo_hp.DECAY_RATE = args.decay_rate
    _hifo_hp.PROBATION_USAGE_COUNT = args.probation_usage
    _hifo_hp.CREDIT_BEST = args.credit_best
    _hifo_hp.CREDIT_INC = args.credit_inc
    _hifo_hp.CREDIT_PEN = args.credit_pen
    _hifo_hp.PROGRESS_EPS = args.progress_eps
    _hifo_hp.STAGNATION_THRESHOLD = args.stagnation_threshold
    _hifo_hp.PROGRESS_THRESHOLD = args.progress_threshold
    _hifo_hp.DIVERSITY_THRESHOLD = args.diversity_threshold
    print(f"HiFo internals: C_pool={_hifo_hp.POOL_CAPACITY} nov={_hifo_hp.NOVELTY_THRESHOLD} "
          f"s={_hifo_hp.SELECTION_COUNT} w_u={_hifo_hp.USAGE_PENALTY_WEIGHT} "
          f"tau_r={_hifo_hp.RECENCY_BONUS}/T_w={_hifo_hp.RECENCY_WINDOW} alpha={_hifo_hp.EMA_ALPHA} "
          f"R_decay={_hifo_hp.DECAY_RATE} T_usage={_hifo_hp.PROBATION_USAGE_COUNT} | "
          f"credit=[{_hifo_hp.CREDIT_BEST},{_hifo_hp.CREDIT_INC},{_hifo_hp.CREDIT_PEN}] | "
          f"nav: eps={_hifo_hp.PROGRESS_EPS} tau_stag={_hifo_hp.STAGNATION_THRESHOLD} "
          f"tau_prog={_hifo_hp.PROGRESS_THRESHOLD} delta_p={_hifo_hp.DIVERSITY_THRESHOLD}")

    # HiFo runs ONE offspring's LLM-generation AND its evaluation inside a single joblib
    # task, gated by two hifo_hp knobs. The true per-INSTANCE cap is the adapter's
    # result(timeout=eval_timeout); a candidate's K subset runs in ceil(K/num_cores) waves,
    # plus the serialised-eval lock queue (up to num_threads candidates ahead). Both HiFo
    # budgets are set generously so a legitimately-working offspring is never killed.
    waves = math.ceil(args.K / max(1, args.num_cores))
    per_cand_cap = waves * eval_timeout
    _hifo_hp.EVAL_TIMEOUT = int(max(1, args.num_threads) * per_cand_cap + 60)
    _hifo_hp.TASK_BUDGET = int(args.llm_timeout + _hifo_hp.EVAL_TIMEOUT + 30)
    print(f"HiFo timeouts: per-(heuristic,instance) cap = {eval_timeout}s | per-candidate "
          f"eval <= {per_cand_cap:.0f}s ({waves} waves x {eval_timeout}s) | HiFo eval net = "
          f"{_hifo_hp.EVAL_TIMEOUT}s | per-offspring budget = {_hifo_hp.TASK_BUDGET}s")

    # ---- HiFo parameters ------------------------------------------------- #
    paras = Paras()
    paras.set_paras(
        method="hifo",
        problem=adapter,                # non-string -> HiFo's Probs uses it directly
        ec_pop_size=args.pop_size,
        ec_n_pop=eff_gen,               # budget-matched effective generation count
        ec_m=max(2, args.selection_num),
        exp_n_proc=args.num_threads,
        exp_output_path=str(log_dir),
        exp_debug_mode=False,
        eva_timeout=_hifo_hp.EVAL_TIMEOUT,  # fallback; hifo_hp.EVAL_TIMEOUT/TASK_BUDGET drive it
        eva_numba_decorator=False,      # candidate get_matrix_and_jobs is plain numpy; no numba
        llm_use_local=False,
        llm_api_endpoint=args.llm_backend,   # unused (InterfaceLLM is bridged)
        llm_api_key="bridged",
        llm_model=cached.client.model,
    )

    # Eval concurrency is bounded by the number of sampler threads that submit evals
    # (each HiFo sampler thread submits ONE eval and blocks on .result(), like EoH). So
    # effective evaluator parallelism = min(num_threads, num_cores). If num_cores >
    # num_threads the surplus workers are pre-forked but NEVER active — they only cost
    # memory (the nc>nt trap; note this differs from racing, where eval is per-instance).
    if args.num_cores > args.num_threads:
        print(f"  [warn] --num-cores {args.num_cores} > --num-threads {args.num_threads}: "
              f"only ~{args.num_threads} evaluators can ever be active (each sampler "
              f"thread submits one eval and blocks), so the extra "
              f"{args.num_cores - args.num_threads} pre-warmed workers just consume "
              f"memory. Raise --num-threads to >= --num-cores to actually use them.")
    print(f"[{args.label}] running HiFo (native engine, threading backend, "
          f"n_proc={args.num_threads}, evaluators={args.num_cores}, eff_gen={eff_gen})...")
    t0 = time.time()

    # HiFo dumps results/pops/population_generation_*.json AFTER every generation; a
    # background watcher rebuilds the eoh-style trajectory.json / heuristics.json from
    # them every few seconds (idempotent, atomic), and a final guaranteed build runs in
    # `finally` even if the engine raises. In the tiny variant each candidate is scored
    # on K instances -> pass n_instances=K.
    _stop = threading.Event()

    def _log_watcher():
        while not _stop.wait(_ARTIFACT_REFRESH_SECONDS):
            try:
                _build_eoh_style_logs(log_dir, args.label, args.K,
                                      n_operators=n_op, pop_size=args.pop_size,
                                      n_init_batches=n_init_batches,
                                      wandb_logger=None, verbose=False)
            except Exception:
                pass  # transient (e.g. a dump being written) -> retry next tick

    watcher = threading.Thread(target=_log_watcher, name="hifo-log-watcher", daemon=True)
    watcher.start()

    # Force joblib's THREADING backend so the shared CachedLLM bridge is not
    # pickled to worker processes (HiFo parallelises offspring with joblib).
    import joblib
    try:
        with joblib.parallel_backend("threading", n_jobs=args.num_threads):
            if args.fix_init_pop:
                # Inject 2*pop_size fixed heuristics as HiFo's gen-0 batch (via the
                # native seed evaluator, so they're re-scored on THIS run's instances),
                # then let HiFo's normal population_management prune to pop_size.
                from utils.fixed_init_pop import load_fixed_initial_population
                from hifo.methods.hifo.hifo_interface_EC import InterfaceEC as _IEC
                import re as _re
                _fp = ROOT / "src" / "init_pop" / "hifo_fssp_gls.json"
                _srcs = [h["source"] for h in
                         load_fixed_initial_population(_fp, 2 * args.pop_size)]
                def _fixed_population_generation(self, __srcs=_srcs):
                    import math as _math
                    from joblib import Parallel as _Par, delayed as _del
                    seeds = []
                    for _c in __srcs:
                        _m = _re.search(r"def\s+(\w+)", _c) or _re.search(r"class\s+(\w+)", _c)
                        seeds.append({"algorithm": (_m.group(1) if _m else "fixed"), "code": _c})
                    # Robust seed init: HiFo's population_generation_seed does
                    # np.round(np.array(fitness),5) and exit()s if ANY seed evaluates to
                    # None (a fixed heuristic that crashes/times-out on some instance).
                    # Evaluate here and DROP the None/non-finite seeds so one bad heuristic
                    # degrades gracefully instead of aborting the run; population_management
                    # then prunes the finite survivors to pop_size. (Analog of the EoH
                    # fix-init -1e6 coercion — the failure just must not kill the run.)
                    _fit = _Par(n_jobs=self.n_p)(
                        _del(self.interface_eval.evaluate)(_s["code"]) for _s in seeds)
                    _pop = []
                    for _s, _f in zip(seeds, _fit):
                        if _f is None or not isinstance(_f, (int, float)) or not _math.isfinite(_f):
                            continue
                        _pop.append({"algorithm": _s["algorithm"], "code": _s["code"],
                                     "objective": round(float(_f), 5), "other_inf": None})
                    print(f"  [fixed-init-pop] {len(_pop)}/{len(seeds)} seeds evaluated "
                          f"finite (dropped {len(seeds) - len(_pop)} that failed); no HiFo "
                          f"exit() on failure", flush=True)
                    if not _pop:
                        raise RuntimeError(
                            "fixed-init-pop: ALL seed heuristics failed evaluation — cannot "
                            "seed HiFo gen-0. Check src/init_pop/*.json (heuristics must "
                            "evaluate finite on this run's instances).")
                    return _pop
                _IEC.population_generation = _fixed_population_generation
                print(f"  [fixed-init-pop] injecting {len(_srcs)} heuristics (2*pop_size) "
                      f"into HiFo gen-0 then normal management, from {_fp}", flush=True)
            EVOL(paras).run()
    finally:
        _stop.set()
        watcher.join(timeout=5)
        adapter.shutdown()   # tear down the evaluator process pool
        # Guaranteed final rebuild from whatever generations completed (so even an
        # interrupted / crashed run leaves usable trajectory.json + heuristics.json).
        _build_eoh_style_logs(log_dir, args.label, args.K,
                              n_operators=n_op, pop_size=args.pop_size,
                              n_init_batches=n_init_batches, wandb_logger=wandb_logger)
    dt = time.time() - t0
    print(f"[{args.label}] HiFo finished in {dt:.1f}s, {adapter.n_evals} evaluations")

    # Post-evolution validation on a FIXED held-out 64 instances (mirror tiny/eoh_fssp_gls.py).
    try:
        _final_eval(log_dir=log_dir, num_cores=args.num_cores, n_jobs=args.n_jobs,
                    m_low=args.m_low, m_high=args.m_high, instance_source=args.instance_source)
    except Exception as e:
        print(f"  [_final_eval] WARN: final evaluation failed: {type(e).__name__}: {e}", flush=True)

    wandb_logger.finish()
    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
