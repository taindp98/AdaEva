"""Reproduce HiFo-Prompt on the LLM4AD TSP-GLS task.

The TSP-GLS counterpart of ``reprod/hifo_obp.py`` (and companion to
``reprod/eoh_tsp_gls.py``): it drives the **HiFo-Prompt package's native engine**
(InsightPool + EvolutionaryNavigator + the ``m3`` operator) on the TSP Guided
Local Search task. Its TSP-GLS benchmark defaults (n_instances / problem_size /
timeout + the GLS evaluation itself) STRICTLY FOLLOW ``reprod/eoh_tsp_gls.py`` so the
two reproductions score on an identical benchmark and are 1:1 comparable. For a CHEAP
debug run — useful for validating the ported HiFo<->TSP patterns (the
``update_edge_distance`` prompt spec, the per-instance GLS adapter, the bridge,
the eoh-style logging) before the expensive racing/hifo_tsp_gls run — override
``--pop-size`` / ``--max-generations`` / ``--n-instances`` on the CLI (or a wrapper).

We plug the SAME pieces into HiFo as ``reprod/eoh_tsp_gls.py`` uses for EoH:
    - Evaluator: an adapter (``_TSPAdapter``) exposing HiFo's problem interface
      (``.prompts`` + ``.evaluate(code)``) over the TSP-GLS instance pool. The generated
      function is ``update_edge_distance`` (the GLS penalty-matrix update); the objective
      is the mean tour cost over the instances (lower better; ``None`` if any instance
      crashes or times out).
    - LLM: HiFo's internal ``InterfaceLLM`` is monkeypatched to a bridge over the
      SAME ``CachedLLM`` backends (openrouter/ollama/mistral/vllm), with a
      per-call cache salt so identical prompts (e.g. i1 init) still diversify.
    - Parallelism: per-(heuristic, instance), mirroring reprod/reevo_tsp_gls (NOT EoH's
      per-candidate model). HiFo samples offspring with ``joblib.Parallel`` on the
      *threading* backend (``--num-threads`` LLM samplers; keeps the CachedLLM bridge
      shared, not pickled), and ``_TSPAdapter`` evaluates each candidate by submitting its
      n_instances GLS runs as INDEPENDENT tasks to a shared
      ``ProcessPoolExecutor(max_workers=--num-cores)`` — each capped by a PER-INSTANCE
      ``result(timeout=--eval-timeout)`` (~65s), so a heuristic that hangs inside
      ``update_edge_distance`` is reclaimed in seconds (pool rebuilt), not after a
      ``65 x n_instances`` whole-candidate kill. ``--num-cores`` parallelises a candidate's
      instances (``num_cores > num_threads`` is beneficial).
    - Logging: HiFo's per-generation population dumps are post-processed into
      eoh-style ``trajectory.json`` + ``heuristics.json`` (+ ``args.yaml`` /
      ``terminal.txt`` / run-tag). ``gap = (mean_cost - mean_opt) / mean_opt`` with
      ``mean_opt`` the mean Concorde optimum over the instance pool.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import concurrent.futures
import pathlib
import random
import sys
import tempfile
import threading
import time
import uuid
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "packages" / "HiFo-Prompt" / "hifo" / "src"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, make_wandb_logger, OpenRouterClient, GoogleClient, OllamaClient,
                   MistralClient, vLLMClient)
from utils.logger import make_log_dir, mirror_stdout_to

from llm4ad.task.optimization.tsp_gls_2O.evaluation import TSP_GLS_2O_Evaluation_wo_Time
from llm4ad.task.optimization.tsp_gls_2O.get_instance import GetData

# Concorde optimum (for the trajectory gap), identical to reprod/eoh_tsp_gls.py.
_CONCORDE_COORD_SCALE = 1e6


def _silence_fds():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    return saved


def _restore_fds(saved):
    os.dup2(saved[0], 1)
    os.dup2(saved[1], 2)
    os.close(saved[0])
    os.close(saved[1])


def _opt_cost(inst) -> float:
    """Concorde optimal tour cost for one TSP instance (for the gap metric)."""
    from concorde.tsp import TSPSolver
    coords = np.asarray(inst[0] if isinstance(inst, tuple) else inst.positions)
    xs = (coords[:, 0] * _CONCORDE_COORD_SCALE).tolist()
    ys = (coords[:, 1] * _CONCORDE_COORD_SCALE).tolist()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        saved = _silence_fds()
        try:
            solver = TSPSolver.from_data(xs, ys, norm="EUC_2D", name=uuid.uuid4().hex)
            sol = solver.solve(verbose=False)
        finally:
            _restore_fds(saved)
            os.chdir(cwd)
    tour = list(sol.tour)
    cost = 0.0
    for a, b in zip(tour, tour[1:] + [tour[0]]):
        cost += float(np.linalg.norm(coords[a] - coords[b]))
    return cost

# HiFo-Prompt package (native engine)
from hifo.hifo import EVOL
from hifo.utils.getParas import Paras
import hifo.methods.hifo.hifo_evolution as _hifo_evolution
import hifo.methods.hifo.hifo_hp as _hifo_hp


# The TSP-GLS benchmark + search defaults STRICTLY FOLLOW reprod/eoh_tsp_gls.py so
# the two reproductions score on an identical benchmark and are 1:1 comparable
# (n_instances/problem_size/timeout + the GLS evaluation are the same). For a CHEAP
# debug run, override pop_size/max_generations/n_instances on the CLI / in a wrapper.
# Used for the folder-name tag (mirrors eoh_tsp_gls: n_instances=64 is the wrapper /
# paper value, while the bare-script argparse default is 16).
_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "pop_size": 10,
    "max_generations": 20,
    "n_instances": 64,
    "problem_size": 100,
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
    "problem_size": "prob",
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


def _mean_opt(instances: list) -> float:
    """Mean Concorde optimal tour cost over the instance pool (for the gap metric)."""
    return float(np.mean([_opt_cost(inst) for inst in instances]))


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
# Problem adapter: HiFo's problem interface (.prompts + .evaluate) over LLM4AD's
# TSP_GLS_2O_Evaluation, so HiFo scores on the SAME task/instances as eoh_tsp_gls.
# --------------------------------------------------------------------------- #
class _TSPPrompts:
    """GetPrompts-compatible prompt spec for TSP-GLS. The generated function is
    ``update_edge_distance`` (the GLS penalty-matrix update). ``func_name``
    (``update_edge_distance``) differs from ``func_outputs`` (``updated_edge_distance``)
    as HiFo's code reconstruction requires."""

    def __init__(self):
        self.prompt_task = (
            "Given an edge distance matrix and a local optimal route, design a strategy "
            "to update the distance matrix so that Guided Local Search escapes the local "
            "optimum and ultimately finds a shorter tour. Create a heuristic that updates "
            "the edge distance matrix."
        )
        self.prompt_func_name = "update_edge_distance"
        self.prompt_func_inputs = ["edge_distance", "local_opt_tour", "edge_n_used"]
        self.prompt_func_outputs = ["updated_edge_distance"]
        self.prompt_inout_inf = (
            "'edge_distance' is a Numpy matrix of pairwise edge distances; "
            "'local_opt_tour' is a Numpy array of node IDs giving the current "
            "local-optimum tour; 'edge_n_used' is a Numpy matrix counting how often "
            "each edge was used across the search. Return 'updated_edge_distance', a "
            "Numpy matrix the same shape as 'edge_distance' holding the updated "
            "(penalised) distances."
        )
        self.prompt_other_inf = (
            "All inputs and outputs are Numpy arrays. Keep the function signature, "
            "inputs and outputs unchanged, avoid division-by-zero on 'edge_n_used', and "
            "include 'import numpy as np' at the top of the code."
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


# --------------------------------------------------------------------------- #
# Per-(heuristic, instance) evaluation workers (run in the process pool).
#
# The eval is flattened to ONE (candidate, instance) task per pool submission so the
# pool parallelises across INSTANCES: a single candidate's n_instances GLS runs use all
# --num-cores workers instead of running serially in one subprocess. This mirrors
# reprod/reevo_tsp_gls (and reprod/llamea_/openevolve_tsp_gls) rather than EoH's
# per-candidate model. The instances live as worker-process globals (set ONCE by the
# pool initializer), so only ``(code, inst_idx)`` crosses the pickle boundary per task —
# not the distance matrices.
# --------------------------------------------------------------------------- #
_W_INSTANCES = None   # per-worker cache of the TSP instance list (set by the initializer)


def _tsp_worker_init(instances):
    global _W_INSTANCES
    _W_INSTANCES = instances


def _tsp_compile_update(code: str):
    """Compile ``code`` and return the ``update_edge_distance`` callable (raises if
    absent/unparseable — the aggregator turns that into an invalid candidate)."""
    ns = {"np": np}
    exec(code, ns)
    fn = ns.get("update_edge_distance")
    if not callable(fn):
        raise ValueError("no 'update_edge_distance' defined in code")
    return fn


def _eval_one_tsp(args):
    """Evaluate ONE candidate on ONE instance -> tour cost (lower better; +inf on a
    heuristic crash). ``args`` is ``(code, inst_idx)``; the instance comes from the
    worker global. The GLS run is internally capped at 60s/instance (gls.py); the OUTER
    per-instance wall-clock cap (result(timeout=...) in the parent) reclaims a heuristic
    that infinite-loops INSIDE ``update_edge_distance`` (before the 60s check)."""
    from llm4ad.task.optimization.tsp_gls_2O.evaluation import solve_without_time
    code, inst_idx = args
    fn = _tsp_compile_update(code)   # exec is ~µs; numba JIT is per-worker, not per-task
    t0 = time.process_time()
    c = solve_without_time(_W_INSTANCES[inst_idx], fn)   # inf on internal crash
    cpu = time.process_time() - t0                       # worker CPU for this instance's GLS
    return float(c), float(cpu)


class _TSPAdapter:
    """HiFo problem object over the TSP-GLS instance pool.

    HiFo calls ``.evaluate(code_string)`` and minimises the returned objective, so we
    return the **mean tour cost (lower = better)** over the n_instances — or ``None`` if
    ANY instance crashes or times out (matching ``evaluate_without_time``'s
    ``mean``-with-``inf`` semantics, so the objective is identical to the old per-
    candidate path; only the parallelism and timeout granularity change).

    Parallel evaluation is per-(heuristic, instance), mirroring reprod/reevo_tsp_gls:
    ``evaluate()`` submits the candidate's n_instances GLS runs as INDEPENDENT tasks to a
    shared ``ProcessPoolExecutor(max_workers=num_cores)`` (workers cache the instances via
    the initializer), and collects each with a PER-INSTANCE ``result(timeout=eval_timeout)``
    cap (~65s = GLS 60s + slack). A hung instance (heuristic infinite-looping inside
    ``update_edge_distance``) is caught in ``eval_timeout`` seconds — not the old
    ``65 x n_instances`` whole-candidate kill — after which the pool is rebuilt to
    terminate the stuck worker. So ``num_cores`` parallelises a candidate's instances and
    ``num_cores > num_threads`` is now beneficial. ``evaluate()`` is serialised with a lock
    (one candidate's instances fill the pool at a time), which keeps the pool-rebuild-on-
    hang safe under HiFo's ``num_threads`` concurrent sampler threads; throughput is
    ``num_cores`` instance-evals in parallel while the samplers keep the LLM pipeline full."""

    def __init__(self, instances, num_cores: int = 1, eval_timeout: float = 65.0,
                 log_dir: pathlib.Path = None):
        self.prompts = _TSPPrompts()
        self._instances = list(instances)
        self.n_instances = len(self._instances)
        self.num_cores = max(1, int(num_cores))
        self.eval_timeout = (float(eval_timeout)
                             if eval_timeout and float(eval_timeout) > 0 else None)
        self._pool_init = (_tsp_worker_init, (self._instances,))
        self._eval_pool = None
        self._eval_lock = threading.Lock()   # serialise candidates (pool-rebuild safety)
        self.n_evals = 0
        # Per-candidate CPU seconds (summed across the parallel instance workers = CPU
        # across cores, matching tiny/eoh_tsp_gls's cum_cpu). Appended in candidate order
        # to eval_cpu.jsonl; _build_eoh_style_logs prefix-sums it into the trajectory's
        # cumulative ``cpu_seconds`` per generation.
        self._cpu_seconds = 0.0
        self._cpu_log_fh = (open(log_dir / "eval_cpu.jsonl", "a", buffering=1)
                            if log_dir is not None else None)
        self._start_pool()

    def _record_cpu(self, cand_cpu: float) -> None:
        """Accumulate one candidate's CPU and append it (in candidate order) to
        eval_cpu.jsonl. Called under ``_eval_lock`` (evaluate is serialised)."""
        self._cpu_seconds += float(cand_cpu)
        if self._cpu_log_fh is not None:
            try:
                self._cpu_log_fh.write(json.dumps({"cpu": round(float(cand_cpu), 4)}) + "\n")
            except Exception:
                pass

    def _start_pool(self):
        # Workers cache the instance list once (initializer), so tasks ship only
        # (code, inst_idx). Pre-warm here in the single-threaded main (before HiFo's
        # sampler threads start) so workers fork from a single-threaded parent
        # (ProcessPoolExecutor otherwise forks lazily from a multi-threaded parent —
        # a fork+threads deadlock hazard).
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
            results = self._map_instances_with_timeout(code_string)
            if results is None:                   # pool broke mid-eval
                self._record_cpu(0.0)
                return None
            costs = [c for (c, _cpu) in results]
            cand_cpu = float(sum(cpu for (_c, cpu) in results))  # CPU across parallel workers
            self._record_cpu(cand_cpu)            # count CPU even for an invalid candidate
            # Match evaluate_without_time: mean over ALL instances; a single crashed
            # (inf) or timed-out (None) instance invalidates the candidate.
            if any(c is None or not math.isfinite(c) for c in costs):
                return None
            self.n_evals += 1
            return float(np.mean(costs))          # mean tour cost (lower is better)

    def _map_instances_with_timeout(self, code):
        """Submit the candidate's n_instances GLS tasks; collect each as ``(cost, cpu)``
        with a per-instance timeout. On a genuine hang, salvage finished siblings, rebuild
        the pool (to kill the stuck worker) and short-circuit — the candidate is already
        invalid, so there is no need to re-run its remaining instances. Returns a list of
        ``(cost_or_None, cpu_seconds)`` aligned to the instances, or ``None`` if the pool
        broke."""
        pool = self._eval_pool
        n = self.n_instances
        to = self.eval_timeout
        fut = {j: pool.submit(_eval_one_tsp, (code, j)) for j in range(n)}
        out: dict = {}
        j = 0
        while j < n:
            if j in out:
                j += 1
                continue
            try:
                cost, cpu = fut[j].result(timeout=to)
                out[j] = (float(cost), float(cpu))
            except concurrent.futures.TimeoutError:
                print(f"    [eval] instance {j} exceeded {to}s "
                      f"(heuristic likely infinite-loops in update_edge_distance) -> "
                      f"candidate invalid; rebuilding pool", flush=True)
                out[j] = (None, float(to or 0.0))   # a hung worker burned ~eval_timeout CPU
                # Salvage siblings already finished; mark the rest None (candidate is
                # already doomed, so don't waste compute re-running them).
                for k in range(j + 1, n):
                    if k not in out and fut[k].done():
                        try:
                            c2, cpu2 = fut[k].result(timeout=0)
                            out[k] = (float(c2), float(cpu2))
                        except Exception:
                            out[k] = (None, 0.0)
                self._recreate_pool()
                for k in range(j + 1, n):
                    out.setdefault(k, (None, 0.0))
                break
            except concurrent.futures.process.BrokenProcessPool:
                print("    [eval] evaluator pool broke (worker died) — rebuilding, "
                      "candidate invalid.", flush=True)
                self._recreate_pool()
                return None
            except Exception:
                out[j] = (None, 0.0)              # heuristic crash on this instance (fast)
            j += 1
        return [out.get(k, (None, 0.0)) for k in range(n)]

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
# Logging: HiFo population dumps -> eoh_obp-style trajectory.json + heuristics.json
# --------------------------------------------------------------------------- #
_ARTIFACT_REFRESH_SECONDS = 30  # how often the background watcher rebuilds the logs


def _build_eoh_style_logs(log_dir: pathlib.Path, label: str, mean_opt: float,
                          n_instances: int, n_operators: int, pop_size: int,
                          n_init_batches: int = 2, wandb_logger=None,
                          verbose: bool = True) -> None:
    """Post-process HiFo's per-generation population dumps
    (``results/pops/population_generation_*.json``) into the same log surface
    ``eoh_tsp_gls.py`` writes: a per-generation best-so-far ``trajectory.json`` and a
    ``heuristics.json`` of the (deduplicated) candidates seen.

    Each dumped individual carries ``objective`` = mean tour cost (lower better, from
    ``_TSPAdapter.evaluate``). ``score`` = incumbent mean tour cost;
    ``gap`` = (score - mean_opt) / mean_opt (mean_opt = mean Concorde optimum).

    ``used_budget`` = cumulative candidate EVALUATIONS x n_instances, matching
    ``eoh_obp.py`` (which counts all sampled candidates). It is derived from HiFo's
    generation STRUCTURE, not from the dump sizes: the dumps hold only the survivors
    (after ``population_management`` drops None-objective candidates, dedups by
    objective, and keeps the ``pop_size`` best), so counting them would badly
    undercount the real budget. HiFo evaluates ``n_init_batches * pop_size``
    offspring for the initial population (gid 0) and ``n_operators * pop_size`` per
    later generation (one ``pop_size`` batch for each of the e1/e2/m1/m2/m3
    operators)."""
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
            mean_bins = float(obj) if isinstance(obj, (int, float)) else float("inf")
            if code not in code_to_cand:
                cand_id = f"gen{gid:02d}_cand{i:02d}"
                code_to_cand[code] = cand_id
                heuristics.append({"cand_id": cand_id, "gen_id": gid,
                                   "score": mean_bins, "source": code})
            per_gen[gid].append((code, mean_bins))

    # Real per-generation CPU seconds. The adapter appends one candidate's total CPU
    # (summed across its parallel instance workers) to eval_cpu.jsonl IN CANDIDATE ORDER
    # (init batches, then each generation's operator batches). So the cumulative CPU
    # through generation gid is the prefix-sum of the first ``cand_evaluated(gid)`` entries
    # — consistent with ``used_budget`` (which uses the same structural attempt count).
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
    # incumbent resolves to the SAME source _final_eval re-evaluates. With the full offspring
    # record, heuristics.json's cand_ids come from all_candidates.jsonl — a DIFFERENT namespace
    # than the survivor-dump `code_to_cand`; using the dump cand_id would point the incumbent at
    # a different heuristic (exploded valid scores). Falls back to the dump map for old runs.
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
        for code, mean_bins in per_gen[gid]:
            if math.isfinite(mean_bins) and mean_bins < best:
                best, best_code = mean_bins, code
        best_cand = _inc_cand_by_code.get(best_code) if best_code is not None else None
        score = best if best_code is not None else float("inf")
        gap = ((score - mean_opt) / mean_opt
               if (best_code is not None and mean_opt > 0) else float("inf"))
        # Candidates EVALUATED through this generation (attempts, incl. failed/dupes),
        # from HiFo's structure -- NOT the survivor count in the dumps.
        cand_evaluated = n_init_batches * pop_size + gid * n_operators * pop_size
        trajectory.append({
            "gen_id": gid,
            "cand_id": best_cand,
            "score": float(score),
            "gap": float(gap),
            "n_instances": n_instances,
            "used_budget": cand_evaluated * n_instances,
            # Cumulative CPU seconds (summed across cores) burned by ALL candidate
            # evaluations through this generation — from eval_cpu.jsonl (see above).
            "cpu_seconds": round(_cpu_through(cand_evaluated), 3),
        })

    # Atomic-ish writes (write to a temp then replace) so a reader/analysis never
    # sees a half-written file while the watcher rebuilds mid-run.
    for name, payload in (("trajectory.json", {"label": label, "trajectory": trajectory}),
                          ("heuristics.json", {"label": label,
                                               "total_sampled": len(heuristics),
                                               "heuristics": heuristics})):
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
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce HiFo-Prompt on TSP-GLS (benchmark defaults follow reprod/eoh_tsp_gls).")
    p.add_argument("--pop-size", type=int, default=10, help="HiFo ec_pop_size (population size).")
    p.add_argument("--max-generations", type=int, default=20, help="HiFo ec_n_pop (number of generations).")
    p.add_argument("--n-instances", type=int, default=16, help="Number of TSP instances.")
    p.add_argument("--problem-size", type=int, default=100, help="Cities per TSP instance.")
    p.add_argument("--selection-num", type=int, default=5,
                   help="HiFo ec_m: number of parents for e1/e2 (>=2).")
    p.add_argument("--eval-timeout", type=float, default=65.0,
                   help="PER-(heuristic, instance) wall-clock cap (s). Each of a "
                        "candidate's n_instances GLS runs is capped here (~65s = the GLS "
                        "60s/instance internal limit + slack); a heuristic that infinite-"
                        "loops inside update_edge_distance is reclaimed in this time (not "
                        "the old 65 x n_instances whole-candidate kill).")
    p.add_argument("--timeout-seconds", type=int, default=None,
                   help="DEPRECATED / ignored — the eval is now per-(heuristic, instance); "
                        "use --eval-timeout for the per-instance cap. Accepted for wrapper "
                        "back-compat only.")
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

    p.add_argument("--label", type=str, default="hifo/obp")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048, help="Max tokens for LLM generation.")
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"],
                   help="LLM backend (same options as reprod/eoh_obp.py).")
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next",
                   help="Model id passed to the selected backend.")
    p.add_argument("--ollama-host", type=str, default=None,
                   help="Ollama host:port (only for --llm-backend=ollama).")
    p.add_argument("--num-threads", type=int, default=4,
                   help="HiFo exp_n_proc: parallel offspring SAMPLER workers (joblib "
                        "threading backend). Mirrors reprod/eoh_tsp_gls's num_samplers.")
    p.add_argument("--num-cores", type=int, default=4,
                   help="Parallel EVALUATORS (process pool), mirroring reprod/eoh_tsp_gls's "
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

    n_op = 5  # HiFo default operators: e1, e2, m1, m2, m3
    print(f"pop_size={args.pop_size}  max_generations={args.max_generations}  "
          f"n_instances={args.n_instances}  problem_size={args.problem_size}")
    print(f"HiFo operators/gen = {n_op} (e1,e2,m1,m2,m3); nominal candidates ~ "
          f"{args.pop_size * n_op * args.max_generations}")

    # ---- LLM backend (identical to reprod/eoh_obp.py) -> CachedLLM -------- #
    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
    elif args.llm_backend == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            print("GOOGLE_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = GoogleClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
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
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args)}"
    wandb_logger = make_wandb_logger(enabled=args.use_wandb, project="llm4ad",
                                     name=run_name, config=args_dict)

    # ---- TSP-GLS evaluator (same instances/scoring as reprod/eoh_tsp_gls) - #
    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluation = TSP_GLS_2O_Evaluation_wo_Time()
    evaluation.n_instance = args.n_instances
    evaluation.problem_size = args.problem_size
    eval_timeout = args.eval_timeout if args.eval_timeout and args.eval_timeout > 0 else 65.0
    # The built-in dataset is (16, 100); regenerate for any other (n_instances, size).
    if (args.n_instances, args.problem_size) != (16, 100):
        evaluation._datasets = GetData(args.n_instances, args.problem_size).generate_instances()
    random.seed(args.seed)
    np.random.seed(args.seed)

    instances = list(evaluation._datasets)
    for i, inst in enumerate(instances):
        inst._id = i
    print(f"  instances: n={args.n_instances}  problem_size={args.problem_size}  "
          f"per-(heuristic,instance) eval timeout={eval_timeout}s")
    print("  solving per-instance optima with Concorde (for the gap metric)...")
    mean_opt = _mean_opt(instances)
    print(f"  mean Concorde optimum = {mean_opt:.4f}  (over {args.n_instances} instances "
          f"of {args.problem_size} cities)")

    # Per-(heuristic, instance) evaluator process pool (mirror of reprod/reevo_tsp_gls):
    # num_cores parallelises a candidate's n_instances GLS runs, each capped at
    # eval_timeout. --num-threads stays HiFo's sampler (LLM) concurrency.
    adapter = _TSPAdapter(instances, num_cores=args.num_cores, eval_timeout=eval_timeout,
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
    # task, gated by two hifo_hp knobs (read at call time inside hifo_interface_EC):
    #   EVAL_TIMEOUT = OUTER net on the eval CALL. The real per-instance bound is the
    #                  adapter's result(timeout=eval_timeout); a candidate's whole eval
    #                  runs its n_instances in ceil(n_instances/num_cores) waves, plus the
    #                  serialised-eval lock queue (up to num_threads candidates ahead).
    #   TASK_BUDGET  = whole-offspring budget = LLM (<= --llm-timeout) + eval net + margin.
    # Both are set generously so HiFo never kills a legitimately-working offspring; the
    # per-INSTANCE eval_timeout is the true cap that reclaims a hung heuristic.
    waves = math.ceil(args.n_instances / max(1, args.num_cores))
    per_cand_cap = waves * eval_timeout                          # one candidate's eval
    _hifo_hp.EVAL_TIMEOUT = int(max(1, args.num_threads) * per_cand_cap + 60)  # + lock-queue wait
    _hifo_hp.TASK_BUDGET = int(args.llm_timeout + _hifo_hp.EVAL_TIMEOUT + 30)
    print(f"HiFo timeouts: per-(heuristic,instance) cap = {eval_timeout}s | "
          f"per-candidate eval <= {per_cand_cap:.0f}s ({waves} waves x {eval_timeout}s) | "
          f"HiFo eval net = {_hifo_hp.EVAL_TIMEOUT}s | per-offspring budget = "
          f"{_hifo_hp.TASK_BUDGET}s")

    # ---- HiFo parameters ------------------------------------------------- #
    paras = Paras()
    paras.set_paras(
        method="hifo",
        problem=adapter,                # non-string -> HiFo's Probs uses it directly
        ec_pop_size=args.pop_size,
        ec_n_pop=args.max_generations,
        ec_m=max(2, args.selection_num),
        exp_n_proc=args.num_threads,
        exp_output_path=str(log_dir),
        exp_debug_mode=False,
        eva_timeout=_hifo_hp.EVAL_TIMEOUT,  # fallback; hifo_hp.EVAL_TIMEOUT/TASK_BUDGET drive it
        eva_numba_decorator=False,      # candidate update_edge_distance is plain numpy; no numba
        llm_use_local=False,
        llm_api_endpoint=args.llm_backend,   # unused (InterfaceLLM is bridged)
        llm_api_key="bridged",
        llm_model=cached.client.model,
    )

    # Per-(heuristic, instance) eval: --num-cores parallelises a candidate's n_instances
    # GLS runs, so num_cores > num_threads is now BENEFICIAL (more instance parallelism),
    # unlike the old per-candidate model. num_threads is the LLM sampler concurrency.
    print(f"[{args.label}] running HiFo (native engine, threading backend, "
          f"{args.num_threads} samplers, {args.num_cores} per-instance evaluators)...")
    t0 = time.time()

    # HiFo dumps results/pops/population_generation_*.json AFTER every generation,
    # but the eoh-style trajectory.json / heuristics.json were only built once, at
    # the very end. A HiFo run is slow (~n_op x pop_size LLM calls per generation),
    # so a long or interrupted run left NO trajectory.json / heuristics.json. Fix:
    # a background watcher rebuilds them from the per-gen dumps every few seconds
    # (idempotent: _build_eoh_style_logs reads all dumps and rewrites atomically),
    # and a final guaranteed build runs in `finally` even if the engine raises.
    _stop = threading.Event()

    def _log_watcher():
        while not _stop.wait(_ARTIFACT_REFRESH_SECONDS):
            try:
                _build_eoh_style_logs(log_dir, args.label, mean_opt, args.n_instances,
                                      n_operators=n_op, pop_size=args.pop_size,
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
                _fp = ROOT / "src" / "init_pop" / "hifo_tsp_gls.json"
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
        _build_eoh_style_logs(log_dir, args.label, mean_opt, args.n_instances,
                              n_operators=n_op, pop_size=args.pop_size,
                              wandb_logger=wandb_logger)
    dt = time.time() - t0
    print(f"[{args.label}] HiFo finished in {dt:.1f}s, {adapter.n_evals} evaluations")
    wandb_logger.finish()
    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
