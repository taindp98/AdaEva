"""Random Policy (Fixed-K) evaluation for EoH on FSSP-GLS.

This method evaluates every newly generated heuristic on exactly K instances.
It acts as a baseline fixed-allocation approach compared to the UCB-based MAB.

Per-instance GLS caps are overridden to match TSP-GLS: 60 s wall-clock and
1000 iterations (the FSSP evaluation module defaults to time_max=10 s; we
patch it to 60 s here).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import pathlib
import random
import concurrent.futures
import sys
import time
import yaml
from threading import Lock
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, make_wandb_logger, OpenRouterClient, OllamaClient,
                   MistralClient, vLLMClient)
from utils.llm import OpenRouterLLM4AD, OllamaLLM4AD, MistralLLM4AD, vLLMLLM4AD
from utils.logger import make_log_dir, mirror_stdout_to
from utils.fixed_init_pop import install_fixed_initial_population

from llm4ad.method.eoh import EoH
from llm4ad.method.eoh.profiler import EoHProfiler
from llm4ad.task.optimization.fssp_gls.evaluation import FSSP_GLS_Evaluation
from llm4ad.task.optimization.fssp_gls.get_instance import GetData
import llm4ad.task.optimization.fssp_gls.evaluation as _fssp_eval_mod


_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "pop_size": 10,
    "max_generations": 20,
    "n_instances": 64,
    "K": 1,
    "instance_mode": "fixed",
    "n_jobs": 50,
    "selection_num": 5,
    "num_threads": 4,
    "num_cores": 4,
    "llm_model": "qwen/qwen3-coder-next",
    "llm_backend": "openrouter",
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "pop_size": "ps", "max_generations": "mg", "n_instances": "ni", "K": "K",
    "instance_mode": "im",
    "n_jobs": "nj", "selection_num": "sn",
    "num_threads": "nt", "num_cores": "nc", "llm_model": "model",
    "llm_backend": "llm",
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


def save_run_log(log_dir: pathlib.Path, label: str, trajectory: list) -> None:
    path = log_dir / "trajectory.json"
    log = {"label": label, "trajectory": trajectory}
    json.dump(log, open(path, "w"), indent=2)
    print(f"  logged -> {path}  ({len(trajectory)} generations)")


# ----------------------------------------------------------------------------- #
# K-instance subset selection (fixed | random)
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
# its own worker), mirroring src/tiny/hifo_fssp_gls.py. Workers cache the FULL instance
# pool once (initializer) and re-set the module GLS caps (the module defaults to
# time_max=10s; each worker imports it fresh, so the 60s cap set in main() must be
# re-applied IN THE WORKER). Only ``(code, inst_idx)`` crosses the pickle boundary per task.
#
# Numerics are preserved vs the previous per-heuristic path: the base FSSP evaluation has
# use_numba_accelerate / use_protected_div off and random_seed=None, so SecureEvaluator's
# _modify_program_code is a no-op — compiling the raw ``program_str`` here yields the SAME
# callable, ``solve_without_time`` is the SAME per-instance solver, and the candidate score
# is the SAME NEGATED mean makespan over the SAME K-subset instances.
# --------------------------------------------------------------------------- #
_FSSP_TIME_MAX = 60.0
_FSSP_ITER_MAX = 1000

_W_INSTANCES = None   # per-worker cache of the FULL FSSP instance list (set by initializer)


def _fssp_worker_init(instances):
    global _W_INSTANCES
    _W_INSTANCES = instances
    # solve_without_time reads the GLS caps from module globals at call time; the worker
    # imports fssp_gls.evaluation fresh (defaults 10s), so re-set them to the main-process
    # values (60s / 1000 iters) IN THE WORKER PROCESS.
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
    """Evaluate ONE candidate on ONE instance -> ``makespan`` (+inf on a heuristic crash).
    ``args`` is ``(code, inst_idx)``; the instance comes from the worker global. GLS is
    internally capped; the OUTER per-instance ``result(timeout=...)`` in the parent reclaims
    a heuristic that infinite-loops inside ``get_matrix_and_jobs`` before that check."""
    from llm4ad.task.optimization.fssp_gls.evaluation import solve_without_time
    code, inst_idx = args
    fn = _fssp_compile_update(code)   # exec is ~µs
    c = solve_without_time(_W_INSTANCES[inst_idx], fn)   # inf on internal crash
    return float(c)


class _SubsetFSSPEvaluation(FSSP_GLS_Evaluation):
    """``FSSP_GLS_Evaluation`` restricted to a K-instance subset, evaluated at
    per-(heuristic, instance) granularity via a shared process pool (mirrors
    src/tiny/hifo_fssp_gls.py).

    ``instance_mode``:
      - ``fixed``: one K-subset, sampled once (seeded by ``--seed``) before evolution and
        reused for EVERY heuristic.
      - ``random``: a fresh K-subset is drawn from the FULL pool on every heuristic
        evaluation, seeded reproducibly by the heuristic source. Either way each heuristic
        is evaluated on ALL K instances of its subset.

    ``evaluate_program`` submits the candidate's K subset instances as independent
    ``(code, inst_idx)`` tasks to a shared ``ProcessPoolExecutor(num_cores)``, each capped
    by a PER-INSTANCE ``result(timeout=eval_timeout)``; a hung instance is reclaimed (pool
    rebuilt) and the candidate is marked invalid. It is serialised with a lock so EoH's
    concurrent sampler threads (num_evaluators=1, thread-dispatch) submit one candidate's
    fan-out at a time. The FULL pool is always shipped to workers (initializer); the
    per-candidate K-subset is selected here as INDICES into it. The returned score is the
    NEGATED mean makespan (higher = better), matching the base FSSP evaluation."""

    _instance_mode = "fixed"
    _full_pool = None

    def _configure_subset(self, full_list: list, k: int, mode: str, seed: int,
                          problem_size: int):
        self._instance_mode = mode
        self._k = int(k)
        self._subset_seed = int(seed)
        self.problem_size = problem_size
        full = list(full_list)
        self._full_pool = full                 # always cached in workers (per-instance tasks)
        if mode == "fixed":
            idx = _sample_subset_idx(len(full), self._k, seed)
            self._datasets = [full[i] for i in idx]
            self.n_instance = len(self._datasets)
            self._fixed_idx = idx
            print(f"  [instance-mode=fixed] sampled K={len(idx)} of {len(full)} "
                  f"instances once (seed={seed}): idx={idx}")
        else:
            self._fixed_idx = None
            print(f"  [instance-mode=random] K={self._k} of {len(full)} instances "
                  f"resampled per heuristic (seeded by source ^ {seed})")
        return self

    def _configure_pool(self, num_cores: int, eval_timeout: float = 65.0):
        """Create the shared per-instance evaluator pool. ``eval_timeout`` is the PER-INSTANCE
        wall-clock cap (65s matches the HiFo tiny default: GLS is internally ~60s, the outer
        timeout reclaims an infinite-looping heuristic before that check)."""
        self.num_cores = max(1, int(num_cores))
        self.eval_timeout = (float(eval_timeout)
                             if eval_timeout and float(eval_timeout) > 0 else None)
        self._pool_init = (_fssp_worker_init, (self._full_pool,))
        self._eval_pool = None
        self._eval_lock = Lock()
        self._start_pool()
        return self

    def _subset_idx(self, code: str) -> list:
        """The K instance indices this candidate is scored on (into the FULL pool)."""
        if self._instance_mode == "random":
            seed = _program_subset_seed(code, self._subset_seed)
            return _sample_subset_idx(len(self._full_pool), self._k, seed)
        return self._fixed_idx if self._fixed_idx is not None else list(range(self._k))

    def _start_pool(self):
        self._eval_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_cores,
            initializer=self._pool_init[0], initargs=self._pool_init[1])
        try:
            list(self._eval_pool.map(int, range(self.num_cores)))
        except Exception:
            pass

    def _recreate_pool(self):
        """Terminate the current workers (some may be stuck in an infinite-loop heuristic)
        and start a fresh pool. Mirrors src/tiny/hifo_fssp_gls.py."""
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

    def _map_instances_with_timeout(self, code, idxs):
        """Submit the candidate's K subset GLS tasks; collect each as a makespan with a
        per-instance timeout. On a genuine hang, salvage finished siblings, rebuild the pool
        and short-circuit. Returns a list of ``makespan_or_None`` aligned to ``idxs``, or
        ``None`` if the pool broke."""
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
                out[ji] = float(fut[ji].result(timeout=to))
            except concurrent.futures.TimeoutError:
                print(f"    [eval] instance {ji} exceeded {to}s "
                      f"(heuristic likely infinite-loops in get_matrix_and_jobs) -> "
                      f"candidate invalid; rebuilding pool", flush=True)
                out[ji] = None
                for jk in order[pos + 1:]:
                    if jk not in out and fut[jk].done():
                        try:
                            out[jk] = float(fut[jk].result(timeout=0))
                        except Exception:
                            out[jk] = None
                self._recreate_pool()
                for jk in order[pos + 1:]:
                    out.setdefault(jk, None)
                break
            except concurrent.futures.process.BrokenProcessPool:
                print("    [eval] evaluator pool broke (worker died) — rebuilding, "
                      "candidate invalid.", flush=True)
                self._recreate_pool()
                return None
            except Exception:
                out[ji] = None                 # heuristic crash on this instance (fast)
            pos += 1
        return [out.get(ji) for ji in order]

    def evaluate_program(self, program_str, callable_func, **kwargs):
        """Score the candidate as the NEGATED mean makespan over its K subset instances, run
        per-(heuristic, instance) on the shared pool. Returns ``None`` (invalid) if any
        subset instance crashes / times out — matching the base ``evaluate_without_time``
        (which propagates +inf into the mean) and the EoH convention that a non-finite score
        is rejected. Negated mean (higher = better), same as the base FSSP evaluation."""
        with self._eval_lock:
            idxs = self._subset_idx(program_str)
            makespans = self._map_instances_with_timeout(program_str, idxs)
            if makespans is None:              # pool broke mid-eval
                return None
            if any(c is None or not math.isfinite(c) for c in makespans):
                return None
            return -float(np.mean(makespans))  # NEGATED mean makespan over the K subset

    def shutdown(self):
        """Tear down the evaluator process pool (call once the EoH run finishes)."""
        pool = getattr(self, "_eval_pool", None)
        if pool is None:
            return
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=True)
        except Exception:
            pass


# ----------------------------------------------------------------------------- #
# Recording profiler
# ----------------------------------------------------------------------------- #
class RecordingProfiler(EoHProfiler):
    def __init__(self, label: str, log_dir: pathlib.Path, pop_size: int,
                 n_instance: int, wandb_logger=None):
        super().__init__(log_dir=None, create_random_path=False)
        self.label = label
        self.log_dir = log_dir
        self.pop_size = pop_size
        self.n_instance = n_instance
        self.records: List[dict] = []
        self._heuristics: List[dict] = []
        self.times: List[float] = []
        self.populations: Dict[int, list] = {}
        self._t0 = time.time()
        self._last_pop_gen = 0
        self._gen_lock = Lock()
        self.wandb_logger = wandb_logger
        self._wandb_logged_rows = 0

    def record_parameters(self, *args, **kwargs): pass

    def register_function(self, func, program=None, *args, **kwargs):
        s = func.score if (func is not None and func.score is not None) else float("-inf")
        try:
            src = str(func) if func is not None else ""
        except Exception:
            src = ""
        # Per-candidate evaluation wall time (LLM4AD EoH sets func.evaluate_time =
        # wall of the whole subprocess eval over all n_instances). The eval is
        # CPU-bound single-threaded numba, so wall ~= CPU per candidate; summing
        # across candidates (which run in parallel workers) gives the total CPU
        # consumed, summed across cores.
        et = getattr(func, "evaluate_time", None)
        eval_time = float(et) if et is not None else 0.0
        with self._gen_lock:
            self.records.append({"gen_id": None, "score": float(s), "eval_time": eval_time})
            self._heuristics.append({"gen_id": None, "score": float(s), "source": src})
        elapsed = time.time() - self._t0
        self.times.append(elapsed)
        dt = elapsed - (self.times[-2] if len(self.times) > 1 else 0.0)
        if math.isfinite(s):
            mean_makespan = -float(s)
            tag = f"score={s:.4f} mean_makespan={mean_makespan:.4f}"
        else:
            tag = "FAILED (score=-inf)"
        print(f"  [pending] {tag}  +{dt:.1f}s  (elapsed {elapsed:.0f}s)", flush=True)

    def register_population(self, pop):
        try:
            cur = int(pop.generation)
        except Exception:
            return
        with self._gen_lock:
            advanced = cur > self._last_pop_gen
            for r in self.records:
                if r["gen_id"] is None:
                    r["gen_id"] = cur
            for h in self._heuristics:
                if h["gen_id"] is None:
                    h["gen_id"] = cur
            self._last_pop_gen = cur
            need_snapshot = advanced and cur not in self.populations
        if need_snapshot:
            self._snapshot_population(pop, cur)
            self._checkpoint()

    def _snapshot_population(self, pop, gen_id: int):
        # Capture each survivor's IDENTITY (source) + score, not just the score, so the
        # environmental-selection survivor set entering a generation can be matched back to
        # heuristics.json cand_ids (ranking analysis: parent+offspring selection pool).
        survivors = []
        for f in pop.population:
            try:
                src = str(f)
            except Exception:
                src = ""
            survivors.append({"source": src,
                              "score": (float(f.score) if f.score is not None
                                        else float("-inf"))})
        with self._gen_lock:
            self.populations[gen_id] = survivors

    def finalize_population(self, pop):
        try:
            cur = int(pop.generation)
        except Exception:
            return
        with self._gen_lock:
            if cur in self.populations:
                return
        self._snapshot_population(pop, cur)
        self._checkpoint()

    def _checkpoint(self):
        with self._gen_lock:
            snapshot_records = list(self.records)
            snapshot_heuristics = list(self._heuristics)
        finalized: list = []
        finalized_h: list = []
        for i, r in enumerate(snapshot_records):
            if r.get("gen_id") is not None:
                finalized.append(r)
                finalized_h.append(snapshot_heuristics[i])
        init_records: list = []
        init_h: list = []
        valid_count = 0
        evo_start = 0
        for i, r in enumerate(finalized):
            init_records.append(r)
            init_h.append(finalized_h[i])
            if math.isfinite(r["score"]):
                valid_count += 1
            if valid_count >= self.pop_size:
                evo_start = i + 1
                break
        if valid_count < self.pop_size:
            evo_start = len(finalized)
        evo_records = finalized[evo_start:]
        evo_h = finalized_h[evo_start:]
        groups: Dict[int, list] = {0: init_records}
        heur_groups: Dict[int, list] = {0: init_h}
        for k in range(0, len(evo_records), self.pop_size):
            chunk = evo_records[k:k + self.pop_size]
            if len(chunk) < self.pop_size:
                break
            gid = 1 + k // self.pop_size
            groups[gid] = chunk
            heur_groups[gid] = evo_h[k:k + self.pop_size]

        trajectory = []
        best_mean_makespan = float("inf")
        best_cand = None
        cum_budget = 0
        cum_cpu = 0.0   # cumulative CPU seconds (summed across cores) over ALL candidates
        for gid in sorted(groups.keys()):
            chunk = groups[gid]
            cum_budget += len(chunk) * self.n_instance
            cum_cpu += sum((r.get("eval_time") or 0.0) for r in chunk)
            for i, r in enumerate(chunk):
                s = r["score"]
                if math.isfinite(s):
                    mean_makespan = -float(s)
                    if mean_makespan < best_mean_makespan:
                        best_mean_makespan = mean_makespan
                        best_cand = f"gen{gid:02d}_cand{i:02d}"
            score = best_mean_makespan if best_cand is not None else float("inf")
            trajectory.append({
                "gen_id": gid,
                "cand_id": best_cand,
                "score": float(score),
                "n_instances": self.n_instance,
                "used_budget": cum_budget,
                # Cumulative CPU seconds (summed across cores) burned by ALL
                # candidate evaluations up to this generation — "how costly it
                # was to discover this incumbent".
                "cpu_seconds": round(cum_cpu, 3),
            })
        save_run_log(self.log_dir, self.label, trajectory)
        for row in trajectory[self._wandb_logged_rows:]:
            self.wandb_logger.log_trajectory_row(row)
        self._wandb_logged_rows = len(trajectory)
        self._save_evaluations_per_gen(groups)
        self._save_heuristics(heur_groups)
        self._save_survivors(heur_groups)

    def _save_survivors(self, heur_groups: Dict[int, list]) -> None:
        """Dump the environmental-selection SURVIVOR set per generation to survivors.json.

        ``self.populations[g]`` holds the survivors kept AFTER generation g's survival()
        (each: source + score). We map every survivor's ``source`` back to its cand_id, so
        downstream analysis can form the true parent+offspring selection pool
        (survivors(g-1) + offspring(g)) by cand_id.

        The source->cand_id map MUST match heuristics.json's ``gen{gid}_cand{i}`` scheme, so we
        reproduce ``_checkpoint``'s chunking over the full per-sample record
        (``self._heuristics``): gen0 = the first pop_size VALID (finite-score) samples, then
        consecutive pop_size chunks. Unlike heuristics.json — which DROPS the trailing
        <pop_size chunk — we also index that partial trailing chunk so a freshly-sampled
        surviving elite still resolves instead of logging cand_id=None. A survivor whose source
        is genuinely unlogged stays cand_id=None. (Survivor keys are EoH's pop.generation, which
        can be offset from the chunk gid, e.g. fix-init; mapping is by SOURCE so that's fine.)"""
        with self._gen_lock:
            full_h = list(self._heuristics)
            pops = {g: list(s) for g, s in self.populations.items()}
        # Key by a DOCSTRING-INSENSITIVE canonical source: survivors from pop.population have
        # their docstring stripped in str(f) while register_function logged str(func) WITH the
        # docstring, so a raw-string match misses exactly those candidates (cand_id=None).
        from utils.source_key import canon_source
        finalized = [h for h in full_h if h.get("gen_id") is not None]
        src_to_cid: Dict[str, str] = {}

        def _assign(gid: int, entries: list) -> None:
            for i, h in enumerate(entries):
                src_to_cid.setdefault(canon_source(h.get("source", "")), f"gen{gid:02d}_cand{i:02d}")

        valid_count = 0
        evo_start = len(finalized)
        for idx, h in enumerate(finalized):
            if math.isfinite(h.get("score", float("-inf"))):
                valid_count += 1
            if valid_count >= self.pop_size:
                evo_start = idx + 1
                break
        _assign(0, finalized[:evo_start])                     # gen0 = initial population
        evo_h = finalized[evo_start:]
        for k in range(0, len(evo_h), self.pop_size):          # gen1.. = pop_size chunks
            _assign(1 + k // self.pop_size, evo_h[k:k + self.pop_size])

        out: Dict[str, list] = {}
        for gid in sorted(pops.keys()):
            rows = []
            for s in pops[gid]:
                cid = src_to_cid.get(canon_source(s.get("source", "")))
                rows.append({"cand_id": cid, "score": float(s["score"])})
            out[str(gid)] = rows
        payload = {"label": self.label, "pop_size": self.pop_size, "survivors_by_gen": out}
        json.dump(payload, open(self.log_dir / "survivors.json", "w"), indent=2)

    def _save_evaluations_per_gen(self, groups: Dict[int, list]) -> None:
        entries = []
        for gid in sorted(groups.keys()):
            n_heur = len(groups[gid])
            entries.append({
                "gen_id": gid,
                "num_heuristics": n_heur,
                "n_instance": self.n_instance,
                "total_instance_evals": n_heur * self.n_instance,
            })
        payload = {
            "label": self.label,
            "n_instance": self.n_instance,
            "cumulative_instance_evals": sum(e["total_instance_evals"] for e in entries),
            "generations": entries,
        }
        json.dump(payload, open(self.log_dir / "evaluations_per_gen.json", "w"), indent=2)

    def _save_heuristics(self, heur_groups: Dict[int, list]) -> None:
        out: List[dict] = []
        for gid in sorted(heur_groups.keys()):
            for i, h in enumerate(heur_groups[gid]):
                out.append({"cand_id": f"gen{gid:02d}_cand{i:02d}",
                            "gen_id": gid,
                            "score": float(h["score"]),
                            "source": h["source"]})
        payload = {"label": self.label, "total_sampled": len(out), "heuristics": out}
        json.dump(payload, open(self.log_dir / "heuristics.json", "w"), indent=2)

    def finish(self, *args, **kwargs): pass


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def _final_eval(log_dir: pathlib.Path, num_cores: int, n_jobs: int = 50, m_low: int = 2, m_high: int = 20, instance_source: str = "pregenerated") -> dict:
    """Post-evolution evaluation using src/analyses/eval_fssp.py with mode='valid' and has_incumbent=True.
    Saves output to valid_trajectory.json using num_cores."""
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-K random policy evaluation for EoH on FSSP-GLS.")
    p.add_argument("--pop-size", type=int, default=10,
                   help="Population size.")
    p.add_argument("--max-generations", type=int, default=20,
                   help="Number of evolution generations.")
    p.add_argument("--n-instances", type=int, default=64,
                   help="Number of FSSP instances in the generated pool.")
    p.add_argument("--K", type=int, default=1,
                   help="Number of instances to evaluate each heuristic on (default: 1).")
    p.add_argument("--instance-mode", type=str, default="fixed", choices=["fixed", "random"],
                   help="K-subset selection from the --n-instances pool. 'fixed' "
                        "(default): sample K instances once (seeded by --seed) before "
                        "evolution and reuse for every heuristic. 'random': sample a "
                        "fresh K-subset per heuristic evaluation (seeded reproducibly "
                        "by the heuristic source). Both evaluate each heuristic on all "
                        "K instances of its subset.")
    p.add_argument("--n-jobs", type=int, default=50,
                   help="Jobs per instance (paper: 50).")
    p.add_argument("--m-low", type=int, default=2,
                   help="Minimum machines per instance (paper: 2).")
    p.add_argument("--m-high", type=int, default=20,
                   help="Maximum machines per instance (paper: 20).")
    p.add_argument("--instance-source", type=str, default="pregenerated",
                   choices=["pregenerated", "synthetic"],
                   help="FSSP instance source. 'pregenerated' (default) loads the "
                        "fixed pre-generated benchmark instances from disk (integer "
                        "processing times); 'synthetic' generates U[0,1] instances "
                        "with --n-jobs jobs and machines drawn uniformly in "
                        "[--m-low, --m-high].")
    p.add_argument("--selection-num", type=int, default=5,
                   help="Parents p for E1/E2 prompts (paper: 5).")
    p.add_argument("--timeout-seconds", type=int, default=None,
                   help="Per-candidate subprocess timeout. Default: 65 × K.")
    p.add_argument("--label", type=str, default="eoh/tiny_fssp_gls")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048, help="Max tokens for LLM generation.")
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm"],
                   help="LLM backend.")
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next",
                   help="Model id for the selected backend.")
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=4,
                   help="Parallel samplers for LLM API.")
    p.add_argument("--num-cores", type=int, default=4,
                   help="Parallel evaluators (process pool).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed generation 0 from the paired fixed population in "
                        "src/init_pop/ (re-evaluated this run) instead of sampling it "
                        "from the LLM.")
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
    log_dir = make_log_dir(args.log_root, args.label, dt_stamp, args.seed,
                           tag=_run_tag(args))
    cache_dir = args.cache_root / dt_stamp / str(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mirror_stdout_to(log_dir / "terminal.txt")
    print(f"terminal output mirrored -> {log_dir / 'terminal.txt'}")

    target_budget = args.pop_size * args.max_generations * args.n_instances
    effective_max_generations = math.ceil(target_budget / (args.pop_size * args.K))
    total_samples = args.pop_size * effective_max_generations
    budget = total_samples * args.K
    print(f"pop_size={args.pop_size}  target_budget={target_budget}  "
          f"effective_max_generations={effective_max_generations}  "
          f"pool_instances={args.n_instances}  K={args.K}  n_jobs={args.n_jobs}  "
          f"m_range=[{args.m_low},{args.m_high}]")
    print(f"total candidates = {total_samples}, function evaluations = {budget}")

    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model,
                              timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = OllamaLLM4AD(cached)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = MistralLLM4AD(cached)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = vLLMLLM4AD(cached)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label,
                                  model=args.llm_model)
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = OpenRouterLLM4AD(cached)
    print(f"Backend: {args.llm_backend}  model: {cached.client.model}  cache: {cache_dir}")

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    args_dict["llm_model"] = cached.client.model
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args)}"
    wandb_logger = make_wandb_logger(
        enabled=args.use_wandb,
        project="llm4ad",
        name=run_name,
        config=args_dict,
    )

    # GetData seeds np.random internally; re-seed after
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Override per-instance GLS time cap to 60s (matching TSP-GLS's
    # solve_without_time). The FSSP evaluation module defaults to time_max=10s;
    # iter_max=1000 is already correct. (Each per-instance worker re-applies these in
    # _fssp_worker_init since it imports the module fresh; this sets them for the MAIN
    # process, used by _final_eval.)
    _fssp_eval_mod.time_max = _FSSP_TIME_MAX
    _fssp_eval_mod.iter_max = _FSSP_ITER_MAX

    evaluation = _SubsetFSSPEvaluation()

    # Generate the global pool of instances (for consistency across methods).
    # Source selected by --instance-source (default 'pregenerated' = load the
    # fixed benchmark files; 'synthetic' = generate U[0,1] with n_jobs jobs and
    # machines in [m_low, m_high]). n_jobs/m_low/m_high are ignored by GetData in
    # pregenerated mode (the files carry their own dimensions).
    full_datasets = GetData(
        args.n_instances, n_jobs=args.n_jobs, m_low=args.m_low, m_high=args.m_high,
        use_pregenerated=(args.instance_source == "pregenerated"),
    ).generate_instances()

    # Restrict to a K-instance subset (fixed once, or resampled per heuristic).
    evaluation._configure_subset(full_datasets, args.K, args.instance_mode,
                                 args.seed, args.n_jobs)

    # Per-(heuristic, instance) parallelism: EoH now dispatches CANDIDATES on threads
    # (num_evaluators=1, safe_evaluate off) and this evaluation fans each candidate's K
    # instances onto a shared ProcessPoolExecutor(num_cores). The 65s cap is PER INSTANCE
    # (matching the HiFo tiny default) rather than the old per-candidate 65s×K.
    per_instance_timeout = 65.0
    evaluation.safe_evaluate = False      # our per-instance pool owns timeouts + kills
    evaluation.timeout_seconds = None
    evaluation._configure_pool(num_cores=args.num_cores, eval_timeout=per_instance_timeout)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"  instances: evaluated K={args.K}  n_jobs={args.n_jobs}  "
          f"machines=[{args.m_low},{args.m_high}]  instance_mode={args.instance_mode}  "
          f"parallel=<heuristic,instance> on {args.num_cores} cores, "
          f"timeout={per_instance_timeout}s/instance")

    prof = RecordingProfiler(label=args.label, log_dir=log_dir,
                             pop_size=args.pop_size, n_instance=args.K,
                             wandb_logger=wandb_logger)
    method = EoH(
        llm=llm,
        profiler=prof,
        evaluation=evaluation,
        # +1 compensates for EoH's phantom survival generation.
        max_generations=effective_max_generations + 1,
        max_sample_nums=10**9,
        pop_size=args.pop_size,
        selection_num=args.selection_num,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,
        num_samplers=args.num_threads,
        # Candidate dispatch is now THREADS with a single evaluator; the real CPU
        # parallelism lives in this evaluation's shared per-instance ProcessPoolExecutor
        # (num_cores). safe_evaluate is off (this evaluation owns timeouts/kills), so a
        # nested framework process pool would only oversubscribe cores.
        num_evaluators=1,
        multi_thread_or_process_eval="thread",
        debug_mode=False,
    )

    print(f"[{args.label}] running EoH (budget = {budget} evaluations)...")
    t0 = time.time()
    if args.fix_init_pop:
        install_fixed_initial_population(method, ROOT / "src" / "init_pop" / "eoh_fssp_gls.json")
    try:
        method.run()
    finally:
        evaluation.shutdown()
    dt = time.time() - t0
    print(f"[{args.label}] done in {dt:.1f}s, recorded {len(prof.records)} candidates")
    try:
        prof.finalize_population(method._population)
    except Exception as e:
        print(f"  [finalize_population] skipped: {type(e).__name__}: {e}")
    prof._checkpoint()
    try:
        _final_eval(log_dir=log_dir, num_cores=args.num_cores, n_jobs=args.n_jobs, m_low=args.m_low, m_high=args.m_high, instance_source=args.instance_source)
    except Exception as e:
        print(f"  [_final_eval] WARN: final evaluation failed: {type(e).__name__}: {e}", flush=True)
    wandb_logger.finish()
    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
