"""Re-evaluate BBOB trajectory checkpoints on all 72 BBOB functions for EoH & OpenEvolve.

The EoH / OpenEvolve counterpart of ``eval_bbob.py``, for function-based BBOB runs
(``reprod/eoh_bbob.py``, ``racing/eoh_bbob.py``, ``reprod/openevolve_bbob.py``,
``racing/openevolve_bbob.py``). Same resumable, parallel structure, with function-based
heuristics:
  - The heuristic is a FUNCTION ``optimize(func, budget, dim)`` (not a class).
    Scored with ``ioh`` + the AOCC logger, matching ``reprod/eoh_bbob.py``.
  - The instance pool is the fixed 72 = 24 fids x 3 iids problems (all noiseless
    BBOB functions) at a single ``--dim``, each run ``--n-rep`` times (default 1).
  - The metric is mean AOCC (Area Over the Convergence Curve) in [0, 1], HIGHER is
    better (1.0 = optimum found instantly) — so ``best`` is a MAX.
  - A per-(candidate, instance) ``--eval-timeout`` (SIGALRM) guards against
    non-terminating loops; a timed-out run is scored on its PARTIAL AOCC with a
    WARNING naming the heuristic + instance.
"""

import math
import multiprocessing as mp
import numpy as np
import pandas as pd
import pathlib
import signal
import sys
import threading
import time
import warnings
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD"), str(ROOT / "packages" / "LLaMEA")]

from ioh import get_problem, logger
from misc import OverBudgetException, aoc_logger, correct_aoc

# Fixed BBOB evaluation parameters (match reprod/eoh_bbob.py)
_FIDS = range(1, 25)
_IIDS = (1, 2, 3)
_AOC_LOWER = 1e-8
_AOC_UPPER = 1e2
_N_FUNCTIONS = len(_FIDS) * len(_IIDS)  # 72


def _auto_eval_timeout(budget_factor: int, dim: int) -> float:
    """Per-(candidate, instance) wall-clock cap that scales with the func-eval
    budget (= budget_factor * dim). Mirrors reprod/eoh_bbob._auto_eval_timeout_per_instance:
    dim 5 -> 160s, dim 20 -> 460s."""
    return 60.0 + float(budget_factor) * float(dim) / 100.0


class _EvalTimeout(Exception):
    """Raised when a single BBOB eval exceeds its wall-clock cap (SIGALRM)."""


def _compile_bbob_func(source: str):
    """Exec the heuristic source and return the optimizer FUNCTION (e.g. ``optimize(func, budget, dim)``).
    Falls back to the first top-level function defined in source."""
    ns = {"np": np, "numpy": np}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(source, ns)
    if "optimize" in ns and callable(ns["optimize"]):
        return ns["optimize"]
    funcs = [v for k, v in ns.items()
             if callable(v) and not isinstance(v, type) and not k.startswith("_")]
    if not funcs:
        raise ValueError("no optimizer function defined in source")
    return funcs[0]


def score_bbob_inst(opt_func, fid: int, iid: int, dim: int, budget: int, rep: int,
                    eval_timeout: float = None, cand_id: str = "") -> tuple:
    """Run one heuristic function on one (fid, iid, rep) BBOB problem; return (AOCC, runtime).

    AOCC in [0, 1], higher = better. ``np.random.seed(rep)`` makes the (stochastic)
    optimizer reproducible (matches training). A crash -> AOCC 0.0 (worst valid). A
    timeout -> PARTIAL AOCC (``correct_aoc`` extrapolates from best-so-far) + WARNING.
    """
    t0 = time.perf_counter()
    l2 = aoc_logger(budget, lower=_AOC_LOWER, upper=_AOC_UPPER, triggers=[logger.trigger.ALWAYS])
    problem = get_problem(fid, iid, dim)
    problem.attach_logger(l2)
    np.random.seed(rep)

    use_alarm = bool(eval_timeout) and float(eval_timeout) > 0 and hasattr(signal, "SIGALRM")
    old_handler = None
    if use_alarm:
        def _on_alarm(signum, frame):
            raise _EvalTimeout()
        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, float(eval_timeout))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                opt_func(problem, budget, dim)
            except TypeError:
                try:
                    opt_func(problem, dim=dim, budget=budget)
                except TypeError:
                    opt_func(problem)
    except OverBudgetException:
        pass                       # func-eval budget exhausted -> score partial AOCC
    except _EvalTimeout:
        print(f"  [WARNING] eval TIMEOUT: heuristic '{cand_id}' on instance "
              f"(fid={fid}, iid={iid}, rep={rep}) exceeded {float(eval_timeout):.0f}s "
              f"-> partial AOCC", flush=True)
    except Exception:
        return 0.0, time.perf_counter() - t0   # crash -> worst valid AOCC
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    try:
        aocc = float(correct_aoc(problem, l2, budget))
    except Exception:
        aocc = 0.0
    if not math.isfinite(aocc):
        aocc = 0.0
    return aocc, time.perf_counter() - t0


def build_tasks(n_rep: int = 1) -> list:
    """The 72 * n_rep BBOB (fid, iid, rep) tasks (all 24 functions x 3 instances)."""
    return [(fid, iid, rep) for fid in _FIDS for iid in _IIDS for rep in range(n_rep)]


def single_batch_scoring(tasks: list, source: str, dim: int, budget: int,
                         eval_timeout: float = None) -> tuple:
    """Score one heuristic function across all tasks. Returns (mean AOCC, mean runtime)."""
    try:
        opt_func = _compile_bbob_func(source)
    except Exception as e:
        print(f"  [compile] error: {e}", flush=True)
        return 0.0, 0.0
    aoccs, rts = [], []
    for fid, iid, rep in tasks:
        a, rt = score_bbob_inst(opt_func, fid, iid, dim, budget, rep, eval_timeout)
        aoccs.append(a); rts.append(rt)
    return float(np.mean(aoccs)), float(np.mean(rts))


def _worker(args):
    idx, cand_id, source, tasks, dim, budget, eval_timeout = args
    pid = mp.current_process().pid
    t0 = time.perf_counter()
    try:
        opt_func = _compile_bbob_func(source)
    except Exception as e:
        print(f"  [worker pid={pid}] {cand_id}: compile error: {e}", flush=True)
        return idx, 0.0, 0.0
    aoccs, rts = [], []
    for k, (fid, iid, rep) in enumerate(tasks):
        a, rt = score_bbob_inst(opt_func, fid, iid, dim, budget, rep, eval_timeout, cand_id)
        aoccs.append(a); rts.append(rt)
        if (k + 1) % 12 == 0 or (k + 1) == len(tasks):
            print(f"    [worker pid={pid}] {cand_id}  task {k+1}/{len(tasks)}  "
                  f"last_AOCC={a:.4f} (f{fid}i{iid}r{rep})  "
                  f"running_mean_AOCC={float(np.mean(aoccs)):.4f}  "
                  f"worker_wall={time.perf_counter()-t0:.0f}s", flush=True)
    score = float(np.mean(aoccs)); runtime = float(np.mean(rts))
    print(f"  [worker pid={pid}] {cand_id}: DONE  mean_AOCC={score:.4f}  "
          f"wall={time.perf_counter()-t0:.1f}s", flush=True)
    return idx, score, runtime


def batch_scoring_parallel(tasks: list, heuristics: list, dim: int, budget: int,
                           eval_timeout: float, n_cores: int = 20) -> tuple:
    work = [(i, h["cand_id"], h["source"], tasks, dim, budget, eval_timeout)
            for i, h in enumerate(heuristics)]
    scores = [0.0] * len(heuristics)
    runtimes = [0.0] * len(heuristics)
    n_total = len(heuristics)
    n_done = 0
    t0 = time.time()

    _stop_heartbeat = threading.Event()
    def _heartbeat():
        while not _stop_heartbeat.wait(120):
            print(f"  [heartbeat] {n_done}/{n_total} done  {n_total - n_done} in-flight  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=n_cores) as pool:
            for idx, score, runtime in pool.imap_unordered(_worker, work, chunksize=1):
                scores[idx] = score
                runtimes[idx] = runtime
                n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(f"  [{n_done}/{n_total}] done  elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    finally:
        _stop_heartbeat.set()

    return scores, runtimes


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _out_path(exp_path, mode, dim, n_rep, has_incumbent):
    inc = "_incumbent" if has_incumbent else ""
    return exp_path / f"{mode}_trajectory{inc}_bbob_d{dim}_nrep{n_rep}.json"


def evaluate_single(exp_path: pathlib.Path, dim: int = 5, budget_factor: int = 2000,
                    n_rep: int = 1, eval_timeout: float = -1.0, n_cores: int = 20,
                    mode: str = "valid", has_incumbent: bool = False, incumbent_key: str = "incumbents_mean_rank") -> dict:
    """Evaluate the trajectory of function-based BBOB heuristics over all 72 BBOB functions.

    Resumable: on interrupt only unique unscored candidates are re-evaluated, then
    the output file is updated. Metric is mean AOCC (higher is better).
    """
    t_start = time.time()
    budget = budget_factor * dim
    # Resolve --eval-timeout: -1 => auto-scale with budget; 0 => disabled; >0 => fixed.
    if eval_timeout is not None and eval_timeout < 0:
        eval_timeout = _auto_eval_timeout(budget_factor, dim)
        print(f"  [eval-timeout] auto = 60 + budget_factor({budget_factor}) x dim({dim})/100 "
              f"= {eval_timeout:.0f}s per (candidate, instance)")
    out_path = _out_path(exp_path, mode, dim, n_rep, has_incumbent)

    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing: {trajectory_path}")

    with open(trajectory_path) as f:
        traj_data = json.load(f)
    if has_incumbent:
        incs = traj_data.get(incumbent_key, traj_data.get("incumbents"))
        if not isinstance(incs, list):
            raise KeyError(
                f"trajectory.json at {trajectory_path} has no 'incumbents' list "
                f"(expected a list of per-generation incumbent rows); refusing to "
                f"evaluate. Re-run the search so the incumbent series is logged."
            )
        traj = incs
    else:
        traj = traj_data.get("trajectory")
        if not isinstance(traj, list):
            raise KeyError(
                f"trajectory.json at {trajectory_path} has no 'trajectory' list; "
                f"refusing to evaluate. Re-run the search so the full trajectory is logged."
            )
    if mode == "gen":
        traj = [traj[-1]]

    with open(heuristic_path) as f:
        heuristics_data = json.load(f)
    df_heuristics = pd.DataFrame(heuristics_data["heuristics"])

    for t in traj:
        cand_id = t["cand_id"]
        row = df_heuristics[df_heuristics["cand_id"] == cand_id]["source"].values
        t["source"] = row[0] if len(row) > 0 else None
        if len(row) == 0:
            print(f"  Warning: no heuristic found for cand_id={cand_id}", flush=True)

    seen_ids: dict = {}
    for t in traj:
        if t["source"] is not None and t["cand_id"] not in seen_ids:
            seen_ids[t["cand_id"]] = t
    all_unique = list(seen_ids.values())
    n_traj_entries = sum(1 for t in traj if t["source"] is not None)

    cached_score: dict = {}   # cand_id -> mean AOCC
    cached_rt: dict = {}      # cand_id -> mean runtime

    pending = [h for h in all_unique if h["cand_id"] not in cached_score]

    if not pending:
        print("All candidates already scored — nothing to evaluate.", flush=True)
    else:
        tasks = build_tasks(n_rep)
        print(f"Scoring {len(pending)} pending heuristics "
              f"({len(cached_score)} cached, {n_traj_entries} total traj entries, "
              f"{_N_FUNCTIONS} functions x {n_rep} rep = {len(tasks)} evals/candidate, "
              f"dim={dim}, budget={budget}, {n_cores} cores)...", flush=True)
        t_score = time.time()
        scores, new_rts = batch_scoring_parallel(tasks, pending, dim, budget, eval_timeout, n_cores)
        print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)
        for h, aocc, rt in zip(pending, scores, new_rts):
            cached_score[h["cand_id"]] = aocc
            cached_rt[h["cand_id"]] = rt

    # Reconstruct full parallel arrays aligned to traj
    score_out = [
        cached_score.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]
    runtimes_out = [
        cached_rt.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]

    valid = [s for s in score_out if s is not None]
    best = max(valid) if valid else float("-inf")   # AOCC: higher is better
    n_covered = sum(1 for s in score_out if s is not None)
    print(f"  covered {n_covered}/{len(traj)} trajectory entries  "
          f"best mean_AOCC={best:.4f}  total_wall={time.time() - t_start:.1f}s", flush=True)

    return {
        "metric": "mean_AOCC (higher is better; in [0,1], 1.0 = optimum)",
        "dim": dim,
        "budget_factor": budget_factor,
        "n_rep": n_rep,
        "n_functions": _N_FUNCTIONS,
        "eval_timeout": eval_timeout,
        "used_budget": [pt.get("used_budget") for pt in traj],
        "cpu_seconds": [pt.get("cpu_seconds") for pt in traj],
        "score": score_out,
        "runtimes": runtimes_out,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-evaluate trajectory checkpoints over all 72 BBOB functions for EoH & OpenEvolve (mean AOCC)."
    )
    parser.add_argument(
        "--exp", type=str, required=True,
        help="Path to the experiment directory, e.g. "
             "'.logs/racing_eoh_bbob_vllm/2026-08-16/.../'",
    )
    parser.add_argument("--dim", type=int, default=5, help="Problem dimension.")
    parser.add_argument("--budget-factor", type=int, default=2000,
                        help="Per-instance func-eval budget = budget_factor * dim.")
    parser.add_argument("--n-rep", type=int, default=1,
                        help="Repeated seeded runs per (fid, iid) function (default 1). "
                             "Total evals/candidate = 72 * n_rep.")
    parser.add_argument("--eval-timeout", type=float, default=-1.0,
                        help="Per-(candidate, instance) wall-clock cap (s). -1 (default) = "
                             "auto-scale = 60 + budget_factor*dim/100; 0 = disabled; >0 = fixed.")
    parser.add_argument("--n-cores", type=int, default=20)
    parser.add_argument("--mode", type=str, default="valid", choices=["valid", "gen"],
                        help="'valid' for full trajectory, 'gen' for the last candidate only.")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="If set, evaluate the per-generation incumbents instead of the full trajectory.")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    print(f"Processing experiment: {exp_path}")
    result = evaluate_single(exp_path, args.dim, args.budget_factor, args.n_rep,
                             args.eval_timeout, args.n_cores, args.mode, args.has_incumbent)
    out_path = _out_path(exp_path, args.mode, args.dim, args.n_rep, args.has_incumbent)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serialisable)
    print(f"Saved -> {out_path}")
