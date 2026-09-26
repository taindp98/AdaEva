"""Evaluate BBOB heuristics under the HOMO scenario.

Homo scenario settings:
  - 1 single BBOB function: Rastrigin (FID 3).
  - 3 dimensionalities: {5, 10, 20}.
  - 3 instance IDs: (1, 2, 3).
  - Repeated runs: configurable via n_rep (default 1). Total evals per candidate = 3 dims x 3 iids x n_rep = 9 * n_rep.
  - Per-instance budget = budget_factor * dim (e.g. 2000 * dim).
"""

import sys, json, pathlib, time, threading, signal
from collections import defaultdict
import numpy as np
import pandas as pd
import multiprocessing as mp
import math
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD"), str(ROOT / "packages" / "LLaMEA")]

from ioh import get_problem, logger
from misc import OverBudgetException, aoc_logger, correct_aoc

# Homo scenario parameters: single BBOB function (Rastrigin = FID 3) across 3 dimensions
_HOMO_FID = 3
_HOMO_DIMS = (5, 10, 20)
_IIDS = (1, 2, 3)
_AOC_LOWER = 1e-8
_AOC_UPPER = 1e2
_N_FUNCTIONS = len(_HOMO_DIMS) * len(_IIDS)  # 9


def _dim_class(dim: int) -> str:
    return f"dim{dim}"


def _auto_eval_timeout(budget_factor: int, dim: int = 20) -> float:
    """Per-(candidate, instance) wall-clock cap that scales with the func-eval
    budget (= budget_factor * dim). Default dim=20 to ensure generous cap across {5,10,20}."""
    return 60.0 + float(budget_factor) * float(dim) / 100.0


class _EvalTimeout(Exception):
    """Raised when a single BBOB eval exceeds its wall-clock cap (SIGALRM)."""


def _compile_bbob(source: str):
    """Exec the heuristic source and return the optimizer CLASS."""
    ns = {"np": np, "numpy": np}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(source, ns)
    classes = [v for k, v in ns.items()
               if isinstance(v, type) and not k.startswith("_") and "__call__" in v.__dict__]
    if not classes:
        classes = [v for k, v in ns.items() if isinstance(v, type) and not k.startswith("_")]
    if not classes:
        raise ValueError("no optimizer class defined in source")
    return classes[0]


_ORIG_NP_SEED = getattr(np.random, "seed", None)


def _restore_np_seed():
    if not callable(getattr(np.random, "seed", None)):
        if callable(_ORIG_NP_SEED):
            np.random.seed = _ORIG_NP_SEED
        else:
            import importlib, numpy.random
            importlib.reload(numpy.random)


def score_bbob_inst(algo_cls, fid: int, iid: int, dim: int, budget: int, rep: int,
                    eval_timeout: float = None, cand_id: str = "") -> tuple:
    """Run one heuristic on one (fid, iid, dim, rep) BBOB problem; return (AOCC, runtime)."""
    t0 = time.perf_counter()
    l2 = aoc_logger(budget, lower=_AOC_LOWER, upper=_AOC_UPPER, triggers=[logger.trigger.ALWAYS])
    problem = get_problem(fid, iid, dim)
    problem.attach_logger(l2)
    _restore_np_seed()
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
            algorithm = algo_cls(budget=budget, dim=dim)
            algorithm(problem)
    except OverBudgetException:
        pass
    except _EvalTimeout:
        print(f"  [WARNING] eval TIMEOUT: heuristic '{cand_id}' on homo instance "
              f"(fid={fid}, iid={iid}, dim={dim}, rep={rep}) exceeded {float(eval_timeout):.0f}s "
              f"-> partial AOCC", flush=True)
    except Exception:
        return 0.0, time.perf_counter() - t0
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        _restore_np_seed()
    try:
        aocc = float(correct_aoc(problem, l2, budget))
    except Exception:
        aocc = 0.0
    if not math.isfinite(aocc):
        aocc = 0.0
    return aocc, time.perf_counter() - t0


def build_tasks(budget_factor: int = 2000, n_rep: int = 1) -> list:
    """The 9 * n_rep HOMO BBOB (fid=3, iid, dim, budget, rep) tasks across dims {5, 10, 20}."""
    tasks = []
    for d in _HOMO_DIMS:
        bgt = budget_factor * d
        for iid in _IIDS:
            for rep in range(n_rep):
                tasks.append((_HOMO_FID, iid, d, bgt, rep))
    return tasks


def _worker(args):
    idx, cand_id, source, tasks, eval_timeout = args
    pid = mp.current_process().pid
    t0 = time.perf_counter()
    try:
        algo_cls = _compile_bbob(source)
    except Exception as e:
        print(f"  [worker pid={pid}] {cand_id}: compile error: {e}", flush=True)
        return idx, 0.0, 0.0, {}
    aoccs, rts = [], []
    dim_aoccs = defaultdict(list)
    for k, (fid, iid, d, bgt, rep) in enumerate(tasks):
        a, rt = score_bbob_inst(algo_cls, fid, iid, d, bgt, rep, eval_timeout, cand_id)
        aoccs.append(a); rts.append(rt)
        dim_aoccs[_dim_class(d)].append(a)
        if (k + 1) % 6 == 0 or (k + 1) == len(tasks):
            print(f"    [worker pid={pid}] {cand_id}  task {k+1}/{len(tasks)}  "
                  f"last_AOCC={a:.4f} (f{fid}i{iid}d{d}r{rep})  "
                  f"running_mean_AOCC={float(np.mean(aoccs)):.4f}  "
                  f"worker_wall={time.perf_counter()-t0:.0f}s", flush=True)
    score = float(np.mean(aoccs)); runtime = float(np.mean(rts))
    per_class = {f"dim{d}": float(np.mean(dim_aoccs[f"dim{d}"]))
                 for d in _HOMO_DIMS if dim_aoccs[f"dim{d}"]}
    print(f"  [worker pid={pid}] {cand_id}: DONE  mean_AOCC={score:.4f}  "
          f"wall={time.perf_counter()-t0:.1f}s", flush=True)
    return idx, score, runtime, per_class


def batch_scoring_parallel(tasks: list, heuristics: list, eval_timeout: float, n_cores: int = 20) -> tuple:
    work = [(i, h["cand_id"], h["source"], tasks, eval_timeout)
            for i, h in enumerate(heuristics)]
    scores = [0.0] * len(heuristics)
    runtimes = [0.0] * len(heuristics)
    perclass: list = [None] * len(heuristics)

    t0 = time.time()
    n_total = len(heuristics)
    n_done = 0

    def _heartbeat():
        while not _stop_heartbeat.is_set():
            time.sleep(30)
            if not _stop_heartbeat.is_set():
                elapsed = time.time() - t0
                print(f"  [eval_homo_bbob] heartbeat: {n_done}/{n_total} heuristics completed "
                      f"({elapsed:.0f}s elapsed)", flush=True)

    _stop_heartbeat = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=min(n_cores, max(1, len(heuristics)))) as pool:
            for idx, score, runtime, per_class in pool.imap_unordered(_worker, work):
                scores[idx] = score
                runtimes[idx] = runtime
                perclass[idx] = per_class
                n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(f"  [{n_done}/{n_total}] done  elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    finally:
        _stop_heartbeat.set()

    return scores, runtimes, perclass


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def _out_path(exp_path, mode, n_rep, has_incumbent):
    inc = "_incumbent" if has_incumbent else ""
    return exp_path / f"{mode}_trajectory{inc}_homo_bbob_nrep{n_rep}.json"


def evaluate_single(exp_path: pathlib.Path, dim: int = 5, budget_factor: int = 2000,
                    n_rep: int = 1, eval_timeout: float = -1.0, n_cores: int = 20,
                    mode: str = "valid", has_incumbent: bool = False, incumbent_key: str = "incumbents_mean_rank") -> dict:
    """Evaluate the trajectory of BBOB heuristics under the HOMO scenario (Rastrigin across dims {5,10,20})."""
    t_start = time.time()
    if eval_timeout is not None and eval_timeout < 0:
        eval_timeout = _auto_eval_timeout(budget_factor, 20)
        print(f"  [eval-timeout] auto = 60 + budget_factor({budget_factor}) x dim(20)/100 "
              f"= {eval_timeout:.0f}s per (candidate, instance)")
    out_path = _out_path(exp_path, mode, n_rep, has_incumbent)

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
                f"evaluate."
            )
        traj = incs
    else:
        traj = traj_data.get("trajectory")
        if not isinstance(traj, list):
            raise KeyError(
                f"trajectory.json at {trajectory_path} has no 'trajectory' list; "
                f"refusing to evaluate."
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

    cached_score: dict = {}
    cached_rt: dict = {}
    cached_perclass: dict = {}

    pending = [h for h in all_unique if h["cand_id"] not in cached_score]

    if not pending:
        print("All candidates already scored — nothing to evaluate.", flush=True)
    else:
        tasks = build_tasks(budget_factor, n_rep)
        print(f"Scoring {len(pending)} pending heuristics in HOMO mode "
              f"({len(cached_score)} cached, {n_traj_entries} total traj entries, "
              f"{_N_FUNCTIONS} instances x {n_rep} rep = {len(tasks)} evals/candidate, "
              f"dims={_HOMO_DIMS}, {n_cores} cores)...", flush=True)
        t_score = time.time()
        scores, new_rts, new_pc = batch_scoring_parallel(tasks, pending, eval_timeout, n_cores)
        print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)
        for h, aocc, rt, pc in zip(pending, scores, new_rts, new_pc):
            cached_score[h["cand_id"]] = aocc
            cached_rt[h["cand_id"]] = rt
            cached_perclass[h["cand_id"]] = pc

    score_out = [
        cached_score.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]
    runtimes_out = [
        cached_rt.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]
    perclass_out = [
        cached_perclass.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]

    valid = [s for s in score_out if s is not None]
    best = max(valid) if valid else float("-inf")
    n_covered = sum(1 for s in score_out if s is not None)
    print(f"  covered {n_covered}/{len(traj)} trajectory entries  "
          f"best mean_AOCC={best:.4f}  total_wall={time.time() - t_start:.1f}s", flush=True)

    return {
        "metric": "mean_AOCC (higher is better; in [0,1], 1.0 = optimum)",
        "dim": list(_HOMO_DIMS),
        "budget_factor": budget_factor,
        "n_rep": n_rep,
        "n_functions": _N_FUNCTIONS,
        "eval_timeout": eval_timeout,
        "used_budget": [pt.get("used_budget") for pt in traj],
        "cpu_seconds": [pt.get("cpu_seconds") for pt in traj],
        "score": score_out,
        "runtimes": runtimes_out,
        "per_class": perclass_out,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-evaluate trajectory checkpoints under HOMO BBOB scenario (Rastrigin across dims {5,10,20})."
    )
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--budget-factor", type=int, default=2000)
    parser.add_argument("--n-rep", type=int, default=1)
    parser.add_argument("--eval-timeout", type=float, default=-1.0)
    parser.add_argument("--n-cores", type=int, default=20)
    parser.add_argument("--mode", type=str, default="valid", choices=["valid", "gen"])
    parser.add_argument("--has-incumbent", action="store_true")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    print(f"Processing HOMO experiment: {exp_path}")
    result = evaluate_single(exp_path, 5, args.budget_factor, args.n_rep,
                             args.eval_timeout, args.n_cores, args.mode, args.has_incumbent)
    out_path = _out_path(exp_path, args.mode, args.n_rep, args.has_incumbent)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serialisable)
    print(f"Saved -> {out_path}")
