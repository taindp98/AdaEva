"""Re-evaluate BBOB trajectory checkpoints on all 72 BBOB functions.

The BBOB counterpart of ``eval_tsp.py`` / ``eval_fssp.py``, for LLaMEA-on-BBOB
runs (``reprod/llamea_bbob.py``, ``racing/llamea_bbob.py``). Same resumable,
parallel structure, with the BBOB specifics:
  - The heuristic is a CLASS ``algo_cls(budget, dim)`` with ``__call__(self, func)``
    (not a plain function). Scored with ``ioh`` + the AOCC logger, exactly like
    ``reprod/llamea_bbob._eval_single_task``.
  - The instance pool is the fixed 72 = 24 fids x 3 iids problems (all noiseless
    BBOB functions) at a single ``--dim``, each run ``--n-rep`` times (default 1).
  - The metric is mean AOCC (Area Over the Convergence Curve) in [0, 1], HIGHER is
    better (1.0 = optimum found instantly) — so ``best`` is a MAX (unlike the TSP
    gap / FSSP makespan, which are minimised). There is no separate gap.
  - BBOB heuristics have no internal wall-clock cap (the func-eval budget only
    bounds ``func`` CALLS, so a loop that never calls ``func`` runs forever). A
    per-(candidate, instance) ``--eval-timeout`` (SIGALRM) guards against that; a
    timed-out run is scored on its PARTIAL AOCC (penalised, not rejected) with a
    WARNING naming the heuristic + instance.
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

# Fixed BBOB evaluation parameters (match reprod/llamea_bbob.py)
_FIDS = range(1, 25)
_IIDS = (1, 2, 3)
_AOC_LOWER = 1e-8
_AOC_UPPER = 1e2
_N_FUNCTIONS = len(_FIDS) * len(_IIDS)  # 72

# The 5 standard BBOB/COCO function classes, for the per-function-class breakdown
# (Dimension 5 radar). fid -> class name.
_BBOB_CLASSES = (
    ("separable", range(1, 6)),        # f1-f5
    ("low_mod_cond", range(6, 10)),    # f6-f9
    ("high_cond", range(10, 15)),      # f10-f14
    ("mm_struct", range(15, 20)),      # f15-f19 (multi-modal, adequate global structure)
    ("mm_weak", range(20, 25)),        # f20-f24 (multi-modal, weak global structure)
)


def _fid_class(fid: int) -> str:
    for name, rng in _BBOB_CLASSES:
        if int(fid) in rng:
            return name
    return "other"


def _auto_eval_timeout(budget_factor: int, dim: int) -> float:
    """Per-(candidate, instance) wall-clock cap that scales with the func-eval
    budget (= budget_factor * dim). Mirrors reprod/llamea_bbob._auto_eval_timeout:
    dim 5 -> 160s, dim 20 -> 460s."""
    return 60.0 + float(budget_factor) * float(dim) / 100.0


class _EvalTimeout(Exception):
    """Raised when a single BBOB eval exceeds its wall-clock cap (SIGALRM)."""


def _compile_bbob(source: str):
    """Exec the heuristic source and return the optimizer CLASS (the one defining
    its own ``__call__``). Falls back to the first top-level class."""
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
    """Run one heuristic on one (fid, iid, rep) BBOB problem; return (AOCC, runtime).

    AOCC in [0, 1], higher = better. ``np.random.seed(rep)`` makes the (stochastic)
    optimizer reproducible (matches training). A crash -> AOCC 0.0 (worst valid). A
    timeout -> PARTIAL AOCC (``correct_aoc`` extrapolates from best-so-far) + WARNING.
    """
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
        _restore_np_seed()
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
    """Score one heuristic across all tasks. Returns (mean AOCC, mean runtime)."""
    try:
        algo_cls = _compile_bbob(source)
    except Exception as e:
        print(f"  [compile] error: {e}", flush=True)
        return 0.0, 0.0
    aoccs, rts = [], []
    for fid, iid, rep in tasks:
        a, rt = score_bbob_inst(algo_cls, fid, iid, dim, budget, rep, eval_timeout)
        aoccs.append(a); rts.append(rt)
    return float(np.mean(aoccs)), float(np.mean(rts))


def _worker(args):
    idx, cand_id, source, tasks, dim, budget, eval_timeout = args
    pid = mp.current_process().pid
    t0 = time.perf_counter()
    try:
        algo_cls = _compile_bbob(source)
    except Exception as e:
        print(f"  [worker pid={pid}] {cand_id}: compile error: {e}", flush=True)
        return idx, 0.0, 0.0, {}, [0.0] * len(tasks)
    aoccs, rts = [], []
    cls_aoccs = defaultdict(list)                 # per-function-class AOCCs
    for k, (fid, iid, rep) in enumerate(tasks):
        a, rt = score_bbob_inst(algo_cls, fid, iid, dim, budget, rep, eval_timeout, cand_id)
        aoccs.append(a); rts.append(rt)
        cls_aoccs[_fid_class(fid)].append(a)
        if (k + 1) % 12 == 0 or (k + 1) == len(tasks):
            print(f"    [worker pid={pid}] {cand_id}  task {k+1}/{len(tasks)}  "
                  f"last_AOCC={a:.4f} (f{fid}i{iid}r{rep})  "
                  f"running_mean_AOCC={float(np.mean(aoccs)):.4f}  "
                  f"worker_wall={time.perf_counter()-t0:.0f}s", flush=True)
    score = float(np.mean(aoccs)); runtime = float(np.mean(rts))
    per_class = {name: float(np.mean(cls_aoccs[name]))
                 for name, _ in _BBOB_CLASSES if cls_aoccs[name]}
    print(f"  [worker pid={pid}] {cand_id}: DONE  mean_AOCC={score:.4f}  "
          f"wall={time.perf_counter()-t0:.1f}s", flush=True)
    return idx, score, runtime, per_class, aoccs


def batch_scoring_parallel(tasks: list, heuristics: list, dim: int, budget: int,
                           eval_timeout: float, n_cores: int = 20,
                           return_per_instance: bool = False) -> tuple:
    work = [(i, h["cand_id"], h["source"], tasks, dim, budget, eval_timeout)
            for i, h in enumerate(heuristics)]
    scores = [0.0] * len(heuristics)
    runtimes = [0.0] * len(heuristics)
    perclass: list = [None] * len(heuristics)
    per_instance: list = [None] * len(heuristics)
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
            for idx, score, runtime, per_class, aoccs in pool.imap_unordered(_worker, work, chunksize=1):
                scores[idx] = score
                runtimes[idx] = runtime
                perclass[idx] = per_class
                per_instance[idx] = aoccs
                n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(f"  [{n_done}/{n_total}] done  elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    finally:
        _stop_heartbeat.set()

    if return_per_instance:
        return scores, runtimes, perclass, per_instance
    return scores, runtimes, perclass


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def _out_path(exp_path, mode, dim, n_rep, has_incumbent):
    inc = "_incumbent" if has_incumbent else ""
    return exp_path / f"{mode}_trajectory{inc}_bbob_d{dim}_nrep{n_rep}.json"


def evaluate_single(exp_path: pathlib.Path, dim: int = 5, budget_factor: int = 2000,
                    n_rep: int = 1, eval_timeout: float = -1.0, n_cores: int = 20,
                    mode: str = "valid", has_incumbent: bool = False, incumbent_key: str = "incumbents_mean_rank") -> dict:
    """Evaluate the trajectory of BBOB heuristics over all 72 BBOB functions.

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

    # --- Per-candidate score caches (filled during evaluation) ---
    cached_score: dict = {}   # cand_id -> mean AOCC
    cached_rt: dict = {}      # cand_id -> mean runtime
    cached_perclass: dict = {}  # cand_id -> {class name -> mean AOCC}

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
        scores, new_rts, new_pc = batch_scoring_parallel(tasks, pending, dim, budget, eval_timeout, n_cores)
        print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)
        for h, aocc, rt, pc in zip(pending, scores, new_rts, new_pc):
            cached_score[h["cand_id"]] = aocc
            cached_rt[h["cand_id"]] = rt
            cached_perclass[h["cand_id"]] = pc

    # Reconstruct full parallel arrays aligned to traj
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
        "cand_id": [pt["cand_id"] for pt in traj],
        "used_budget": [pt.get("used_budget") for pt in traj],
        "cpu_seconds": [pt.get("cpu_seconds") for pt in traj],
        "score": score_out,
        "runtimes": runtimes_out,
        "per_class": perclass_out,   # per-incumbent {class -> mean AOCC}; aligned to `score`
    }


def evaluate_heuristics_file(
    json_path: pathlib.Path,
    dim: int = 5,
    budget_factor: int = 2000,
    n_rep: int = 1,
    eval_timeout: float = -1.0,
    n_cores: int = 20,
    out_root: pathlib.Path | None = None,
) -> dict:
    """Score every heuristic stored in a heuristics JSON (e.g. src/init_pop/llamea_24_bbob.json)
    against the FULL BBOB suite and dump the detailed per-(heuristic, instance) performance
    matrix for post-hoc analysis (Kendall's W across instances, cost/score dispersion).

    Input JSON schema (same as init_pop / heuristics.json): ``{"heuristics": [{"cand_id",
    "source", "score" (optional, the stated/search-time score), ...}, ...]}``. Each heuristic
    is a CLASS ``algo_cls(budget, dim)`` with ``__call__(self, func)`` (LLaMEA-on-BBOB).

    The "instance" axis is the 72 = 24 fids x 3 iids BBOB problems at a single ``dim`` (x
    ``n_rep`` seeded repetitions if n_rep>1). Raw per-instance score = AOCC in [0, 1], HIGHER
    is better (opposite direction to OBP/TSP/FSSP cost) — so the rank matrix is DESCENDING
    (rank 1 = highest AOCC = best). A per-task crash/timeout yields a genuine 0.0 AOCC (a
    valid worst score, kept as-is); only a heuristic that fails to COMPILE is flagged
    ``status=crash`` (its whole row is NaN, excluded from W / dispersion). Nothing else is
    penalty-filled.

    Artefacts saved under ``out_root/<json-stem>/<timestamp>/`` (out_root defaults to
    ``.logs/posthoc_analyses``):
      * ``perf_matrix.csv``  -> rows = heuristics (indexed by cand_id), cols = f{fid}_i{iid}
                                (+ _r{rep} if n_rep>1), values = raw AOCC (empty = compile crash).
      * ``rank_matrix.csv``  -> same shape, per-instance DESCENDING rank of AOCC (1 = best);
                                compile-crashed rows rank last (NaN-aware, method='average').
      * ``perf_matrix.json`` -> the matrix as records + metadata (suite config, per-heuristic
                                status + stated ``score``). AOCC is higher-better; no gap.
    """
    t_start = time.time()
    json_path = pathlib.Path(json_path)
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    if not json_path.exists():
        raise FileNotFoundError(f"Missing heuristics JSON: {json_path}")

    budget = budget_factor * dim
    # Resolve --eval-timeout: -1 => auto-scale with budget; 0 => disabled; >0 => fixed.
    if eval_timeout is not None and eval_timeout < 0:
        eval_timeout = _auto_eval_timeout(budget_factor, dim)
        print(f"  [eval-timeout] auto = 60 + budget_factor({budget_factor}) x dim({dim})/100 "
              f"= {eval_timeout:.0f}s per (candidate, instance)")

    with open(json_path) as f:
        data = json.load(f)
    heuristics = data.get("heuristics")
    if not isinstance(heuristics, list) or not heuristics:
        raise KeyError(f"{json_path} has no non-empty 'heuristics' list.")
    heuristics = [h for h in heuristics if h.get("source")]
    if not heuristics:
        raise ValueError(f"{json_path}: no heuristics with a non-empty 'source'.")
    cand_ids = [h["cand_id"] for h in heuristics]
    stated_score = {h["cand_id"]: h.get("score") for h in heuristics}

    # Flag COMPILE failures in the parent: _worker returns [0.0]*n on a compile crash, which is
    # otherwise indistinguishable from a heuristic that genuinely scored 0.0 AOCC everywhere.
    compile_ok = {}
    for h in heuristics:
        try:
            _compile_bbob(h["source"])
            compile_ok[h["cand_id"]] = True
        except Exception as e:
            print(f"  [compile] {h['cand_id']}: {e}", flush=True)
            compile_ok[h["cand_id"]] = False

    tasks = build_tasks(n_rep)
    print(f"Scoring {len(heuristics)} heuristics x {len(tasks)} instances "
          f"({_N_FUNCTIONS} functions x {n_rep} rep, dim={dim}, budget={budget}) "
          f"on {n_cores} cores...", flush=True)

    t_score = time.time()
    scores, runtimes, _perclass, per_inst = batch_scoring_parallel(
        tasks, heuristics, dim, budget, eval_timeout, n_cores, return_per_instance=True)
    print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)

    # --- Build the (heuristic x instance) raw-AOCC matrix. Compile-crash row -> all NaN. ---
    if n_rep > 1:
        inst_cols = [f"f{fid}_i{iid}_r{rep}" for (fid, iid, rep) in tasks]
    else:
        inst_cols = [f"f{fid}_i{iid}" for (fid, iid, rep) in tasks]

    def _row(cid, pinst):
        if not compile_ok.get(cid, True):
            return [np.nan] * len(inst_cols)     # compile crash -> excluded from W / dispersion
        vals = [float(a) for a in (pinst or [])]
        vals = [(v if np.isfinite(v) else np.nan) for v in vals]  # keep 0.0; drop non-finite
        vals += [np.nan] * (len(inst_cols) - len(vals))
        return vals[:len(inst_cols)]

    mat = pd.DataFrame([_row(cid, p) for cid, p in zip(cand_ids, per_inst)],
                       index=cand_ids, columns=inst_cols)
    mat.index.name = "cand_id"
    # AOCC is higher-better -> rank DESCENDING so rank 1 = best; crashed rows rank last.
    rank_mat = mat.rank(axis=0, method="average", ascending=False, na_option="bottom")

    status = {cid: ("ok" if compile_ok.get(cid, True) else "crash") for cid in cand_ids}

    if out_root is None:
        out_root = ROOT / ".logs" / "posthoc_analyses"
    out_dir = pathlib.Path(out_root) / json_path.stem / time.strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    mat.to_csv(out_dir / "perf_matrix.csv")
    rank_mat.to_csv(out_dir / "rank_matrix.csv")

    meta = {
        "source_json": str(json_path),
        "metric": "mean_AOCC (higher is better; in [0,1], 1.0 = optimum; no gap)",
        "n_heuristics": len(heuristics),
        "n_instances": len(tasks),
        "instance_config": {"dim": dim, "budget_factor": budget_factor, "budget": budget,
                            "n_rep": n_rep, "n_functions": _N_FUNCTIONS,
                            "eval_timeout": eval_timeout},
        "heuristics": [
            {
                "cand_id": cid,
                "stated_score": stated_score.get(cid),   # search-time score from the JSON
                "full_mean_aocc": (float(s) if compile_ok.get(cid, True) else None),
                "mean_runtime": (float(r) if np.isfinite(r) else None),
                "status": status[cid],
            }
            for cid, s, r in zip(cand_ids, scores, runtimes)
        ],
        "perf_matrix": [
            {"cand_id": cid, **{c: (None if pd.isna(v) else float(v))
                                for c, v in zip(inst_cols, mat.loc[cid].values)}}
            for cid in cand_ids
        ],
    }
    with open(out_dir / "perf_matrix.json", "w") as f:
        json.dump(meta, f, indent=2, default=_to_serialisable)

    n_ok = sum(1 for v in status.values() if v == "ok")
    print(f"  {n_ok}/{len(heuristics)} heuristics compiled ok "
          f"({len(heuristics) - n_ok} compile-crash)  "
          f"total_wall={time.time() - t_start:.1f}s", flush=True)
    print(f"Saved -> {out_dir}", flush=True)
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-evaluate trajectory checkpoints over all 72 BBOB functions (mean AOCC)."
    )
    parser.add_argument(
        "--exp", type=str, required=False, default=None,
        help="Path to the experiment directory, e.g. "
             "'.logs/racing_llamea_bbob_vllm/2026-08-08/.../'",
    )
    parser.add_argument(
        "--heuristics-json", type=str, default=None,
        help="Path to a heuristics JSON (e.g. 'src/init_pop/llamea_24_bbob.json'). When given, "
             "score every heuristic in the file against the full BBOB suite and dump the "
             "per-(heuristic, instance) performance matrix to .logs/posthoc_analyses "
             "(for Kendall's W / dispersion analysis). Mutually exclusive with --exp.",
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

    if args.heuristics_json:
        evaluate_heuristics_file(
            args.heuristics_json, args.dim, args.budget_factor, args.n_rep,
            eval_timeout=args.eval_timeout, n_cores=args.n_cores,
        )
        sys.exit(0)

    if not args.exp:
        parser.error("one of --exp or --heuristics-json is required")

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    print(f"Processing experiment: {exp_path}")
    result = evaluate_single(exp_path, args.dim, args.budget_factor, args.n_rep,
                             args.eval_timeout, args.n_cores, args.mode, args.has_incumbent)
    out_path = _out_path(exp_path, args.mode, args.dim, args.n_rep, args.has_incumbent)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serialisable)
    print(f"Saved -> {out_path}")
