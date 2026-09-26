"""Re-evaluate FSSP-GLS trajectory checkpoints on a validation instance pool.

The FSSP counterpart of ``eval_tsp.py``. Same resumable, parallel structure, with
the FSSP-specific differences:
  - The heuristic entry point is ``get_matrix_and_jobs`` (not ``update_edge_distance``).
  - Scoring uses ``fssp_gls.solve_without_time`` -> mean MAKESPAN (lower is better).
  - FSSP has NO known optimum (no Concorde), so there is no gap%: the reported
    metric is the raw mean makespan (``score``), matching how the search logs it.
  - The FSSP evaluation module defaults to ``time_max=10s``; we patch it to 60s
    (+ ``iter_max=1000``) here, exactly as ``reprod/eoh_fssp_gls.py`` does, so the
    validation GLS cap matches the training cap. Patched at import so the fork
    pool's workers inherit it.

Instance pool (``--instance-source``):
  - ``synthetic`` (default): a fresh, deterministically-seeded pool of ``n_jobs``-job
    instances with machines in ``[m_low, m_high]`` — the held-out generalization set
    (analogous to eval_tsp generating fresh TSP instances).
  - ``pregenerated``: the fixed EoH/Taillard benchmark files (the training set).
"""

import sys, json, pathlib, time, threading
import numpy as np
import pandas as pd
import multiprocessing as mp
import math

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD")]

import llm4ad.task.optimization.fssp_gls.evaluation as _fssp_eval_mod
from llm4ad.task.optimization.fssp_gls.evaluation import solve_without_time
from llm4ad.task.optimization.fssp_gls.get_instance import GetData

# Match the training-time GLS cap (the module ships with time_max=10s). Patched
# BEFORE the fork pool is created so worker processes inherit these globals, which
# ``solve_without_time`` reads at call time.
_fssp_eval_mod.time_max = 60.0
_fssp_eval_mod.iter_max = 1000

# Finite penalty for a crash / non-finite makespan (FSSP makespans are ~1e3, so
# this clearly ranks last while keeping the mean numeric).
_FAIL_COST = 1e9


def score_fssp_inst(fn: callable, inst, seed=None) -> tuple:
    """Run ``fn`` as ``get_matrix_and_jobs`` on one FSSP instance; return
    ``(makespan, running_time)``. ``seed`` seeds the (stochastic) GLS search so the
    validation score is reproducible. A crash / non-finite makespan -> ``_FAIL_COST``."""
    t0 = time.perf_counter()
    try:
        cost = solve_without_time(inst, fn, seed=seed)
        cost = float(cost) if math.isfinite(float(cost)) else _FAIL_COST
    except Exception as e:
        print(f"  [score_fssp_inst] error: {e}", flush=True)
        cost = _FAIL_COST
    running_time = time.perf_counter() - t0
    return cost, running_time


def load_instances(n_instances: int = 64, n_jobs: int = 50, m_low: int = 2,
                   m_high: int = 20, instance_source: str = "synthetic") -> list:
    """Build the FSSP validation instance pool. No optima (FSSP has none)."""
    use_pregenerated = (instance_source == "pregenerated")
    np.random.seed(2024)   # synthetic GetData reseeds internally too -> deterministic pool
    instances = GetData(n_instances, n_jobs=n_jobs, m_low=m_low, m_high=m_high,
                        use_pregenerated=use_pregenerated).generate_instances()
    jobs = [getattr(i, "tasks_val", None) for i in instances]
    machines = [getattr(i, "machines_val", None) for i in instances]
    print(f"Loaded {len(instances)} FSSP instances "
          f"(source={instance_source}, jobs={min(j for j in jobs if j)}..{max(j for j in jobs if j)}, "
          f"machines={min(m for m in machines if m)}..{max(m for m in machines if m)})", flush=True)
    return instances


def _compile_get_matrix_and_jobs(source: str) -> callable:
    ns = {"np": np}
    exec(source, ns)
    fn = ns.get("get_matrix_and_jobs")
    if fn is None:
        # Fall back to the first top-level callable if the entry point was renamed.
        cands = [v for k, v in ns.items() if callable(v) and not k.startswith("_") and k != "np"]
        if not cands:
            raise ValueError("no 'get_matrix_and_jobs' (or any callable) defined in source")
        fn = cands[0]
    return fn


def single_batch_scoring(instances: list, source: str) -> tuple:
    """Score one heuristic across all instances. Returns (mean makespan, mean runtime)."""
    try:
        fn = _compile_get_matrix_and_jobs(source)
    except Exception as e:
        print(f"  [compile] error: {e}", flush=True)
        return _FAIL_COST, _FAIL_COST
    objs = []
    for i, inst in enumerate(instances):
        objs.append(score_fssp_inst(fn, inst, seed=2024 + i))
    objs = np.array(objs)
    return float(np.mean(objs[:, 0])), float(np.mean(objs[:, 1]))


def _worker(args):
    idx, cand_id, source, instances = args
    pid = mp.current_process().pid
    t0 = time.perf_counter()
    try:
        fn = _compile_get_matrix_and_jobs(source)
    except Exception as e:
        print(f"  [worker pid={pid}] {cand_id}: compile error: {e}", flush=True)
        return idx, _FAIL_COST, _FAIL_COST, [_FAIL_COST] * len(instances)
    objs = []
    for i, inst in enumerate(instances):
        obj = score_fssp_inst(fn, inst, seed=2024 + i)
        objs.append(obj)
        if (i + 1) % 8 == 0 or (i + 1) == len(instances):
            costs_so_far = np.array(objs)[:, 0]
            print(f"    [worker pid={pid}] {cand_id}  inst {i+1}/{len(instances)}  "
                  f"last_makespan={obj[0]:.4f}  last_gls={obj[1]:.1f}s  "
                  f"running_mean={float(np.mean(costs_so_far)):.4f}  "
                  f"worker_wall={time.perf_counter()-t0:.0f}s", flush=True)
    objs = np.array(objs)
    score = float(np.mean(objs[:, 0])); runtime = float(np.mean(objs[:, 1]))
    dt = time.perf_counter() - t0
    print(f"  [worker pid={pid}] {cand_id}: DONE  "
          f"mean_makespan={score:.4f}  mean_gls_time={runtime:.1f}s  wall={dt:.1f}s", flush=True)
    return idx, score, runtime, objs[:, 0].tolist()


def batch_scoring_parallel(instances: list, heuristics: list, n_cores: int = 20, return_per_instance: bool = False) -> tuple:
    tasks = [(i, h["cand_id"], h["source"], instances) for i, h in enumerate(heuristics)]
    scores = [np.inf] * len(heuristics)
    runtimes = [np.inf] * len(heuristics)
    per_inst = [None] * len(heuristics)
    n_total = len(heuristics)
    n_done = 0
    t0 = time.time()

    _stop_heartbeat = threading.Event()
    def _heartbeat():
        while not _stop_heartbeat.wait(120):
            elapsed = time.time() - t0
            print(f"  [heartbeat] {n_done}/{n_total} done  {n_total - n_done} in-flight  "
                  f"elapsed={elapsed:.0f}s", flush=True)
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=n_cores) as pool:
            for idx, score, runtime, inst_costs in pool.imap_unordered(_worker, tasks, chunksize=1):
                scores[idx] = score
                runtimes[idx] = runtime
                per_inst[idx] = inst_costs
                n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(f"  [{n_done}/{n_total}] done  elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    finally:
        _stop_heartbeat.set()

    if return_per_instance:
        return scores, runtimes, per_inst
    return scores, runtimes


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def _out_path(exp_path, mode, n_instances, n_jobs, has_incumbent):
    inc = "_incumbent" if has_incumbent else ""
    return exp_path / f"{mode}_trajectory{inc}_ni{n_instances}_nj{n_jobs}.json"


def evaluate_single(exp_path: pathlib.Path, n_instances: int = 64, n_jobs: int = 50,
                    m_low: int = 2, m_high: int = 20, instance_source: str = "synthetic",
                    n_cores: int = 20, mode: str = "valid", has_incumbent: bool = False, incumbent_key: str = "incumbents_mean_rank") -> dict:
    """Evaluate the trajectory of FSSP-GLS heuristics on a validation instance pool.

    Resumable: on interrupt only unique unscored candidates are re-evaluated, then
    the output file is updated. The metric is the raw mean makespan (lower better);
    FSSP has no optimum so there is no gap.
    """
    t_start = time.time()
    out_path = _out_path(exp_path, mode, n_instances, n_jobs, has_incumbent)

    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing: {trajectory_path}")

    with open(trajectory_path) as f:
        traj_data = json.load(f)
    # Evaluate the per-generation incumbents (a single list) or the full trajectory.
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

    # Unique candidates in the trajectory
    seen_ids: dict = {}
    for t in traj:
        if t["source"] is not None and t["cand_id"] not in seen_ids:
            seen_ids[t["cand_id"]] = t
    all_unique = list(seen_ids.values())
    n_traj_entries = sum(1 for t in traj if t["source"] is not None)

    # --- Resume: load already-scored candidates from existing output ---
    cached_score: dict = {}   # cand_id -> mean makespan
    cached_rt: dict = {}      # cand_id -> mean runtime
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        prev_scores = prev.get("score", [])
        prev_rts = prev.get("runtimes", [None] * len(prev_scores))
        for i, pt in enumerate(traj):
            if i >= len(prev_scores):
                break
            cid = pt["cand_id"]
            s = prev_scores[i]
            r = prev_rts[i] if i < len(prev_rts) else None
            if s is not None and cid not in cached_score:
                cached_score[cid] = s
                cached_rt[cid] = r
        print(f"Resume: {len(cached_score)}/{len(all_unique)} unique candidates "
              f"already scored in {out_path.name}", flush=True)
    else:
        print("No existing output found — starting from scratch.", flush=True)

    pending = [h for h in all_unique if h["cand_id"] not in cached_score]

    if not pending:
        print("All candidates already scored — nothing to evaluate.", flush=True)
    else:
        instances = load_instances(n_instances, n_jobs, m_low, m_high, instance_source)
        print(f"Scoring {len(pending)} pending heuristics "
              f"({len(cached_score)} cached, {n_traj_entries} total traj entries, "
              f"{len(instances)} instances, {n_cores} cores)...", flush=True)
        t_score = time.time()
        scores, new_rts = batch_scoring_parallel(instances, pending, n_cores=n_cores)
        print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)
        for h, mk, rt in zip(pending, scores, new_rts):
            cached_score[h["cand_id"]] = mk
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
    best = min(valid) if valid else float("inf")
    n_covered = sum(1 for s in score_out if s is not None)
    print(f"  covered {n_covered}/{len(traj)} trajectory entries  "
          f"best mean_makespan={best:.4f}  total_wall={time.time() - t_start:.1f}s", flush=True)

    return {
        "metric": "mean_makespan (lower is better; FSSP has no optimum -> no gap)",
        "instance_source": instance_source,
        "n_instances": n_instances,
        "n_jobs": n_jobs,
        "cand_id": [pt["cand_id"] for pt in traj],
        "used_budget": [pt.get("used_budget") for pt in traj],
        "cpu_seconds": [pt.get("cpu_seconds") for pt in traj],
        "score": score_out,
        "runtimes": runtimes_out,
    }


def evaluate_heuristics_file(
    json_path: pathlib.Path,
    n_instances: int = 64,
    n_jobs: int = 50,
    m_low: int = 2,
    m_high: int = 20,
    instance_source: str = "pregenerated",
    n_cores: int = 20,
    out_root: pathlib.Path | None = None,
) -> dict:
    """Score every heuristic stored in a heuristics JSON (e.g. src/init_pop/eoh_fssp_gls.json)
    against the FULL FSSP-GLS validation suite and dump the detailed per-(heuristic, instance)
    performance matrix for post-hoc analysis (Kendall's W across instances, cost/score
    dispersion).

    Input JSON schema (same as init_pop / heuristics.json): ``{"heuristics": [{"cand_id",
    "source", "score" (optional, the stated/search-time score), ...}, ...]}``. The heuristic
    entry point is ``get_matrix_and_jobs``.

    Raw per-instance cost = mean GLS makespan (lower is better); FSSP has NO optimum, so
    there is no gap. A compile/runtime failure yields the ``_FAIL_COST`` (1e9) sentinel,
    which is preserved as ``null`` in the matrix and marked via a per-heuristic ``status``
    flag (``ok`` | ``crash``) so the downstream W/dispersion computation decides how to treat
    it. Nothing is penalty-filled.

    Artefacts saved under ``out_root/<json-stem>/<timestamp>/`` (out_root defaults to
    ``.logs/posthoc_analyses``):
      * ``perf_matrix.csv``  -> rows = heuristics (indexed by cand_id), cols = instance_00..,
                                values = raw makespan (empty cell = crash).
      * ``rank_matrix.csv``  -> same shape, per-instance ascending rank of makespan (1 = best);
                                crashed cells rank last (NaN-aware, method='average').
      * ``perf_matrix.json`` -> the matrix as records + metadata (instance config, per-heuristic
                                status + stated ``score``). No mean_opt (FSSP has none).
    """
    t_start = time.time()
    json_path = pathlib.Path(json_path)
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    if not json_path.exists():
        raise FileNotFoundError(f"Missing heuristics JSON: {json_path}")

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

    instances = load_instances(n_instances, n_jobs, m_low, m_high, instance_source)
    print(f"Scoring {len(heuristics)} heuristics x {len(instances)} instances "
          f"on {n_cores} cores...", flush=True)

    t_score = time.time()
    scores, runtimes, per_inst = batch_scoring_parallel(
        instances, heuristics, n_cores=n_cores, return_per_instance=True)
    print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)

    # --- Build the (heuristic x instance) raw-makespan matrix. _FAIL_COST / non-finite -> NaN. ---
    inst_cols = [f"instance_{i:03d}" for i in range(len(instances))]

    def _row(pinst):
        vals = [float(c) for c in (pinst or [])]
        # A per-instance failure surfaces as the _FAIL_COST sentinel; a compile crash returns
        # [_FAIL_COST]*n. Treat any non-finite or >= _FAIL_COST cell as a crash (NaN).
        vals = [(v if (np.isfinite(v) and v < _FAIL_COST) else np.nan) for v in vals]
        vals += [np.nan] * (len(inst_cols) - len(vals))
        return vals[:len(inst_cols)]

    mat = pd.DataFrame([_row(p) for p in per_inst], index=cand_ids, columns=inst_cols)
    mat.index.name = "cand_id"
    rank_mat = mat.rank(axis=0, method="average", ascending=True, na_option="bottom")

    status = {
        cid: ("ok" if (np.isfinite(s) and s < _FAIL_COST) else "crash")
        for cid, s in zip(cand_ids, scores)
    }

    if out_root is None:
        out_root = ROOT / ".logs" / "posthoc_analyses"
    out_dir = pathlib.Path(out_root) / json_path.stem / time.strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    mat.to_csv(out_dir / "perf_matrix.csv")
    rank_mat.to_csv(out_dir / "rank_matrix.csv")

    meta = {
        "source_json": str(json_path),
        "metric": "mean_makespan (lower is better; FSSP has no optimum -> no gap)",
        "n_heuristics": len(heuristics),
        "n_instances": len(instances),
        "instance_config": {"n_instances": n_instances, "n_jobs": n_jobs,
                            "m_low": m_low, "m_high": m_high,
                            "instance_source": instance_source},
        "heuristics": [
            {
                "cand_id": cid,
                "stated_score": stated_score.get(cid),   # search-time score from the JSON
                "full_mean_cost": (float(s) if (np.isfinite(s) and s < _FAIL_COST) else None),
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
    print(f"  {n_ok}/{len(heuristics)} heuristics scored ok "
          f"({len(heuristics) - n_ok} crash)  "
          f"total_wall={time.time() - t_start:.1f}s", flush=True)
    print(f"Saved -> {out_dir}", flush=True)
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-evaluate trajectory checkpoints on an FSSP-GLS validation instance pool."
    )
    parser.add_argument(
        "--exp", type=str, required=False, default=None,
        help="Path to the experiment directory, e.g. "
             "'.logs/reprod_eoh_fssp_gls_vllm/2026-08-08/.../'",
    )
    parser.add_argument(
        "--heuristics-json", type=str, default=None,
        help="Path to a heuristics JSON (e.g. 'src/init_pop/eoh_fssp_gls.json'). When given, "
             "score every heuristic in the file against the full instance suite and dump "
             "the per-(heuristic, instance) performance matrix to .logs/posthoc_analyses "
             "(for Kendall's W / dispersion analysis). Mutually exclusive with --exp.",
    )
    parser.add_argument("--n-instances", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=50,
                        help="Jobs per instance (synthetic pool only; ignored for pregenerated).")
    parser.add_argument("--m-low", type=int, default=2, help="Min machines (synthetic only).")
    parser.add_argument("--m-high", type=int, default=20, help="Max machines (synthetic only).")
    parser.add_argument("--instance-source", type=str, default="pregenerated",
                        choices=["synthetic", "pregenerated"],
                        help="'synthetic' (default): fresh seeded held-out pool; 'pregenerated': "
                             "the fixed EoH/Taillard benchmark files.")
    parser.add_argument("--n-cores", type=int, default=20)
    parser.add_argument("--mode", type=str, default="valid", choices=["valid", "gen"],
                        help="'valid' for full trajectory, 'gen' for the last candidate only.")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="If set, evaluate the per-generation incumbents instead of the full trajectory.")
    args = parser.parse_args()

    if args.heuristics_json:
        evaluate_heuristics_file(
            args.heuristics_json, args.n_instances, args.n_jobs, args.m_low, args.m_high,
            instance_source=args.instance_source, n_cores=args.n_cores,
        )
        sys.exit(0)

    if not args.exp:
        parser.error("one of --exp or --heuristics-json is required")

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    print(f"Processing experiment: {exp_path}")
    result = evaluate_single(exp_path, args.n_instances, args.n_jobs, args.m_low, args.m_high,
                             args.instance_source, args.n_cores, args.mode, args.has_incumbent)
    out_path = _out_path(exp_path, args.mode, args.n_instances, args.n_jobs, args.has_incumbent)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serialisable)
    print(f"Saved -> {out_path}")
