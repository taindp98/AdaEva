import sys, json, pathlib, time, threading
import numpy as np
import pandas as pd
import multiprocessing as mp
import math

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD")]

from racing.eoh_tsp_gls import _opt_cost
from llm4ad.task.optimization.tsp_gls_2O.get_instance import GetData
from llm4ad.task.optimization.tsp_gls_2O.evaluation import solve_without_time, solve_with_time

def score_tsp_inst(fn: callable, inst) -> float:
    """Run ``fn`` as ``update_edge_distance`` on one TSP instance; return (cost, running_time).

    Uses solve_without_time (the 60s-capped variant) for the cost, then separately
    times the call so we still get a wall-time figure.  The _with_time variant runs
    all 1000 iterations unconditionally and can take hundreds of seconds per instance
    for slow heuristics — it must not be used here.
    """
    t0 = time.perf_counter()
    try:
        cost = solve_without_time(inst, fn)
        cost = float(cost) if math.isfinite(float(cost)) else 1e6
    except Exception as e:
        print(f"  [score_tsp_inst] error: {e}", flush=True)
        cost = 1e6
    running_time = time.perf_counter() - t0
    return cost, running_time


def load_instances(n_instances: int = 64, problem_size: int = 100, seed: int = 2024):
    """Generate TSP instances and compute per-instance Concorde optima."""
    np.random.seed(seed)
    instances = GetData(n_instances, problem_size).generate_instances()
    print(f"Solving Concorde optima for {n_instances} instances (size {problem_size})...")
    opt_costs = [_opt_cost(inst) for inst in instances]
    mean_opt = float(np.mean(opt_costs))
    print(f"  mean_opt={mean_opt:.4f}  min={min(opt_costs):.4f}  max={max(opt_costs):.4f}")
    return mean_opt, instances


def _compile_update_edge_distance(source: str) -> callable:
    ns = {"np": np}
    exec(source, ns)
    return ns["update_edge_distance"]


def single_batch_scoring(instances: list, source: str) -> float:
    """Score one heuristic across all instances. Returns mean tour cost (lower is better)."""
    try:
        fn = _compile_update_edge_distance(source)
    except Exception as e:
        print(f"  [compile] error: {e}", flush=True)
        return 1e6, 1e6
    objs = []
    for inst in instances:
        obj = score_tsp_inst(fn, inst)
        objs.append(obj)
    objs = np.array(objs)
    costs = objs[:, 0]
    runtimes = objs[:, 1]
    avg_cost = float(np.mean(costs))
    avg_runtime = float(np.mean(runtimes))
    return avg_cost, avg_runtime


def _worker(args):
    idx, cand_id, source, instances = args
    pid = mp.current_process().pid
    t0 = time.perf_counter()
    try:
        fn = _compile_update_edge_distance(source)
    except Exception as e:
        print(f"  [worker pid={pid}] {cand_id}: compile error: {e}", flush=True)
        return idx, 1e6, 1e6, []
    objs = []
    for i, inst in enumerate(instances):
        obj = score_tsp_inst(fn, inst)
        objs.append(obj)
        # print per-instance progress every 8 instances so long workers show signs of life
        if (i + 1) % 8 == 0 or (i + 1) == len(instances):
            costs_so_far = np.array(objs)[:, 0]
            print(f"    [worker pid={pid}] {cand_id}  inst {i+1}/{len(instances)}  "
                  f"last_cost={obj[0]:.4f}  last_gls={obj[1]:.1f}s  "
                  f"running_mean={float(np.mean(costs_so_far)):.4f}  "
                  f"worker_wall={time.perf_counter()-t0:.0f}s", flush=True)
    objs = np.array(objs)
    costs = objs[:, 0]; rts = objs[:, 1]
    score = float(np.mean(costs)); runtime = float(np.mean(rts))
    dt = time.perf_counter() - t0
    print(f"  [worker pid={pid}] {cand_id}: DONE  "
          f"mean_cost={score:.4f}  mean_gls_time={runtime:.1f}s  wall={dt:.1f}s", flush=True)
    return idx, score, runtime, [float(c) for c in costs]   # per-instance costs (heldout_eval)


def batch_scoring_parallel(instances: list, heuristics: list, n_cores: int = 20) -> list:
    tasks = [(i, h["cand_id"], h["source"], instances) for i, h in enumerate(heuristics)]
    scores = [np.inf] * len(heuristics)
    runtimes = [np.inf] * len(heuristics)
    per_inst: list = [None] * len(heuristics)   # per-heuristic list of per-instance costs
    n_total = len(heuristics)
    n_done = 0
    t0 = time.time()

    # Heartbeat: every 120s print how many are still in-flight so we know the pool isn't hung
    _stop_heartbeat = threading.Event()
    def _heartbeat():
        while not _stop_heartbeat.wait(120):
            elapsed = time.time() - t0
            remaining = n_total - n_done
            print(f"  [heartbeat] {n_done}/{n_total} done  {remaining} in-flight  "
                  f"elapsed={elapsed:.0f}s", flush=True)
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=n_cores) as pool:
            for idx, score, runtime, objs in pool.imap_unordered(_worker, tasks, chunksize=1):
                scores[idx] = score
                runtimes[idx] = runtime
                per_inst[idx] = objs
                n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(f"  [{n_done}/{n_total}] done  elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    finally:
        _stop_heartbeat.set()

    return scores, runtimes, per_inst


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def evaluate_single(exp_path: pathlib.Path, n_instances: int = 64, problem_size: int = 100, n_cores: int = 20, mode: str = "valid", has_incumbent: bool = False, incumbent_key: str = "incumbents_mean_rank") -> dict:
    """
    Evaluate the trajectory of TSP-GLS heuristics on Concorde instances.
    Unlike OBP where all raw instance arrays are kept in memory, TSP evaluation
    writes instance data to disk for each candidate process, which can be
    I/O bound. This script is designed to be resumable: if interrupted, only
    unique candidates are evaluated, then the file is updated.
    """
    t_start = time.time()
    if has_incumbent:
        _rule = incumbent_key.replace("incumbents_", "")   # mean_rank | mean_cost
        out_path = exp_path / f"valid_trajectory_{_rule}.json"
    else:
        out_path = exp_path / f"{mode}_trajectory_ni{n_instances}_prob{problem_size}.json"

    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing: {trajectory_path}")

    with open(trajectory_path) as f:
        traj_data = json.load(f)
    # Evaluate the per-generation incumbents, not the full trajectory. A single
    # incumbent series is stored: incumbents=[...] (the coverage-tiered rank
    # incumbent scored by lifetime mean cost). Each entry carries
    # {gen_id, cand_id, used_budget, score, n_instances}.
    if has_incumbent:
        incs = traj_data.get(incumbent_key)
        if not isinstance(incs, list):
            raise KeyError(
                f"trajectory.json at {trajectory_path} has no '{incumbent_key}' list "
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

    # Collect all unique candidates that appear in the trajectory
    seen_ids: dict = {}
    for t in traj:
        if t["source"] is not None and t["cand_id"] not in seen_ids:
            seen_ids[t["cand_id"]] = t
    all_unique = list(seen_ids.values())
    n_traj_entries = sum(1 for t in traj if t["source"] is not None)

    # --- Resume: load already-scored candidates from existing output ---
    # Cache maps cand_id -> (gap_pct, runtime) from the previous run.
    # gap_pct is stored directly (not raw mean_cost) so mean_opt must be
    # consistent; we recompute it fresh and warn if it differs.
    cached_gap: dict = {}    # cand_id -> float gap_pct
    cached_rt: dict = {}     # cand_id -> float mean_runtime
    cached_mean_opt: float | None = None

    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        cached_mean_opt = prev.get("mean_opt")
        prev_gaps = prev.get("gap_pct", [])
        prev_rts = prev.get("runtimes", [None] * len(prev_gaps))
        for i, pt in enumerate(traj):
            if i >= len(prev_gaps):
                break
            cid = pt["cand_id"]
            g = prev_gaps[i]
            r = prev_rts[i] if i < len(prev_rts) else None
            if g is not None and cid not in cached_gap:
                cached_gap[cid] = g
                cached_rt[cid] = r
        print(f"Resume: {len(cached_gap)}/{len(all_unique)} unique candidates "
              f"already scored in {out_path.name}", flush=True)
    else:
        print("No existing output found — starting from scratch.", flush=True)

    pending = [h for h in all_unique if h["cand_id"] not in cached_gap]

    if not pending:
        print("All candidates already scored — nothing to evaluate.", flush=True)
        # Use cached mean_opt; recompute only if missing
        if cached_mean_opt is not None:
            mean_opt = cached_mean_opt
        else:
            mean_opt, _ = load_instances(n_instances, problem_size)
    else:
        mean_opt, instances = load_instances(n_instances, problem_size)
        print(f"  Concorde optima done in {time.time() - t_start:.1f}s", flush=True)

        if cached_mean_opt is not None and abs(cached_mean_opt - mean_opt) > 1e-6:
            print(f"  Warning: cached mean_opt={cached_mean_opt:.6f} != "
                  f"recomputed mean_opt={mean_opt:.6f} — using recomputed value.", flush=True)

        print(f"Scoring {len(pending)} pending heuristics "
              f"({len(cached_gap)} cached, {n_traj_entries} total traj entries, "
              f"{len(instances)} instances, {n_cores} cores)...", flush=True)

        t_score = time.time()
        scores, new_rts, per_inst = batch_scoring_parallel(instances, pending, n_cores=n_cores)
        print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)

        for h, raw_cost, rt in zip(pending, scores, new_rts):
            cid = h["cand_id"]
            cached_gap[cid] = (raw_cost - mean_opt) / mean_opt * 100
            cached_rt[cid] = rt

        # heldout_eval.jsonl: one row per (candidate, held-out instance), tagged
        # with the gen_id of the trajectory entry(ies) referencing the candidate. No extra
        # model evals. TSP instances are a homogeneous size, so per_class is None.
        _cand_gen: dict = {}
        for pt in traj:
            _cid = pt.get("cand_id")
            if _cid is not None and _cid not in _cand_gen:
                _cand_gen[_cid] = pt.get("gen_id")
        _hrows = []
        for h, pinst in zip(pending, per_inst):
            cid = h["cand_id"]
            gid = _cand_gen.get(cid)
            for i, cost in enumerate(pinst or []):
                fin = isinstance(cost, (int, float)) and np.isfinite(cost)
                _hrows.append({
                    "gen_id": gid, "cand_id": cid, "heldout_instance_id": int(i),
                    "incumbent_rule": (incumbent_key.replace("incumbents_", "")
                                       if has_incumbent else None),  # both series share this file
                    "cost": (float(cost) if fin else None),
                    "status": ("ok" if fin and cost < 1e6 else "crash"),
                    "per_class": None,
                })
        if _hrows:
            with open(exp_path / "heldout_eval.jsonl", "a") as _hf:
                for _r in _hrows:
                    _hf.write(json.dumps(_r, default=_to_serialisable) + "\n")
            print(f"  heldout_eval.jsonl: wrote {len(_hrows)} (candidate, instance) rows",
                  flush=True)

    # Reconstruct full parallel arrays aligned to traj
    gap_pct = [
        cached_gap.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]
    runtimes_out = [
        cached_rt.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]

    # Raw full-suite mean tour cost aligned to `traj`; same metric as the in-race
    # partial score, so finalize_reliability can pair partial vs full (doc §F).
    # Recovered from gap_pct (robust to the resume cache, which stores gap not cost):
    # gap = (cost - mean_opt)/mean_opt*100  =>  cost = mean_opt*(1 + gap/100).
    score = [
        (mean_opt * (1 + g / 100.0) if g is not None else None)
        for g in gap_pct
    ]

    valid_gaps = [g for g in gap_pct if g is not None]
    best_gap = min(valid_gaps) if valid_gaps else float("inf")
    n_covered = sum(1 for g in gap_pct if g is not None)
    print(f"  covered {n_covered}/{len(traj)} trajectory entries  "
          f"best gap={best_gap:.4f}%  mean_opt={mean_opt:.4f}  "
          f"total_wall={time.time() - t_start:.1f}s", flush=True)

    return {
        "mean_opt": mean_opt,
        "cand_id": [pt["cand_id"] for pt in traj],
        "used_budget": [pt["used_budget"] for pt in traj],
        "cpu_seconds": [pt["cpu_seconds"] for pt in traj],
        "gap_pct": gap_pct,
        "score": score,   # raw full-suite mean cost (pairs with partial in reliability log)
        "runtimes": runtimes_out,
    }


def evaluate_heuristics_file(
    json_path: pathlib.Path,
    n_instances: int = 64,
    problem_size: int = 100,
    seed: int = 2024,
    n_cores: int = 20,
    out_root: pathlib.Path | None = None,
) -> dict:
    """Score every heuristic stored in a heuristics JSON (e.g. src/init_pop/eoh_tsp_gls.json)
    against the FULL TSP-GLS instance suite and dump the detailed per-(heuristic, instance)
    performance matrix for post-hoc analysis (Kendall's W across instances, cost/score
    dispersion).

    Input JSON schema (same as init_pop / heuristics.json): ``{"heuristics": [{"cand_id",
    "source", "score" (optional, the stated/search-time score), ...}, ...]}``. The heuristic
    entry point is ``update_edge_distance``.

    Raw per-instance cost = mean GLS tour cost (lower is better); a compile/runtime failure
    yields the ``1e6`` sentinel (from score_tsp_inst / crash), which is preserved as ``null``
    in the matrix and marked via a per-heuristic ``status`` flag (``ok`` | ``crash``) so the
    downstream W/dispersion computation decides how to treat it. Nothing is penalty-filled.

    Artefacts saved under ``out_root/<json-stem>/<timestamp>/`` (out_root defaults to
    ``.logs/posthoc_analyses``):
      * ``perf_matrix.csv``  -> rows = heuristics (indexed by cand_id), cols = instance_00..,
                                values = raw tour cost (empty cell = crash).
      * ``rank_matrix.csv``  -> same shape, per-instance ascending rank of cost (1 = best);
                                crashed cells rank last (NaN-aware, method='average').
      * ``perf_matrix.json`` -> the matrix as records + metadata (mean_opt, instance config,
                                per-heuristic status + stated ``score``).
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

    mean_opt, instances = load_instances(n_instances, problem_size, seed=seed)
    print(f"  Concorde optima done in {time.time() - t_start:.1f}s", flush=True)
    print(f"Scoring {len(heuristics)} heuristics x {len(instances)} instances "
          f"on {n_cores} cores...", flush=True)

    t_score = time.time()
    scores, runtimes, per_inst = batch_scoring_parallel(instances, heuristics, n_cores=n_cores)
    print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)

    # --- Build the (heuristic x instance) raw-cost matrix. Sentinel 1e6 / non-finite -> NaN. ---
    inst_cols = [f"instance_{i:03d}" for i in range(len(instances))]

    def _row(pinst):
        vals = [float(c) for c in (pinst or [])]
        # A per-instance failure surfaces as the 1e6 sentinel from score_tsp_inst; treat it
        # (and any non-finite) as a crash cell. Crashed compile returns [] -> all-NaN row.
        vals = [(v if (np.isfinite(v) and v < 1e6) else np.nan) for v in vals]
        vals += [np.nan] * (len(inst_cols) - len(vals))
        return vals[:len(inst_cols)]

    mat = pd.DataFrame([_row(p) for p in per_inst], index=cand_ids, columns=inst_cols)
    mat.index.name = "cand_id"
    rank_mat = mat.rank(axis=0, method="average", ascending=True, na_option="bottom")

    status = {
        cid: ("ok" if (np.isfinite(s) and s < 1e6) else "crash")
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
        "n_heuristics": len(heuristics),
        "n_instances": len(instances),
        "instance_config": {"n_instances": n_instances, "problem_size": problem_size,
                            "seed": seed},
        "mean_opt": mean_opt,
        "heuristics": [
            {
                "cand_id": cid,
                "stated_score": stated_score.get(cid),   # search-time score from the JSON
                "full_mean_cost": (float(s) if (np.isfinite(s) and s < 1e6) else None),
                "gap_pct": ((float(s) - mean_opt) / mean_opt * 100
                            if (np.isfinite(s) and s < 1e6) else None),
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
          f"({len(heuristics) - n_ok} crash)  mean_opt={mean_opt:.4f}  "
          f"total_wall={time.time() - t_start:.1f}s", flush=True)
    print(f"Saved -> {out_dir}", flush=True)
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-evaluate trajectory checkpoints on the full TSP-GLS instance pool."
    )
    parser.add_argument(
        "--exp", type=str, required=False, default=None,
        help="Path to the experiment directory, e.g. "
             "'.logs/reprod_eoh_tsp_gls/2026-06-18/225842_1_default'",
    )
    parser.add_argument(
        "--heuristics-json", type=str, default=None,
        help="Path to a heuristics JSON (e.g. 'src/init_pop/eoh_tsp_gls.json'). When given, "
             "score every heuristic in the file against the full instance suite and dump "
             "the per-(heuristic, instance) performance matrix to .logs/posthoc_analyses "
             "(for Kendall's W / dispersion analysis). Mutually exclusive with --exp.",
    )
    parser.add_argument("--seed", type=int, default=2024,
                        help="RNG seed for the TSP instance suite (heuristics-json mode).")
    parser.add_argument("--n-instances", type=int, default=64)
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--n-cores", type=int, default=20)
    parser.add_argument("--mode", type=str, default="valid", choices=["valid", "gen"],
                        help="Evaluation mode: 'valid' for full trajectory, 'gen' for last candidate only.")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="If set, only evaluate candidates that were selected as incumbents in the trajectory.")
    args = parser.parse_args()

    if args.heuristics_json:
        evaluate_heuristics_file(
            args.heuristics_json, args.n_instances, args.problem_size,
            seed=args.seed, n_cores=args.n_cores,
        )
        sys.exit(0)

    if not args.exp:
        parser.error("one of --exp or --heuristics-json is required")

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    print(f"Processing experiment: {exp_path}")
    result = evaluate_single(exp_path, args.n_instances, args.problem_size, args.n_cores, args.mode, args.has_incumbent)
    if args.has_incumbent:
        out_path = exp_path / f"{args.mode}_trajectory_incumbent_ni{args.n_instances}_prob{args.problem_size}.json"
    else:
        out_path = exp_path / f"{args.mode}_trajectory_ni{args.n_instances}_prob{args.problem_size}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serialisable)
    print(f"Saved -> {out_path}")
