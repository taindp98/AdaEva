import sys, json, time, pathlib, re
import numpy as np
import pandas as pd
import glob
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD")]

from utils import ConfigAS
from racing.eoh_obp import _score_obp_inst
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from utils.obp_utils import _obp_lower_bound
from llm4ad.base import SecureEvaluator


def load_instances(n_instances: int = 25, n_items: int = 5000, capacity: int = 100, seed: int = 1):
    """Create OBP instances independently of any experiment directory."""
    random.seed(seed)
    np.random.seed(seed)
    evaluation = OBPEvaluation(
        timeout_seconds=30,
        n_instances=n_instances,
        n_items=n_items,
        capacity=capacity,
    )
    instances = list(evaluation._datasets.values())
    lower_bounds = []
    for inst in instances:
        inst["items"] = np.array(inst["items"])
        inst["capacity"] = float(inst["capacity"])
        lower_bounds.append(_obp_lower_bound(inst["items"], inst["capacity"]))
    avg_lb = float(np.mean(lower_bounds))
    print(
        f"Average lower bound for {n_instances} instances of size {n_items} with capacity {capacity}: {avg_lb:.4f}"
    )
    return avg_lb, instances


def _compile_priority(source: str) -> callable:
    """Compile a priority function from source string."""
    ns = {"np": np}
    exec(source, ns)
    return ns["priority"]


def single_batch_scoring(instances: list, source: str) -> float:
    """
    Score one heuristic (given as source string) across all instances.
    Returns the average number of bins used (lower is better).
    Returns np.inf on compile/runtime error.
    """
    try:
        priority = _compile_priority(source)
    except Exception as e:
        print(f"  [compile] error: {e}", flush=True)
        return np.inf
    costs = [_score_obp_inst(priority, inst, 0) for inst in instances]
    return float(np.mean(costs))


def _worker(args):
    """Top-level worker for multiprocessing: scores one heuristic on all instances."""
    idx, cand_id, source, instances = args
    pid = mp.current_process().pid
    t0 = time.perf_counter()
    try:
        priority = _compile_priority(source)
    except Exception as e:
        print(f"  [worker pid={pid}] {cand_id}: compile error: {e}", flush=True)
        return idx, np.inf, []
    
    objs = []
    for i, inst in enumerate(instances):
        obj = _score_obp_inst(priority, inst, 0)
        objs.append(obj)
        # print per-instance progress every 8 instances so long workers show signs of life
        if (i + 1) % 8 == 0 or (i + 1) == len(instances):
            costs_so_far = np.array(objs)
            print(f"    [worker pid={pid}] {cand_id}  inst {i+1}/{len(instances)}  "
                  f"last_cost={obj:.4f}  "
                  f"running_mean={float(np.mean(costs_so_far)):.4f}  "
                  f"worker_wall={time.perf_counter()-t0:.0f}s", flush=True)
            
    score = float(np.mean(objs))
    dt = time.perf_counter() - t0
    print(f"  [worker pid={pid}] {cand_id}: DONE  "
          f"mean_cost={score:.4f}  wall={dt:.1f}s", flush=True)
    return idx, score, [float(o) for o in objs]   # per-instance costs for heldout_eval.jsonl


def batch_scoring_parallel(
    instances: list, heuristics: list, n_cores: int = 20
) -> list:
    tasks = [(i, h["cand_id"], h["source"], instances) for i, h in enumerate(heuristics)]
    scores = [np.inf] * len(heuristics)
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
            for idx, score, objs in pool.imap_unordered(_worker, tasks, chunksize=1):
                scores[idx] = score
                per_inst[idx] = objs
                n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(f"  [{n_done}/{n_total}] done  elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    finally:
        _stop_heartbeat.set()

    return scores, per_inst


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def evaluate_single(exp_path: pathlib.Path, n_instances: int = 25, n_items: int = 5000, capacity: int = 100, n_cores: int = 20, mode: str = "valid", has_incumbent: bool = False, incumbent_key: str = "incumbents_mean_rank") -> dict:
    t_start = time.time()
    if has_incumbent:
        _rule = incumbent_key.replace("incumbents_", "")   # mean_rank | mean_cost
        out_path = exp_path / f"valid_trajectory_{_rule}.json"
    else:
        # No incumbent series (single-trajectory callers, e.g. tiny/reprod eoh_obp
        # pass has_incumbent=False). ``_rule`` is still referenced below when writing
        # heldout_eval.jsonl, so bind it here to avoid an UnboundLocalError.
        _rule = None
        out_path = exp_path / f"{mode}_trajectory_ni{n_instances}_nit{n_items}_cap{capacity}.json"

    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing: {trajectory_path}")

    with open(trajectory_path) as f:
        traj_data = json.load(f)
    # Evaluate the per-generation incumbents, not the full trajectory. A single
    # incumbent series is stored: incumbents=[...] (the coverage-tiered rank
    # incumbent scored by lifetime mean cost).
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
        if traj:
            final_budget = traj[-1]["used_budget"]
            slices = [0.2, 0.4, 0.6, 0.8, 1.0]
            target_budgets = [s * final_budget for s in slices]
            selected = []
            for tb in target_budgets:
                closest = min(traj, key=lambda entry: abs(entry["used_budget"] - tb))
                if closest not in selected:
                    selected.append(closest)
            traj = selected
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

    # --- Per-candidate score caches (filled during evaluation) ---
    cached_gap: dict = {}    # cand_id -> float gap_pct
    cached_score: dict = {}  # cand_id -> float raw mean cost (full-suite); pairs with the
                             # in-race partial score in fitness_reliability_log_<rule>.jsonl
    cached_avg_lb: float | None = None


    pending = [h for h in all_unique if h["cand_id"] not in cached_gap]

    if not pending:
        print("All candidates already scored — nothing to evaluate.", flush=True)
        if cached_avg_lb is not None:
            avg_lb = cached_avg_lb
        else:
            avg_lb, _ = load_instances(n_instances, n_items, capacity)
    else:
        avg_lb, instances = load_instances(n_instances, n_items, capacity)
        print(f"  OBP instances prepared in {time.time() - t_start:.1f}s", flush=True)

        if cached_avg_lb is not None and abs(cached_avg_lb - avg_lb) > 1e-6:
            print(f"  Warning: cached avg_lb={cached_avg_lb:.6f} != "
                  f"recomputed avg_lb={avg_lb:.6f} — using recomputed value.", flush=True)

        print(f"Scoring {len(pending)} pending heuristics "
              f"({len(cached_gap)} cached, {n_traj_entries} total traj entries, "
              f"{len(instances)} instances, {n_cores} cores)...", flush=True)

        t_score = time.time()
        scores, per_inst = batch_scoring_parallel(instances, pending, n_cores=n_cores)
        print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)

        for h, raw_cost in zip(pending, scores):
            cid = h["cand_id"]
            cached_gap[cid] = (raw_cost - avg_lb) / avg_lb * 100
            cached_score[cid] = float(raw_cost)

        # heldout_eval.jsonl: one row per (candidate, held-out instance), tagged
        # with the gen_id of the trajectory entry(ies) referencing the candidate. No extra
        # model evals — the per-instance costs come from the validation just run. OBP
        # instances are homogeneous, so per_class is None.
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
                    "incumbent_rule": _rule,   # mean_rank | mean_cost (both series share this file)
                    "cost": (float(cost) if fin else None),
                    "status": ("ok" if fin and cost < 1e5 else "crash"),
                    "per_class": None,
                })
        if _hrows:
            with open(exp_path / "heldout_eval.jsonl", "a") as _hf:
                for _r in _hrows:
                    _hf.write(json.dumps(_r, default=_to_serialisable) + "\n")
            print(f"  heldout_eval.jsonl: wrote {len(_hrows)} (candidate, instance) rows",
                  flush=True)

    gap_pct = [
        cached_gap.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]
    # Raw full-suite mean cost aligned to `traj`; same metric as the in-race partial
    # score, so finalize_reliability can pair partial vs full (doc §F).
    score = [
        cached_score.get(pt["cand_id"]) if pt["source"] is not None else None
        for pt in traj
    ]

    valid_gaps = [g for g in gap_pct if g is not None]
    best_gap = min(valid_gaps) if valid_gaps else float("inf")
    n_covered = sum(1 for g in gap_pct if g is not None)
    print(f"  covered {n_covered}/{len(traj)} trajectory entries  "
          f"best gap={best_gap:.4f}%  avg_lb={avg_lb:.4f}  "
          f"total_wall={time.time() - t_start:.1f}s", flush=True)

    return {
        "avg_lb": avg_lb,
        "cand_id": [pt["cand_id"] for pt in traj],
        "used_budget": [pt["used_budget"] for pt in traj],
        "gap_pct": gap_pct,
        "score": score,   # raw full-suite mean cost (pairs with partial in reliability log)
    }


def evaluate_heuristics_file(
    json_path: pathlib.Path,
    n_instances: int = 25,
    n_items: int = 5000,
    capacity: int = 100,
    seed: int = 1,
    n_cores: int = 20,
    out_root: pathlib.Path | None = None,
) -> dict:
    """Score every heuristic stored in a heuristics JSON (e.g. src/init_pop/eoh_obp.json)
    against the FULL OBP instance suite and dump the detailed per-(heuristic, instance)
    performance matrix for post-hoc analysis (Kendall's W across instances, cost/score
    dispersion).

    Input JSON schema (same as init_pop / heuristics.json): ``{"heuristics": [{"cand_id",
    "source", "score" (optional, the stated/search-time score), ...}, ...]}``.

    Raw per-instance cost = bins used (lower is better); a compile/runtime failure yields
    ``np.inf``, which is preserved as ``null`` in the matrix and marked via a per-heuristic
    ``status`` flag (``ok`` | ``crash``) so the downstream W/dispersion computation decides
    how to treat it. Nothing is penalty-filled here.

    Artefacts saved under ``out_root/<json-stem>/<timestamp>/`` (out_root defaults to
    ``.logs/posthoc_analyses``):
      * ``perf_matrix.csv``  -> rows = heuristics (indexed by cand_id), cols = instance_00..,
                                values = raw cost (empty cell = crash/inf).
      * ``rank_matrix.csv``  -> same shape, per-instance ascending rank of cost (1 = best);
                                crashed cells rank last (NaN-aware, method='average').
      * ``perf_matrix.json`` -> the matrix as records + metadata (avg_lb, instance config,
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
    # Keep only rows with a usable source; remember stated search-time score if present.
    heuristics = [h for h in heuristics if h.get("source")]
    if not heuristics:
        raise ValueError(f"{json_path}: no heuristics with a non-empty 'source'.")
    cand_ids = [h["cand_id"] for h in heuristics]
    stated_score = {h["cand_id"]: h.get("score") for h in heuristics}

    avg_lb, instances = load_instances(n_instances, n_items, capacity, seed=seed)
    print(f"  OBP instances prepared in {time.time() - t_start:.1f}s", flush=True)
    print(f"Scoring {len(heuristics)} heuristics x {len(instances)} instances "
          f"on {n_cores} cores...", flush=True)

    t_score = time.time()
    scores, per_inst = batch_scoring_parallel(instances, heuristics, n_cores=n_cores)
    print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)

    # --- Build the (heuristic x instance) raw-cost matrix. inf -> NaN. ---
    inst_cols = [f"instance_{i:03d}" for i in range(len(instances))]

    def _row(pinst):
        vals = [float(c) for c in (pinst or [])]
        # Pad short/crashed rows (crash returns []) with NaN so the frame stays rectangular.
        vals = [(v if np.isfinite(v) else np.nan) for v in vals]
        vals += [np.nan] * (len(inst_cols) - len(vals))
        return vals[:len(inst_cols)]

    mat = pd.DataFrame([_row(p) for p in per_inst], index=cand_ids, columns=inst_cols)
    mat.index.name = "cand_id"
    # Per-instance ascending rank of cost (1 = fewest bins = best); crashes (NaN) rank last.
    rank_mat = mat.rank(axis=0, method="average", ascending=True, na_option="bottom")

    status = {
        cid: ("ok" if np.isfinite(s) else "crash")
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
        "instance_config": {"n_instances": n_instances, "n_items": n_items,
                            "capacity": capacity, "seed": seed},
        "avg_lb": avg_lb,
        "heuristics": [
            {
                "cand_id": cid,
                "stated_score": stated_score.get(cid),   # search-time score from the JSON
                "full_mean_cost": (float(s) if np.isfinite(s) else None),
                "gap_pct": ((float(s) - avg_lb) / avg_lb * 100 if np.isfinite(s) else None),
                "status": status[cid],
            }
            for cid, s in zip(cand_ids, scores)
        ],
        # matrix as records for a language-agnostic reload (NaN -> None).
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
          f"({len(heuristics) - n_ok} crash)  avg_lb={avg_lb:.4f}  "
          f"total_wall={time.time() - t_start:.1f}s", flush=True)
    print(f"Saved -> {out_dir}", flush=True)
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-evaluate trajectory checkpoints on the full OBP instance pool."
    )
    parser.add_argument(
        "--exp", type=str, required=False, default=None,
        help="Path to the experiment directory, e.g. "
             "'.logs/mab_eoh_obp/2026-06-18/225842_1_default'",
    )
    parser.add_argument(
        "--heuristics-json", type=str, default=None,
        help="Path to a heuristics JSON (e.g. 'src/init_pop/eoh_obp.json'). When given, "
             "score every heuristic in the file against the full instance suite and dump "
             "the per-(heuristic, instance) performance matrix to .logs/posthoc_analyses "
             "(for Kendall's W / dispersion analysis). Mutually exclusive with --exp.",
    )
    parser.add_argument("--seed", type=int, default=1,
                        help="RNG seed for the OBP instance suite (heuristics-json mode).")
    parser.add_argument("--n-instances", type=int, default=25)
    parser.add_argument("--n-items", type=int, default=5000)
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--n-cores", type=int, default=20)
    parser.add_argument("--mode", type=str, default="valid", choices=["valid", "gen"],
                        help="Evaluation mode: 'valid' for full trajectory, 'gen' for sliced budget checkpoint candidates (0.2, 0.4, 0.6, 0.8, 1.0).")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="If set, only evaluate candidates that were selected as incumbents in the trajectory.")
    args = parser.parse_args()

    if args.heuristics_json:
        evaluate_heuristics_file(
            args.heuristics_json, args.n_instances, args.n_items, args.capacity,
            seed=args.seed, n_cores=args.n_cores,
        )
        sys.exit(0)

    if not args.exp:
        parser.error("one of --exp or --heuristics-json is required")

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    print(f"Processing experiment: {exp_path}")
    result = evaluate_single(exp_path, args.n_instances, args.n_items, args.capacity, args.n_cores, args.mode, args.has_incumbent)
    if args.has_incumbent:
        out_path = exp_path / f"{args.mode}_trajectory_incumbent_ni{args.n_instances}_nit{args.n_items}_cap{args.capacity}.json"
    else:
        out_path = exp_path / f"{args.mode}_trajectory_ni{args.n_instances}_nit{args.n_items}_cap{args.capacity}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serialisable)
    print(f"Saved -> {out_path}")
