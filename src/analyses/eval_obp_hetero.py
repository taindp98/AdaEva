"""Heterogeneous OBP trajectory evaluation — flat mean gap over the training set.

Combines two siblings:

  * ``src/analyses/eval_obp.py`` — re-evaluates EVERY heuristic in an experiment's
    ``trajectory.json`` (joined to its source via ``heuristics.json``) on a fresh
    instance pool, in parallel across candidates, with resume + heartbeat.  That is
    the workflow mirrored here.
  * ``src/analyses/test_obp_hetero.py`` — the single-heuristic hetero tester; its
    data source (pre-generated instances from a pickle) and metric (per-instance
    relative gap ``(used_i - lb_i)/lb_i`` with ``lb = round(Σitems/cap)``, EoH-S
    convention) are reused here.

Difference from ``test_obp_hetero.py``: this scores a WHOLE trajectory of heuristics
(not one), and reports a **single flat mean over all 128 training instances** per
heuristic — NO per-``n_items`` grouping.  The metric is the mean-of-ratios relative
gap (each instance normalised by its own lower bound, then averaged), matching
``test_obp_hetero.py``'s ``overall.mean_gap``.

Input : ``--exp <dir>`` containing ``trajectory.json`` + ``heuristics.json``.
Output: ``<exp>/<mode>_hetero_train_trajectory.json`` with, per trajectory entry,
        the recomputed ``gap_pct`` (mean relative gap %) and ``used_budget``.
"""

import sys
import json
import time
import pathlib
import argparse
import pickle
import threading
import multiprocessing as mp

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD")]

from racing.eoh_obp import _score_obp_inst   # greedy packing -> raw bins used

# Training split: the fixed 128-instance heterogeneous set (capacity 100,
# n_items in [207, 1956]); same default as test_obp_hetero.py's training split.
DEFAULT_DATA_FILE = ROOT / "packages" / "EoH-S" / "datasets" / "obp" / "dataset_200_2k_128_5_80.pkl"


# --------------------------------------------------------------------------- #
# Lower bound + instance loading (EoH-S convention, matching test_obp_hetero.py)
# --------------------------------------------------------------------------- #

def _obp_lb_round(items, capacity) -> int:
    """EoH-S lower bound: ``int(Σitems / capacity + 0.5)`` (round-half-up)."""
    return int(float(np.sum(items)) / float(capacity) + 0.5)


def load_training_instances(data_file):
    """Load the FULL set of pre-generated OBP instances + per-instance round lb.
    Returns ``(instances, lbs, mean_lb)``."""
    with open(data_file, "rb") as f:
        datasets = pickle.load(f)
    instances = list(datasets.values())
    lbs = []
    for inst in instances:
        inst["items"] = np.asarray(inst["items"])
        inst["capacity"] = float(inst["capacity"])
        if "num_items" not in inst:
            inst["num_items"] = len(inst["items"])
        lbs.append(_obp_lb_round(inst["items"], inst["capacity"]))
    mean_lb = float(np.mean(lbs)) if lbs else float("inf")
    print(f"Loaded {len(instances)} training instances from {data_file}  "
          f"(mean lb (round) = {mean_lb:.2f})", flush=True)
    return instances, lbs, mean_lb


def _compile_priority(source: str) -> callable:
    ns = {"np": np}
    exec(source, ns)
    if "priority" not in ns:
        raise ValueError("compiled source did not define `priority`")
    return ns["priority"]


# --------------------------------------------------------------------------- #
# Parallel scoring — one worker scores one heuristic over ALL 128 instances.
# Instances + lbs live in worker globals (set once per worker), so only
# (idx, cand_id, source) travels per task — no re-pickling the instance pool.
# --------------------------------------------------------------------------- #

_G: dict = {}


def _init_worker(instances, lbs):
    _G["instances"] = instances
    _G["lbs"] = lbs


def _worker(args):
    """Score one heuristic -> mean relative gap (mean-of-ratios) over all instances.
    Returns ``(idx, mean_gap, mean_bins, per_inst_gaps)``; a compile error or any per-instance
    crash makes ``mean_gap`` (and ``mean_bins``) ``inf`` so the heuristic is
    treated as failed (mirrors eval_obp.py's inf-propagating mean)."""
    idx, cand_id, source = args
    pid = mp.current_process().pid
    t0 = time.perf_counter()
    instances, lbs = _G["instances"], _G["lbs"]
    try:
        priority = _compile_priority(source)
    except Exception as e:
        print(f"  [worker pid={pid}] {cand_id}: compile error: {e}", flush=True)
        return idx, np.inf, np.inf, [float("inf")] * len(instances)

    gaps, bins = [], []
    n = len(instances)
    for i, (inst, lb) in enumerate(zip(instances, lbs)):
        used = _score_obp_inst(priority, inst, 0)     # raw bins, or inf on crash
        bins.append(used)
        gaps.append((used - lb) / lb)                 # inf propagates on crash
        if (i + 1) % 32 == 0 or (i + 1) == n:
            print(f"    [worker pid={pid}] {cand_id}  inst {i+1}/{n}  "
                  f"running_gap={float(np.mean(gaps)) * 100:.4f}%  "
                  f"worker_wall={time.perf_counter()-t0:.0f}s", flush=True)

    mean_gap = float(np.mean(gaps))                   # inf if any instance crashed
    mean_bins = float(np.mean(bins))
    dt = time.perf_counter() - t0
    print(f"  [worker pid={pid}] {cand_id}: DONE  mean_gap={mean_gap*100:.4f}%  "
          f"mean_bins={mean_bins:.2f}  wall={dt:.1f}s", flush=True)
    return idx, mean_gap, mean_bins, [float(g) for g in gaps]


def batch_scoring_parallel(instances, lbs, heuristics, n_cores=20, return_per_instance: bool = False):
    """Score each heuristic (parallel across candidates).  Returns two lists
    aligned to ``heuristics``: ``mean_gaps`` and ``mean_bins`` (or a 3-tuple
    with ``per_inst_gaps`` if return_per_instance=True)."""
    tasks = [(i, h["cand_id"], h["source"]) for i, h in enumerate(heuristics)]
    mean_gaps = [np.inf] * len(heuristics)
    mean_bins = [np.inf] * len(heuristics)
    per_inst_gaps = [None] * len(heuristics)
    n_total = len(heuristics)
    n_done = 0
    t0 = time.time()

    _stop = threading.Event()
    def _heartbeat():
        while not _stop.wait(120):
            print(f"  [heartbeat] {n_done}/{n_total} done  {n_total-n_done} in-flight  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=n_cores, initializer=_init_worker,
                      initargs=(instances, lbs)) as pool:
            for idx, mg, mb, pg in pool.imap_unordered(_worker, tasks, chunksize=1):
                mean_gaps[idx] = mg
                mean_bins[idx] = mb
                per_inst_gaps[idx] = pg
                n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(f"  [{n_done}/{n_total}] done  elapsed={elapsed:.0f}s  "
                      f"rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)
    finally:
        _stop.set()

    if return_per_instance:
        return mean_gaps, mean_bins, per_inst_gaps
    return mean_gaps, mean_bins


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# --------------------------------------------------------------------------- #
# Trajectory evaluation (mirrors eval_obp.py: join sources, dedupe, resume)
# --------------------------------------------------------------------------- #

def evaluate_trajectory(exp_path, data_file, n_cores=20, mode="valid", has_incumbent: bool = False, incumbent_key: str = "incumbents_mean_rank") -> dict:
    t_start = time.time()
    if has_incumbent:
        out_path = exp_path / f"{mode}_hetero_train_trajectory_incumbent.json"
    else:
        out_path = exp_path / f"{mode}_hetero_train_trajectory.json"

    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing: {trajectory_path}")
    if not heuristic_path.exists():
        raise FileNotFoundError(f"Missing: {heuristic_path}")

    with open(trajectory_path) as f:
        traj_data = json.load(f)
    # Evaluate the per-generation incumbents, not the full trajectory. A single
    # incumbent series is stored: incumbents=[...] (the coverage-tiered rank
    # incumbent scored by lifetime mean cost).
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
        df_heuristics = pd.DataFrame(json.load(f)["heuristics"])

    # Join each trajectory entry to its heuristic source.
    for t in traj:
        row = df_heuristics[df_heuristics["cand_id"] == t["cand_id"]]["source"].values
        t["source"] = row[0] if len(row) > 0 else None
        if len(row) == 0:
            print(f"  Warning: no heuristic found for cand_id={t['cand_id']}", flush=True)

    # Unique candidates appearing in the trajectory (score each once).
    seen: dict = {}
    for t in traj:
        if t["source"] is not None and t["cand_id"] not in seen:
            seen[t["cand_id"]] = t
    all_unique = list(seen.values())
    n_traj_entries = sum(1 for t in traj if t["source"] is not None)

    # --- Resume: reload already-scored candidates from a previous output. ---
    cached_gap: dict = {}     # cand_id -> mean relative gap (fraction, not %)
    cached_bins: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        prev_cids = prev.get("cand_id", [])
        prev_gaps = prev.get("gap_pct", [])
        prev_bins = prev.get("mean_bins", [])
        for i, cid in enumerate(prev_cids):
            g = prev_gaps[i] if i < len(prev_gaps) else None
            if cid is not None and g is not None and cid not in cached_gap:
                cached_gap[cid] = g / 100.0
                if i < len(prev_bins) and prev_bins[i] is not None:
                    cached_bins[cid] = prev_bins[i]
        print(f"Resume: {len(cached_gap)}/{len(all_unique)} unique candidates "
              f"already scored in {out_path.name}", flush=True)
    else:
        print("No existing output found — starting from scratch.", flush=True)

    pending = [h for h in all_unique if h["cand_id"] not in cached_gap]

    instances, lbs, mean_lb = load_training_instances(data_file)
    if pending:
        print(f"Scoring {len(pending)} pending heuristics "
              f"({len(cached_gap)} cached, {len(all_unique)} unique, "
              f"{n_traj_entries} traj entries, {len(instances)} instances, "
              f"{n_cores} cores)...", flush=True)
        t_score = time.time()
        gaps, bins = batch_scoring_parallel(instances, lbs, pending, n_cores=n_cores)
        for h, g, b in zip(pending, gaps, bins):
            cached_gap[h["cand_id"]] = float(g)
            cached_bins[h["cand_id"]] = float(b)
        print(f"  Scoring done in {time.time() - t_score:.1f}s", flush=True)
    else:
        print("All candidates already scored — nothing to evaluate.", flush=True)

    # Build per-trajectory-entry aligned outputs.
    gap_pct = [
        (cached_gap[t["cand_id"]] * 100.0 if t["source"] is not None
         and np.isfinite(cached_gap.get(t["cand_id"], np.inf)) else None)
        for t in traj
    ]
    mean_bins_out = [
        (cached_bins.get(t["cand_id"]) if t["source"] is not None else None)
        for t in traj
    ]
    valid = [g for g in gap_pct if g is not None]
    best_gap = min(valid) if valid else float("inf")
    n_covered = sum(1 for g in gap_pct if g is not None)
    print(f"  covered {n_covered}/{len(traj)} trajectory entries  "
          f"best gap={best_gap:.4f}%  mean_lb={mean_lb:.2f}  "
          f"total_wall={time.time() - t_start:.1f}s", flush=True)

    return {
        "data_file": str(data_file),
        "n_instances": len(instances),
        "mean_lb": mean_lb,
        "metric": "mean_relative_gap_pct (mean-of-ratios, lb=round, over all instances)",
        "mode": mode,
        "cand_id": [t["cand_id"] for t in traj],
        "used_budget": [t.get("used_budget") for t in traj],
        "gap_pct": gap_pct,
        "mean_bins": mean_bins_out,
        "score": mean_bins_out,   # raw full-suite mean bins == the race cost; pairs with
                                  # the in-race partial score in fitness_reliability_log_<rule>
        "best_gap_pct": best_gap,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Re-evaluate a whole trajectory of OBP heuristics on the "
                    "heterogeneous TRAINING set (flat mean relative gap over all "
                    "128 instances; no per-group split).")
    p.add_argument("--exp", type=str, required=True,
                   help="Experiment directory containing trajectory.json + "
                        "heuristics.json, e.g. "
                        "'.logs/racing_eoh_obp_hetero_vllm/2026-07-30/140320_2_...'")
    p.add_argument("--data-file", type=pathlib.Path, default=DEFAULT_DATA_FILE,
                   help="Training instance pickle (default: dataset_200_2k_128_5_80.pkl).")
    p.add_argument("--n-cores", type=int, default=20)
    p.add_argument("--mode", type=str, default="valid", choices=["valid", "gen"],
                   help="'valid': full trajectory; 'gen': last entry only.")
    p.add_argument("--has-incumbent", action="store_true",
                   help="If set, only evaluate candidates that were selected as incumbents in the trajectory.")
    args = p.parse_args(argv)

    exp_path = (pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute()
                else ROOT / args.exp)
    print(f"Processing experiment: {exp_path}")

    result = evaluate_trajectory(exp_path, args.data_file, args.n_cores, args.mode, args.has_incumbent)

    if args.has_incumbent:
        out_path = exp_path / f"{args.mode}_hetero_train_trajectory_incumbent.json"
    else:
        out_path = exp_path / f"{args.mode}_hetero_train_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serialisable)
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
