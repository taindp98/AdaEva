#!/usr/bin/env python3
"""Evaluate candidate checkpoints of an --exp run on the Taillard FSSP benchmark sets.

FSSP counterpart of ``test_tsplib_slice.py``: keeps the Taillard evaluation
machinery of ``test_fssp_taillard.py`` (the fixed Taillard test sets, the
fssp_gls GLS engine, the makespan gap vs. each instance's Taillard upper bound,
parallel over instances, per-set + overall gap summary), but instead of the
run's FINAL heuristic it loads the candidates from ``trajectory.json`` (or
``incumbents.json``) sliced at proportions [0.2, 0.4, 0.6, 0.8, 1.0] of the
final budget, evaluates each one over the full Taillard suite, and saves the
output to ``fssp_taillard_slice_s{seed}.json``.

This module does not modify any existing source.
"""
import sys
import random
import time
import json
import argparse
import pathlib
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "src"))

from llm4ad.task.optimization.fssp_gls.get_instance import FSSPInstance
from llm4ad.task.optimization.fssp_gls.gls import gls

TAILLARD_DIR = (ROOT / "packages" / "LLM4AD" / "llm4ad" / "task" / "optimization"
                / "fssp_gls" / "TestingData" / "Taillard")

# jobs x machines -> Taillard filename (10 instances per file)
TEST_SETS = {
    "n20m5":   "t_j20_m5.txt",
    "n20m10":  "t_j20_m10.txt",
    "n20m20":  "t_j20_m20.txt",
    "n50m5":   "t_j50_m5.txt",
    "n50m10":  "t_j50_m10.txt",
    "n50m20":  "t_j50_m20.txt",
    "n100m5":  "t_j100_m5.txt",
    "n100m10": "t_j100_m10.txt",
    "n100m20": "t_j100_m20.txt",
    "n200m10": "t_j200_m10.txt",
    "n200m20": "t_j200_m20.txt",
}

# module-level GLS budget (set from CLI in main, read by the forked workers).
# EoH FSSP-GLS setting: 60 s / instance and 1000 local-search iterations.
TIME_MAX = 60.0
ITER_MAX = 1000
N_PROC = 8


# --------------------------------------------------------------------------- #
# Load the sliced-budget candidates from an --exp dir (mirrors test_tsplib_slice.py)
# --------------------------------------------------------------------------- #

def _load_slice_entries(exp_path, trajectory_path, has_incumbent, incumbent_mode):
    """Return the ``[{'cand_id','used_budget'}, ...]`` list to slice: the raw
    ``trajectory`` list (default) or, with ``has_incumbent``, the aligned
    ``cand_id``/``used_budget`` series from ``valid_trajectory_<incumbent_mode>.json``
    (``incumbent_mode`` in {``mean_rank``, ``mean_cost``})."""
    if has_incumbent:
        vt_path = exp_path / f"valid_trajectory_{incumbent_mode}.json"
        if not vt_path.exists():
            raise FileNotFoundError(
                f"Missing: {vt_path}; run the racing validation (_final_eval) to emit it.")
        with open(vt_path) as f:
            vt = json.load(f)
        cids = vt.get("cand_id")
        ubs = vt.get("used_budget")
        if not isinstance(cids, list) or not cids:
            raise KeyError(f"{vt_path} has no non-empty 'cand_id' list")
        if not isinstance(ubs, list) or len(ubs) != len(cids):
            raise KeyError(f"{vt_path} has no 'used_budget' list aligned to 'cand_id'")
        return [{"cand_id": c, "used_budget": u} for c, u in zip(cids, ubs)]
    with open(trajectory_path) as f:
        traj_data = json.load(f)
    lst = traj_data.get("trajectory")
    if not isinstance(lst, list) or not lst:
        raise KeyError(
            f"trajectory.json at {trajectory_path} has no non-empty 'trajectory' list.")
    return lst


def load_gen_heuristics(exp_path: pathlib.Path, has_incumbent: bool = False,
                        incumbent_mode: str = "mean_rank") -> list:
    """Return a list of dicts [{'cand_id': ..., 'source': ..., 'used_budget': ..., 'slice': ...}]
    for sliced budget checkpoints [0.2, 0.4, 0.6, 0.8, 1.0]."""
    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing: {trajectory_path}")
    if not heuristic_path.exists():
        raise FileNotFoundError(f"Missing: {heuristic_path}")

    lst = _load_slice_entries(exp_path, trajectory_path, has_incumbent, incumbent_mode)

    final_budget = lst[-1]["used_budget"]
    slices = [0.2, 0.4, 0.6, 0.8, 1.0]
    target_budgets = [s * final_budget for s in slices]

    with open(heuristic_path) as f:
        heuristics_data = json.load(f)
    src_by_id = {h["cand_id"]: h["source"] for h in heuristics_data["heuristics"]}

    selected = []
    for s, tb in zip(slices, target_budgets):
        closest = min(lst, key=lambda entry: abs(entry["used_budget"] - tb))
        cid = closest["cand_id"]
        source = src_by_id.get(cid)
        if source is None:
            print(f"  Warning: no heuristic source for cand_id={cid!r} in {heuristic_path}")
            continue
        selected.append({
            "cand_id": cid,
            "source": source,
            "used_budget": closest["used_budget"],
            "slice": s,
        })
    return selected


def _compile_get_matrix_and_jobs(source: str) -> callable:
    """Compile the FSSP perturbation heuristic. Prefers ``get_matrix_and_jobs``
    (the EoH FSSP entry point), else falls back to the first top-level callable
    (mirrors ``eval_fssp.py``)."""
    ns = {"np": np}
    exec(source, ns)
    fn = ns.get("get_matrix_and_jobs")
    if fn is None:
        cands = [v for k, v in ns.items()
                 if callable(v) and not k.startswith("_") and k != "np"]
        if not cands:
            raise ValueError("no 'get_matrix_and_jobs' (or any callable) defined in source")
        fn = cands[0]
    return fn


def read_taillard(path):
    """Parse a Taillard flow-shop file into a list of (n_jobs, n_machines, tasks, ub).

    Header line: "number of jobs, number of machines, initial seed, upper bound and
    lower bound :" followed by "<n> <m> <seed> <UB> <LB>", a "processing times :"
    line, then n_machines rows (machine-major) of n_jobs times; we transpose to
    tasks[job][machine]. UB is taken from the header (the reference for the gap)."""
    lines = pathlib.Path(path).read_text().splitlines()
    instances = []
    i = 0
    while i < len(lines):
        if "number of jobs" in lines[i]:
            vals = lines[i + 1].split()
            n_jobs, n_machines = int(vals[0]), int(vals[1])
            ub = int(vals[3])                      # Taillard upper bound (best-known)
            rows = [[int(v) for v in lines[i + 3 + k].split()] for k in range(n_machines)]
            tasks = np.array(rows, dtype=float).T  # -> (n_jobs, n_machines)
            instances.append((n_jobs, n_machines, tasks, ub))
            i += 3 + n_machines
        else:
            i += 1
    return instances


def eval_instance(args):
    setname, inst, global_seed, source = args
    # Isolate RNG per (seed, instance) so each worker is deterministic per seed
    # (the heuristic is stochastic — np.random). Belt-and-suspenders: gls() also
    # re-seeds internally from `seed` before the run.
    np.random.seed((global_seed * 1000 + inst._id) % (2 ** 32))
    random.seed((global_seed * 1000 + inst._id) % (2 ** 32))

    # Compile the --exp heuristic inside the worker (source is picklable; the
    # compiled fn is not — so we pass the string and exec it here).
    try:
        heuristic = _compile_get_matrix_and_jobs(source)
    except Exception as e:
        print(f"    [compile] {setname}#{inst._id}: {type(e).__name__}: {e}", flush=True)
        return setname, inst._id, float("inf"), inst._ub, None, 0.0

    t0 = time.perf_counter()
    try:
        cost = gls(inst.tasks_val, inst.tasks, inst.machines_val,
                   TIME_MAX, ITER_MAX, heuristic, seed=global_seed)
        cost = float(cost)
    except Exception as e:
        print(f"    [eval] {setname}#{inst._id} crashed: {type(e).__name__}: {e}", flush=True)
        cost = float("inf")
    dt = time.perf_counter() - t0

    ub = inst._ub
    gap = (cost - ub) / ub * 100.0 if (ub and np.isfinite(cost)) else None
    return setname, inst._id, cost, ub, gap, dt


def load_taillard_instances():
    """Build the flat [(setname, FSSPInstance)] work list once; each instance gets
    a global _id (RNG isolation) and its Taillard upper bound _ub."""
    instances = []
    gid = 0
    print(f"Loading Taillard test sets from {TAILLARD_DIR}")
    for setname, fname in TEST_SETS.items():
        path = TAILLARD_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Taillard file '{fname}' not found at {path}")
        parsed = read_taillard(path)
        for (n_jobs, n_machines, tasks, ub) in parsed:
            inst = FSSPInstance(tasks)
            inst._id = gid
            inst._ub = ub
            inst._set = setname
            instances.append((setname, inst))
            gid += 1
        print(f"  {setname:>8s} ({fname}): {len(parsed)} instances  "
              f"jobs={parsed[0][0]} machines={parsed[0][1]}")
    return instances


def main():
    # Declared up-front: reassigned below (after arg parsing) so the forked pool
    # workers inherit the CLI-resolved caps.
    global TIME_MAX, ITER_MAX
    parser = argparse.ArgumentParser(
        description="Evaluate an --exp run's sliced candidate checkpoints on the Taillard "
                    "FSSP sets (makespan gap vs. upper bound).")
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment directory with trajectory.json + heuristics.json.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (e.g. passed by a SLURM array task ID).")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="Take INCUMBENT entries instead of raw trajectory entries.")
    parser.add_argument("--time-max", type=float, default=60.0,
                        help="Per-instance GLS wall-clock budget (s). EoH FSSP-GLS setting = 60s.")
    parser.add_argument("--iter-max", type=int, default=1000,
                        help="Per-instance local-search iteration cap. EoH FSSP-GLS setting = 1000.")
    parser.add_argument("--n-proc", type=int, default=N_PROC)
    parser.add_argument("--incumbent-mode", type=str, default="mean_rank",
                        choices=["mean_rank", "mean_cost"],
                        help="With --has-incumbent, which validated incumbent series to slice: "
                             "'mean_rank' -> valid_trajectory_mean_rank.json, "
                             "'mean_cost' -> valid_trajectory_mean_cost.json.")
    args = parser.parse_args()

    TIME_MAX, ITER_MAX = args.time_max, args.iter_max

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    candidates = load_gen_heuristics(exp_path, args.has_incumbent, args.incumbent_mode)
    tag = "incumbent" if args.has_incumbent else "gen"

    print(f"Loaded {len(candidates)} candidate checkpoints from {exp_path} (source={tag})")
    for c in candidates:
        print(f"  cand_id={c['cand_id']:<15s} budget={c['used_budget']:<8d} slice={c['slice']:<4.1f}")
    print()

    # The same Taillard instances are reused for every candidate.
    instances = load_taillard_instances()
    print(f"\nLoaded {len(instances)} Taillard instances across {len(TEST_SETS)} sets.")
    print(f"GLS budget: time_max={TIME_MAX}s  iter_max={ITER_MAX}  | seed={args.seed}  "
          f"workers={args.n_proc}")

    results_file = exp_path / f"fssp_taillard_slice_s{args.seed}.json"

    output_json = {
        "exp": str(exp_path),
        "source": tag,
        "seed": args.seed,
        "time_max": TIME_MAX,
        "iter_max": ITER_MAX,
        "test_sets": TEST_SETS,
        "candidates": [],
    }

    for idx, cand in enumerate(candidates, 1):
        cand_id = cand["cand_id"]
        source = cand["source"]

        print()
        print(f"[{idx}/{len(candidates)}] Evaluating {cand_id} "
              f"(used_budget={cand['used_budget']}, slice={cand['slice']}) "
              f"over {len(instances)} instances...", flush=True)

        pool_args = [(setname, inst, args.seed, source) for setname, inst in instances]
        # setname -> {instance_id: {makespan, ub, gap_pct, dt}}
        cand_sets = {s: {} for s in TEST_SETS}

        with Pool(processes=args.n_proc) as pool:
            for setname, iid, cost, ub, gap, dt in pool.imap_unordered(eval_instance, pool_args):
                cand_sets[setname][iid] = {"makespan": cost, "ub": ub,
                                           "gap_pct": gap, "dt": dt}
                gstr = f"{gap:7.3f}%" if gap is not None else "   NA  "
                print(f"  {setname:>8s} #{iid:<3d}: makespan={cost:10.2f}  ub={ub:<7d}  "
                      f"gap={gstr}  time={dt:6.1f}s", flush=True)

        # Per-set mean gap summary (the EoH-paper test metric).
        print(f"  --- per-set mean gap (%) for {cand_id} ---")
        set_summary = {}
        for setname in TEST_SETS:
            gaps = [r["gap_pct"] for r in cand_sets[setname].values() if r["gap_pct"] is not None]
            set_mean = float(np.mean(gaps)) if gaps else None
            set_summary[setname] = {
                "n": len(gaps),
                "mean_gap": set_mean,
                "median_gap": float(np.median(gaps)) if gaps else None,
            }
            mg = f"{set_mean:7.3f}%" if set_mean is not None else "  NA "
            print(f"    {setname:>8s}: n={len(gaps):<3d} mean_gap={mg}")

        allg = [r["gap_pct"] for s in cand_sets.values()
                for r in s.values() if r["gap_pct"] is not None]
        cand_mean_gap = float(np.mean(allg)) if allg else None
        cand_median_gap = float(np.median(allg)) if allg else None
        omg = f"{cand_mean_gap:7.3f}%" if cand_mean_gap is not None else "  NA "
        print(f"    {'OVERALL':>8s}: n={len(allg):<3d} mean_gap={omg}")

        cand_entry = {
            "cand_id": cand_id,
            "used_budget": cand["used_budget"],
            "slice": cand["slice"],
            "mean_gap": cand_mean_gap,
            "median_gap": cand_median_gap,
            "n": len(allg),
            "sets": set_summary,
            "instances": cand_sets,
        }
        output_json["candidates"].append(cand_entry)

        with open(results_file, "w") as f:
            json.dump(output_json, f, indent=4)

    print()
    print(f"Seed {args.seed} finished across {len(output_json['candidates'])} candidate checkpoints!")
    for c in output_json["candidates"]:
        mg_str = f"{c['mean_gap']:.4f}%" if c["mean_gap"] is not None else "n/a"
        print(f"  slice {c['slice']:<4.1f} (budget {c['used_budget']:<7d}) "
              f"cand={c['cand_id']:<15s} mean_gap={mg_str}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
