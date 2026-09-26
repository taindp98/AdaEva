#!/usr/bin/env python3
"""Test the LLM4AD fssp_gls evaluator + instances on Taillard benchmark sets.

FSSP counterpart of ``test_eoh_tsplib.py``: injects a single ``get_matrix_and_jobs``
heuristic, runs the fssp_gls GLS engine on the Taillard test instances, and reports
the makespan gap vs. each instance's Taillard upper bound (the metric the EoH paper
uses at test time). Test sets: n20m10, n20m20, n50m10, n50m20, n100m10, n100m20
(10 instances each). Each Taillard file's header carries the upper/lower bound, so
no external BKS file is needed.
"""
import sys
import os
import pathlib
import random
import time
import json
import argparse
from multiprocessing import Pool

import numpy as np

# Resolve paths
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

# The heuristic under test (get_matrix_and_jobs signature: current_sequence, time_matrix, m, n).
heuristic_src = '''
import numpy as np

def heuristic(current_sequence, time_matrix, m, n):
    machine_subset = np.random.choice(m, max(1, int(0.3 * m)), replace=False)  # randomly select a subset of machines
    weighted_avg_execution_time = np.average(time_matrix[:, machine_subset], axis=1,
                                             weights=np.random.rand(len(machine_subset)))  # weighted average execution time
    perturb_jobs = np.argsort(weighted_avg_execution_time)[-int(0.3 * n):]  # jobs with the largest weighted average time
    new_matrix = time_matrix.copy()
    perturbation_factors = np.random.uniform(0.8, 1.2, size=(len(perturb_jobs), len(machine_subset)))  # perturbation factors
    new_matrix[perturb_jobs[:, np.newaxis], machine_subset] *= perturbation_factors  # guiding matrix
    return new_matrix, perturb_jobs
'''
_ns = {}
exec(heuristic_src, _ns)
heuristic = _ns["heuristic"]


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


# module-level GLS budget (set from CLI in main, read by workers).
# EoH FSSP-GLS setting: 60 s / instance and 1000 local-search iterations.
TIME_MAX = 60.0
ITER_MAX = 1000


def eval_instance(args):
    setname, inst, global_seed = args
    # Isolate RNG per (seed, instance) so each worker is deterministic per seed
    # (the heuristic is stochastic — np.random). Belt-and-suspenders: gls() also
    # re-seeds internally from `seed` before the run.
    np.random.seed((global_seed * 1000 + inst._id) % (2 ** 32))
    random.seed((global_seed * 1000 + inst._id) % (2 ** 32))

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


def main():
    parser = argparse.ArgumentParser(
        description="Test fssp_gls evaluator on Taillard benchmark sets (gap vs upper bound).")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed to run (e.g. passed by a SLURM array task ID).")
    parser.add_argument("--time-max", type=float, default=60.0,
                        help="Per-instance GLS wall-clock budget (s). EoH FSSP-GLS "
                             "setting = 60s per instance (incl. Taillard).")
    parser.add_argument("--iter-max", type=int, default=1000,
                        help="Per-instance local-search (swap/relocate) iteration "
                             "cap. EoH FSSP-GLS setting = 1000.")
    parser.add_argument("--n-proc", type=int, default=8)
    args = parser.parse_args()

    global TIME_MAX, ITER_MAX
    TIME_MAX, ITER_MAX = args.time_max, args.iter_max

    # Build the flat (set, instance) work list; assign a global _id for RNG isolation.
    pool_args = []
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
            pool_args.append((setname, inst, args.seed))
            gid += 1
        print(f"  {setname:>8s} ({fname}): {len(parsed)} instances  "
              f"jobs={parsed[0][0]} machines={parsed[0][1]}")
    print(f"\nLoaded {len(pool_args)} Taillard instances across {len(TEST_SETS)} sets.")
    print(f"GLS budget: time_max={TIME_MAX}s  iter_max={ITER_MAX}  | seed={args.seed}  "
          f"workers={args.n_proc}\n")

    out_dir = ROOT / ".logs" / "fssp_taillard_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / f"evaluation_results_EoH_new_gls_s{args.seed}.json"

    # setname -> {instance_id: {makespan, ub, gap, dt}}
    results = {s: {} for s in TEST_SETS}
    with Pool(processes=args.n_proc) as pool:
        for setname, iid, cost, ub, gap, dt in pool.imap_unordered(eval_instance, pool_args):
            results[setname][iid] = {"makespan": cost, "ub": ub,
                                     "gap_pct": gap, "dt": dt}
            gstr = f"{gap:7.3f}%" if gap is not None else "   NA  "
            print(f"  {setname:>8s} #{iid:<3d}: makespan={cost:10.2f}  ub={ub:<7d}  "
                  f"gap={gstr}  time={dt:6.1f}s", flush=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # Per-set mean gap summary (the EoH-paper test metric).
    print("\n=== per-set mean gap (%) ===")
    summary = {}
    for setname in TEST_SETS:
        gaps = [r["gap_pct"] for r in results[setname].values() if r["gap_pct"] is not None]
        mean_gap = float(np.mean(gaps)) if gaps else None
        summary[setname] = {"n": len(gaps), "mean_gap_pct": mean_gap}
        mg = f"{mean_gap:7.3f}%" if mean_gap is not None else "  NA "
        print(f"  {setname:>8s}: n={len(gaps):<3d} mean_gap={mg}")
    allg = [r["gap_pct"] for s in results.values() for r in s.values() if r["gap_pct"] is not None]
    overall = float(np.mean(allg)) if allg else None
    summary["OVERALL"] = {"n": len(allg), "mean_gap_pct": overall}
    print(f"  {'OVERALL':>8s}: n={len(allg):<3d} mean_gap="
          f"{overall:7.3f}%" if overall is not None else "  NA ")

    results["_summary"] = summary
    results["_config"] = {"seed": args.seed, "time_max": TIME_MAX, "iter_max": ITER_MAX,
                          "test_sets": TEST_SETS}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSeed {args.seed} finished. Saved -> {results_file}")


if __name__ == "__main__":
    main()
