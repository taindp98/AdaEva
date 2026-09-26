#!/usr/bin/env python3
"""Evaluate a search run's FINAL heuristic on the Taillard FSSP benchmark sets.

FSSP counterpart of ``test_tsplib.py``: keeps the Taillard evaluation machinery of
``test_eoh_fssp_taillard.py`` (the fixed Taillard test sets, the fssp_gls GLS
engine, the makespan gap vs. each instance's Taillard upper bound, parallel over
instances, per-set + overall gap summary), but instead of a HARDCODED heuristic it
loads the heuristic from an experiment directory ``--exp`` — the same way
``eval_tsp.py`` / ``test_tsplib.py`` do:

  * read ``<exp>/trajectory.json`` and take the LAST entry (``traj = [traj[-1]]``,
    the final best-so-far) of the ``trajectory`` list (default) or the
    ``incumbents`` list (``--has-incumbent``);
  * resolve that entry's ``cand_id`` to its ``source`` via ``<exp>/heuristics.json``;
  * use that source's ``get_matrix_and_jobs`` as the GLS perturbation heuristic.

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
# Load the FINAL-trajectory heuristic from an --exp dir (mirrors eval_tsp.py)
# --------------------------------------------------------------------------- #

def load_gen_heuristic(exp_path: pathlib.Path, has_incumbent: bool = False,
                       incumbent_mode: str = "mean_rank") -> tuple:
    """Return ``(cand_id, source)`` of the run's final heuristic.

    Mirrors ``eval_tsp.py`` / ``test_tsplib.py``: take the LAST entry of the
    ``trajectory`` list (default) or, with ``has_incumbent``, the FINAL validated
    incumbent from ``valid_trajectory_<incumbent_mode>.json`` (``incumbent_mode`` in
    {``mean_rank``, ``mean_cost``}), then resolve its ``cand_id`` to a ``source`` via
    ``heuristics.json``."""
    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not heuristic_path.exists():
        raise FileNotFoundError(f"Missing: {heuristic_path}")

    if has_incumbent:
        vt_path = exp_path / f"valid_trajectory_{incumbent_mode}.json"
        if not vt_path.exists():
            raise FileNotFoundError(
                f"Missing: {vt_path}; run the racing validation (_final_eval) to emit it.")
        with open(vt_path) as f:
            vt = json.load(f)
        cids = vt.get("cand_id")
        if not isinstance(cids, list) or not cids:
            raise KeyError(f"{vt_path} has no non-empty 'cand_id' list")
        cand_id = cids[-1]               # the final validated incumbent
    else:
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Missing: {trajectory_path}")
        with open(trajectory_path) as f:
            traj_data = json.load(f)
        lst = traj_data.get("trajectory")
        if not isinstance(lst, list) or not lst:
            raise KeyError(
                f"trajectory.json at {trajectory_path} has no non-empty 'trajectory' list; "
                f"cannot pick the final heuristic."
            )
        cand_id = lst[-1]["cand_id"]     # traj = [traj[-1]]: the final best-so-far

    with open(heuristic_path) as f:
        heuristics_data = json.load(f)
    src_by_id = {h["cand_id"]: h["source"] for h in heuristics_data["heuristics"]}
    source = src_by_id.get(cand_id)
    if source is None:
        raise KeyError(
            f"no heuristic source for cand_id={cand_id!r} in {heuristic_path}"
        )
    return cand_id, source


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
    heuristic = _compile_get_matrix_and_jobs(source)

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
    # Declared up-front: reassigned below (after arg parsing) so the forked pool
    # workers inherit the CLI-resolved caps.
    global TIME_MAX, ITER_MAX
    parser = argparse.ArgumentParser(
        description="Evaluate an --exp run's final heuristic on the Taillard FSSP sets "
                    "(makespan gap vs. upper bound).")
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment directory with trajectory.json + heuristics.json.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (e.g. passed by a SLURM array task ID).")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="Take the last INCUMBENT entry instead of the last trajectory entry.")
    parser.add_argument("--incumbent-mode", type=str, default="mean_rank",
                        choices=["mean_rank", "mean_cost"],
                        help="With --has-incumbent, which validated incumbent series to load: "
                             "'mean_rank' -> valid_trajectory_mean_rank.json, "
                             "'mean_cost' -> valid_trajectory_mean_cost.json.")
    parser.add_argument("--time-max", type=float, default=60.0,
                        help="Per-instance GLS wall-clock budget (s). EoH FSSP-GLS setting = 60s.")
    parser.add_argument("--iter-max", type=int, default=1000,
                        help="Per-instance local-search iteration cap. EoH FSSP-GLS setting = 1000.")
    parser.add_argument("--n-proc", type=int, default=N_PROC)
    args = parser.parse_args()

    TIME_MAX, ITER_MAX = args.time_max, args.iter_max

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    cand_id, source = load_gen_heuristic(exp_path, args.has_incumbent, args.incumbent_mode)
    tag = "incumbent" if args.has_incumbent else "gen"
    print(f"Loaded final heuristic from {exp_path}")
    print(f"  source={tag}  cand_id={cand_id}  ({len(source)} chars)")

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
            pool_args.append((setname, inst, args.seed, source))
            gid += 1
        print(f"  {setname:>8s} ({fname}): {len(parsed)} instances  "
              f"jobs={parsed[0][0]} machines={parsed[0][1]}")
    print(f"\nLoaded {len(pool_args)} Taillard instances across {len(TEST_SETS)} sets.")
    print(f"GLS budget: time_max={TIME_MAX}s  iter_max={ITER_MAX}  | seed={args.seed}  "
          f"workers={args.n_proc}\n")

    results_file = exp_path / f"fssp_taillard_s{args.seed}.json"

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
    overall_str = f"{overall:7.3f}%" if overall is not None else "  NA "
    print(f"  {'OVERALL':>8s}: n={len(allg):<3d} mean_gap={overall_str}")

    results["_summary"] = summary
    results["_config"] = {"exp": str(exp_path), "source": tag, "cand_id": cand_id,
                          "seed": args.seed, "time_max": TIME_MAX, "iter_max": ITER_MAX,
                          "test_sets": TEST_SETS}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSeed {args.seed} finished. Saved -> {results_file}")


if __name__ == "__main__":
    main()
