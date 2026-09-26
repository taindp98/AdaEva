"""Evaluate candidate checkpoints of an --exp run on the TSPLib benchmark set.

Mirrors test_tsplib.py: loads candidates from trajectory.json (or incumbents.json)
sliced at proportions [0.2, 0.4, 0.6, 0.8, 1.0] of the final budget, evaluates
each candidate across all TSPLIB instances in parallel via prob.solve_instance,
and saves the output to tsplib_slice_s{seed}.json.
"""

import sys
import os
import time
import json
import argparse
import pathlib
import numpy as np
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "utils" / "updated_tsp_eval"))

from prob import solve_instance

_TSPLIB_DIR = ROOT / "src" / "utils" / "updated_tsp_eval" / "TestingData" / "TSPLib"

TIME_LIMIT = 60.0
ITE_MAX = 1000
N_PROC = 8

# Sentinel gap (%) recorded when a candidate's source fails to compile (LLM prose / broken
# code) or crashes during GLS. Keeping it as a large finite penalty lets the run continue and
# marks the candidate as failed on that instance instead of aborting the whole pool.
PENALTY_GAP = 1e6

BKS = {
    'eil51': 426, 'berlin52': 7542, 'st70': 675, 'eil76': 538,
    'pr76': 108159, 'rat99': 1211, 'kroA100': 21282, 'kroB100': 22141,
    'kroC100': 20749, 'kroD100': 21294, 'kroE100': 22068, 'rd100': 7910,
    'eil101': 629, 'lin105': 14379, 'pr107': 44303, 'pr124': 59030,
    'bier127': 118282, 'ch130': 6110, 'pr136': 96772, 'pr144': 58537,
    'ch150': 6528, 'kroA150': 26524, 'kroB150': 26130, 'pr152': 73682,
    'u159': 42080, 'rat195': 2323, 'd198': 15780, 'kroA200': 29368,
    'kroB200': 29437,
}


def read_tsplib(path):
    with open(path) as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('NODE_COORD_SECTION')) + 1
    coords = []
    for l in lines[start:]:
        if l == 'EOF' or not l:
            break
        parts = l.strip().split()
        coords.append([float(parts[1]), float(parts[2])])
    coords = np.array(coords)
    cmin = coords.min(axis=0)
    scale = (coords.max(axis=0) - cmin).max()
    coords = (coords - cmin) / scale
    diff = coords[:, None, :] - coords[None, :, :]
    dis_matrix = np.sqrt((diff ** 2).sum(axis=-1))
    return dis_matrix, scale


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
            continue
        selected.append({
            "cand_id": cid,
            "source": source,
            "used_budget": closest["used_budget"],
            "slice": s,
        })
    return selected


def _compile_update_edge_distance(source: str) -> callable:
    ns = {"np": np}
    exec(source, ns)
    fn = ns.get("update_edge_distance")
    if fn is None:
        raise KeyError("source does not define 'update_edge_distance'")
    return fn


def eval_instance(args):
    name, global_seed, source = args

    inst_idx = list(BKS.keys()).index(name)
    np.random.seed((global_seed * 1000 + inst_idx) % (2**32))

    dis_matrix, scale = read_tsplib(_TSPLIB_DIR / f'{name}.tsp')
    opt_cost = BKS[name] / scale

    # A candidate's source can be malformed (uncompilable LLM prose) or crash during GLS.
    # Catch it and record a penalty gap so this one instance/candidate fails gracefully
    # instead of raising out of the pool worker and aborting the whole run.
    t0 = time.time()
    try:
        update_edge_distance_eoh = _compile_update_edge_distance(source)
        gap = solve_instance(opt_cost, dis_matrix, TIME_LIMIT, ITE_MAX, 1, update_edge_distance_eoh)
    except Exception as e:
        dt = time.time() - t0
        print(f"  [eval] {name}: candidate failed ({type(e).__name__}: {e}) -> penalty gap", flush=True)
        return name, PENALTY_GAP, float((PENALTY_GAP / 100.0 + 1.0) * opt_cost), dt
    dt = time.time() - t0

    cost = (gap / 100.0 + 1.0) * opt_cost
    return name, gap, float(cost), dt


def main():
    global TIME_LIMIT, ITE_MAX
    parser = argparse.ArgumentParser(
        description="Evaluate an --exp run's sliced candidate checkpoints on the TSPLib benchmark set."
    )
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment directory with trajectory.json + heuristics.json.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (e.g. passed by a SLURM array task ID).")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="Take INCUMBENT entries instead of raw trajectory entries.")
    parser.add_argument("--n-proc", type=int, default=N_PROC,
                        help="Parallel workers (one per instance).")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help="Per-instance GLS wall-clock cap (s).")
    parser.add_argument("--ite-max", type=int, default=ITE_MAX,
                        help="Per-instance GLS iteration cap.")
    parser.add_argument("--incumbent-mode", type=str, default="mean_rank",
                        choices=["mean_rank", "mean_cost"],
                        help="With --has-incumbent, which validated incumbent series to slice: "
                             "'mean_rank' -> valid_trajectory_mean_rank.json, "
                             "'mean_cost' -> valid_trajectory_mean_cost.json.")
    args = parser.parse_args()

    TIME_LIMIT = args.time_limit
    ITE_MAX = args.ite_max

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    candidates = load_gen_heuristics(exp_path, args.has_incumbent, args.incumbent_mode)
    tag = "incumbent" if args.has_incumbent else "gen"

    print(f"Loaded {len(candidates)} candidate checkpoints from {exp_path} (source={tag})")
    for c in candidates:
        print(f"  cand_id={c['cand_id']:<15s} budget={c['used_budget']:<8d} slice={c['slice']:<4.1f}")

    names = sorted(BKS.keys(), key=lambda n: int(''.join(c for c in n if c.isdigit())))
    results_file = exp_path / f"tsplib_slice_s{args.seed}.json"

    output_json = {
        "exp": str(exp_path),
        "source": tag,
        "seed": args.seed,
        "time_limit": TIME_LIMIT,
        "ite_max": ITE_MAX,
        "candidates": [],
    }

    for idx, cand in enumerate(candidates, 1):
        cand_id = cand["cand_id"]
        source = cand["source"]
        used_budget = cand["used_budget"]
        slice_val = cand["slice"]

        print()
        print(f"[{idx}/{len(candidates)}] Evaluating {cand_id} (used_budget={used_budget}, slice={slice_val}) over {len(names)} instances...")

        pool_args = [(name, args.seed, source) for name in names]
        cand_instances = {}

        with Pool(processes=args.n_proc) as pool:
            for name, gap, cost, dt in pool.imap(eval_instance, pool_args):
                cand_instances[name] = {"cost": cost, "gap": gap, "dt": dt}
                print(f"  {name:<10s}: gap = {gap:7.4f}%  time = {dt:6.1f}s")

        gaps = [v["gap"] for v in cand_instances.values()]
        cand_mean_gap = float(np.mean(gaps)) if gaps else None
        cand_median_gap = float(np.median(gaps)) if gaps else None

        cand_entry = {
            "cand_id": cand_id,
            "used_budget": used_budget,
            "slice": slice_val,
            "mean_gap": cand_mean_gap,
            "median_gap": cand_median_gap,
            "instances": cand_instances,
        }
        output_json["candidates"].append(cand_entry)

        with open(results_file, "w") as f:
            json.dump(output_json, f, indent=4)

    print()
    print(f"Seed {args.seed} finished across {len(output_json['candidates'])} candidate checkpoints!")
    for c in output_json["candidates"]:
        mg_str = f"{c['mean_gap']:.4f}%" if c['mean_gap'] is not None else "n/a"
        print(f"  slice {c['slice']:<4.1f} (budget {c['used_budget']:<7d}) cand={c['cand_id']:<15s} mean_gap={mg_str}")
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
