#!/usr/bin/env python3
"""Evaluate a search run's FINAL heuristic on the TSPLib benchmark set.

Same TSPLib evaluation machinery as ``test_eoh_tsplib_new.py`` (the fixed BKS
instance set, per-instance GLS via ``prob.solve_instance``, parallel over
instances, per-seed output), but instead of a HARDCODED ``heuristic_src`` it
loads the heuristic from an experiment directory ``--exp`` — exactly the way
``eval_tsp.py`` does:

  * read ``<exp>/trajectory.json`` and take the LAST entry (``traj = [traj[-1]]``,
    the final best-so-far incumbent) of either the ``trajectory`` list (default)
    or the ``incumbents`` list (``--has-incumbent``);
  * resolve that entry's ``cand_id`` to its ``source`` via
    ``<exp>/heuristics.json``;
  * use that source as ``update_edge_distance``.

Nothing else (BKS, scaling, RNG seeding, GLS caps, output format) changes, so a
TSPLib run of a search-produced heuristic is directly comparable to the reference
script's runs.  This module does not modify any existing source.
"""

import sys
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

# GLS caps + parallelism (overridable on the CLI). Kept as module globals so the
# forked pool workers inherit the CLI-resolved values (Pool is created AFTER main
# reassigns them).
TIME_LIMIT = 60.0
ITE_MAX = 1000
N_PROC = 8

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
    for line in lines[start:]:
        parts = line.split()
        if not parts or parts[0] == 'EOF':
            break
        coords.append((float(parts[1]), float(parts[2])))
    coords = np.array(coords)

    cmin, cmax = coords.min(), coords.max()
    scale = cmax - cmin
    coords = (coords - cmin) / scale

    diff = coords[:, None, :] - coords[None, :, :]
    dis_matrix = np.sqrt((diff ** 2).sum(axis=-1))
    return dis_matrix, scale


# --------------------------------------------------------------------------- #
# Load the FINAL-trajectory heuristic from an --exp dir (mirrors eval_tsp.py)
# --------------------------------------------------------------------------- #

def load_gen_heuristic(exp_path: pathlib.Path, has_incumbent: bool = False,
                       incumbent_mode: str = "mean_rank") -> tuple:
    """Return ``(cand_id, source)`` of the run's final heuristic.

    Mirrors ``eval_tsp.py``: take the LAST entry of the ``trajectory`` list
    (default) or, with ``has_incumbent``, the FINAL validated incumbent from
    ``valid_trajectory_<incumbent_mode>.json`` (``incumbent_mode`` in {``mean_rank``,
    ``mean_cost``}), then resolve its ``cand_id`` to a ``source`` via
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


def _compile_update_edge_distance(source: str) -> callable:
    ns = {"np": np}
    exec(source, ns)
    fn = ns.get("update_edge_distance")
    if fn is None:
        raise KeyError("source does not define 'update_edge_distance'")
    return fn


def eval_instance(args):
    name, global_seed, source = args

    # Very important: When using multiprocessing and np.random in the heuristic,
    # the subprocesses inherit the exact same RNG state from the fork!
    # To ensure each instance gets a properly isolated but deterministic RNG stream,
    # we seed it uniquely using a combination of the SLURM seed and the instance name.
    inst_idx = list(BKS.keys()).index(name)
    np.random.seed((global_seed * 1000 + inst_idx) % (2**32))

    # Compile the --exp heuristic inside the worker (source is picklable; the
    # compiled fn is not — so we pass the string and exec it here).
    update_edge_distance_eoh = _compile_update_edge_distance(source)

    dis_matrix, scale = read_tsplib(_TSPLIB_DIR / f'{name}.tsp')
    opt_cost = BKS[name] / scale

    t0 = time.time()
    # solve_instance from prob.py returns the gap directly.
    gap = solve_instance(opt_cost, dis_matrix, TIME_LIMIT, ITE_MAX, 1, update_edge_distance_eoh)
    dt = time.time() - t0

    # Reconstruct the scaled cost from the gap percentage
    cost = (gap / 100.0 + 1.0) * opt_cost

    return name, gap, float(cost), dt


def main():
    # Declared up-front: reassigned below (after arg parsing) so the forked pool
    # workers inherit the CLI-resolved caps.
    global TIME_LIMIT, ITE_MAX
    parser = argparse.ArgumentParser(
        description="Evaluate an --exp run's final heuristic on the TSPLib benchmark set."
    )
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
    parser.add_argument("--n-proc", type=int, default=N_PROC,
                        help="Parallel workers (one per instance).")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help="Per-instance GLS wall-clock cap (s).")
    parser.add_argument("--ite-max", type=int, default=ITE_MAX,
                        help="Per-instance GLS iteration cap.")
    args = parser.parse_args()

    # Propagate CLI caps to the module globals BEFORE the pool forks so workers
    # inherit them.
    TIME_LIMIT = args.time_limit
    ITE_MAX = args.ite_max

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    cand_id, source = load_gen_heuristic(exp_path, args.has_incumbent, args.incumbent_mode)
    tag = "incumbent" if args.has_incumbent else "gen"
    print(f"Loaded final heuristic from {exp_path}")
    print(f"  source={tag}  cand_id={cand_id}  ({len(source)} chars)")

    names = sorted(BKS.keys(), key=lambda n: int(''.join(c for c in n if c.isdigit())))

    results_file = exp_path / f"tsplib_s{args.seed}.json"
    output_json = {
        "exp": str(exp_path),
        "source": tag,
        "cand_id": cand_id,
        "seed": args.seed,
        "time_limit": TIME_LIMIT,
        "ite_max": ITE_MAX,
        "instances": {},
    }
    print(f"Starting parallel evaluation over {len(names)} instances for seed {args.seed} "
          f"({args.n_proc} procs)...")

    pool_args = [(name, args.seed, source) for name in names]

    with Pool(processes=args.n_proc) as pool:
        for name, gap, cost, dt in pool.imap(eval_instance, pool_args):
            output_json["instances"][name] = {"cost": cost, "gap": gap, "dt": dt}
            print(f"{name:<10s}: gap = {gap:7.4f}%  time = {dt:6.1f}s")

            # Save progressively in case of premature termination.
            with open(results_file, "w") as f:
                json.dump(output_json, f, indent=4)

    gaps = [v["gap"] for v in output_json["instances"].values()]
    output_json["mean_gap"] = float(np.mean(gaps)) if gaps else None
    output_json["median_gap"] = float(np.median(gaps)) if gaps else None
    with open(results_file, "w") as f:
        json.dump(output_json, f, indent=4)

    mg = output_json["mean_gap"]
    mg_str = f"{mg:.4f}%" if mg is not None else "n/a"
    print(f"\nSeed {args.seed} finished! mean gap = {mg_str}  "
          f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
