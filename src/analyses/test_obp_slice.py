"""Evaluate candidate checkpoints of an --exp run on the OBP benchmark suite.

Mirrors test_tsplib_slice.py: loads candidates from trajectory.json (or
incumbents.json) sliced at proportions [0.2, 0.4, 0.6, 0.8, 1.0] of the final
budget, evaluates each candidate sequentially across every setting in
PROBLEM_SIZES (instances within a setting run in parallel), and saves the
output to obp_slice_s{seed}.json.
"""

import sys
import time
import json
import random
import argparse
import pathlib
import numpy as np
import multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD")]

from racing.eoh_obp import _score_obp_inst
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from utils.obp_utils import _obp_lower_bound

N_PROC = 8

PROBLEM_SIZES = {
    "1k_C100":  {"n_instances": 25, "n_items": 1000,  "capacity": 100},
    "5k_C100":  {"n_instances": 25, "n_items": 5000,  "capacity": 100},
    "10k_C100": {"n_instances": 25, "n_items": 10000, "capacity": 100},
    "1k_C500":  {"n_instances": 25, "n_items": 1000,  "capacity": 500},
    "5k_C500":  {"n_instances": 25, "n_items": 5000,  "capacity": 500},
    "10k_C500": {"n_instances": 25, "n_items": 10000, "capacity": 500},
}


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
        f"  avg lower bound for {n_instances} instances of size {n_items} "
        f"with capacity {capacity}: {avg_lb:.4f}",
        flush=True,
    )
    return avg_lb, instances


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
            print(f"  Warning: no heuristic source found for cand_id={cid}", flush=True)
            continue
        selected.append({
            "cand_id": cid,
            "source": source,
            "used_budget": closest["used_budget"],
            "slice": s,
        })
    return selected


def _compile_priority(source: str) -> callable:
    """Compile a priority function from a source string."""
    ns = {"np": np}
    exec(source, ns)
    fn = ns.get("priority")
    if fn is None:
        raise KeyError("source does not define 'priority'")
    return fn


# Filled in the parent before each pool is forked so the (large) instance list
# and the heuristic source are inherited instead of pickled per task.
_CTX = {}


def _pool_init(instances, source):
    _CTX["instances"] = instances
    try:
        _CTX["priority"] = _compile_priority(source)
    except Exception as e:
        print(f"  [compile] error: {e}", flush=True)
        _CTX["priority"] = None


def eval_instance(idx):
    priority = _CTX["priority"]
    if priority is None:
        return idx, float("inf"), 0.0
    inst = _CTX["instances"][idx]
    t0 = time.perf_counter()
    cost = _score_obp_inst(priority, inst, 0)
    return idx, float(cost), time.perf_counter() - t0


def _to_serialisable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an --exp run's sliced candidate checkpoints on the OBP benchmark suite."
    )
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment directory with trajectory.json + heuristics.json.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for instance generation (e.g. a SLURM array task ID).")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="Take INCUMBENT entries instead of raw trajectory entries.")
    parser.add_argument("--n-proc", type=int, default=N_PROC,
                        help="Parallel workers (instances of one setting run in parallel).")
    parser.add_argument("--incumbent-mode", type=str, default="mean_cost",
                        choices=["mean_rank", "mean_cost"],
                        help="With --has-incumbent, which validated incumbent series to slice: "
                             "'mean_rank' -> valid_trajectory_mean_rank.json, "
                             "'mean_cost' -> valid_trajectory_mean_cost.json.")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    candidates = load_gen_heuristics(exp_path, args.has_incumbent, args.incumbent_mode)
    tag = "incumbent" if args.has_incumbent else "gen"

    print(f"Loaded {len(candidates)} candidate checkpoints from {exp_path} (source={tag})")
    for c in candidates:
        print(f"  cand_id={c['cand_id']:<15s} budget={c['used_budget']:<8d} slice={c['slice']:<4.1f}")

    # Build every setting's instance pool once, up front; all candidates are
    # scored on exactly the same instances.
    print()
    print(f"Preparing OBP instances for {len(PROBLEM_SIZES)} settings (seed={args.seed})...")
    settings = {}
    for name, cfg in PROBLEM_SIZES.items():
        avg_lb, instances = load_instances(
            cfg["n_instances"], cfg["n_items"], cfg["capacity"], seed=args.seed
        )
        settings[name] = {"avg_lb": avg_lb, "instances": instances, "cfg": cfg}

    results_file = exp_path / f"obp_slice_s{args.seed}.json"

    output_json = {
        "exp": str(exp_path),
        "source": tag,
        "seed": args.seed,
        "problem_sizes": PROBLEM_SIZES,
        "avg_lb": {name: s["avg_lb"] for name, s in settings.items()},
        "candidates": [],
    }

    ctx = mp.get_context("fork")

    for idx, cand in enumerate(candidates, 1):
        cand_id = cand["cand_id"]
        source = cand["source"]

        print()
        print(f"[{idx}/{len(candidates)}] Evaluating {cand_id} "
              f"(used_budget={cand['used_budget']}, slice={cand['slice']}) "
              f"over {len(settings)} settings...", flush=True)

        cand_settings = {}
        for name, s in settings.items():
            instances = s["instances"]
            avg_lb = s["avg_lb"]
            t0 = time.time()
            costs = [np.inf] * len(instances)
            with ctx.Pool(processes=args.n_proc,
                          initializer=_pool_init,
                          initargs=(instances, source)) as pool:
                for i, cost, _dt in pool.imap_unordered(eval_instance, range(len(instances))):
                    costs[i] = cost
            dt = time.time() - t0

            mean_cost = float(np.mean(costs))
            gap = (mean_cost - avg_lb) / avg_lb * 100
            cand_settings[name] = {
                "cost": mean_cost,
                "gap": gap,
                "avg_lb": avg_lb,
                "dt": dt,
                "costs": [float(c) for c in costs],
            }
            print(f"  {name:<10s}: cost = {mean_cost:9.4f}  gap = {gap:7.4f}%  time = {dt:6.1f}s",
                  flush=True)

        gaps = [v["gap"] for v in cand_settings.values()]
        cand_entry = {
            "cand_id": cand_id,
            "used_budget": cand["used_budget"],
            "slice": cand["slice"],
            "mean_gap": float(np.mean(gaps)) if gaps else None,
            "median_gap": float(np.median(gaps)) if gaps else None,
            "settings": cand_settings,
        }
        output_json["candidates"].append(cand_entry)

        with open(results_file, "w") as f:
            json.dump(output_json, f, indent=4, default=_to_serialisable)

    print()
    print(f"Seed {args.seed} finished across {len(output_json['candidates'])} candidate checkpoints!")
    for c in output_json["candidates"]:
        mg_str = f"{c['mean_gap']:.4f}%" if c["mean_gap"] is not None else "n/a"
        print(f"  slice {c['slice']:<4.1f} (budget {c['used_budget']:<7d}) "
              f"cand={c['cand_id']:<15s} mean_gap={mg_str}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
