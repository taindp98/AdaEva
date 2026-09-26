import argparse
import json
import logging
import multiprocessing
import os
import pathlib
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

# Add packages/reevo/problems/tsp_gls to path so we can import its modules
REEVO_TSP_DIR = ROOT / "packages" / "reevo" / "problems" / "tsp_gls"
sys.path.append(str(REEVO_TSP_DIR))

# Import required modules from the TSP GLS package
from gen_inst import TSPInstance, dataset_conf, generate_dataset, load_dataset
from gls import guided_local_search

# Import the loader from test_tsplib
from src.analyses.test_tsplib import load_gen_heuristic

N_PROC = 8

# Constants defined in test.ipynb
perturbation_moves_map = {
    20: 5,
    50: 30,
    100: 40,
    200: 40,
}
iter_limit_map = {
    20: 73,
    50: 175,
    100: 1800,
    200: 800,
}

# Pre-computed optimal solutions from test.ipynb
optimal_objs_dict = {
    20: 3.8362853943492015,
    50: 5.68457994395107,
    100: 7.778580370400294,
    200: 10.71194600194464
}


def _compile_heuristics(source: str):
    ns = {"np": np}
    exec(source, ns)
    fn = ns.get("heuristics") or ns.get("heuristics_v2") or ns.get("heuristics_reevo")
    if fn is None:
        # Fallback if there's only one callable
        callables = [v for k, v in ns.items() if callable(v) and not k.startswith("_") and k != "np"]
        if callables:
            fn = callables[-1]
        else:
            raise KeyError("source does not define 'heuristics' function")
    return fn


def calculate_cost(inst: TSPInstance, path: np.ndarray):
    return inst.distmat[path, np.roll(path, 1)].sum().item()


def eval_instance(args):
    """Worker function for multiprocessing pool."""
    inst_idx, inst, source, global_seed = args
    
    # Inherit seed for deterministic evaluation if heuristic uses random
    np.random.seed((global_seed * 1000 + inst_idx) % (2**32))
    
    heuristics_fn = _compile_heuristics(source)
    
    start_time = time.time()
    start_cpu = time.process_time()
    heu = heuristics_fn(inst.distmat.copy())
    
    # guided_local_search is an imported cython function
    result = guided_local_search(
        inst.distmat, 
        heu, 
        perturbation_moves_map[inst.n], 
        iter_limit_map[inst.n]
    )
    
    duration = time.time() - start_time
    cpu_duration = time.process_time() - start_cpu
    cost = calculate_cost(inst, result)
    
    return inst_idx, cost, duration, cpu_duration


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an --exp run's final heuristic on synthetic TSP instances (via ReEvo test.ipynb)."
    )
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment directory with trajectory.json + heuristics.json.")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="Take the last INCUMBENT entry instead of the last trajectory entry.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (e.g. passed by a SLURM array task ID).")
    parser.add_argument("--n-proc", type=int, default=N_PROC,
                        help="Parallel workers.")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    cand_id, source = load_gen_heuristic(exp_path, args.has_incumbent)
    tag = "incumbent" if args.has_incumbent else "gen"
    print(f"[*] Loaded final heuristic from {exp_path}")
    print(f"[*]   source={tag}  cand_id={cand_id}  ({len(source)} chars)")
    print()

    results_file = exp_path / f"synth_tsp_s{args.seed}.json"
    output_json = {
        "exp": str(exp_path),
        "source": tag,
        "cand_id": cand_id,
        "seed": args.seed,
        "results": {}
    }

    # Generate test datasets in memory/temp dynamically, matching gen_inst.py seed rules exactly.
    # The original script does: 
    # np.random.seed(len(mood))
    # for n in problem_sizes: ...
    # We will replicate this for the 'test' mood locally to ensure perfectly matching instances.
    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(len('test'))
        for n in dataset_conf['test']:
            filepath = os.path.join(tmpdir, f"test{n}_dataset.npy")
            generate_dataset(filepath, n, batch_size=64)
            
        print(f"[*] Function: {tag}_{cand_id} \n")
        
        for problem_size in iter_limit_map.keys():
            dataset_path = os.path.join(tmpdir, f"test{problem_size}_dataset.npy")
            dataset = load_dataset(dataset_path)
            logging.info(f"[*] Evaluating {dataset_path}")
            
            pool_args = [(i, inst, source, args.seed) for i, inst in enumerate(dataset)]
            
            objs = [None] * len(dataset)
            durations = [None] * len(dataset)
            cpu_durations = [None] * len(dataset)
            
            output_json["results"][str(problem_size)] = {
                "instances": {},
                "mean_gap": None,
                "mean_cost": None,
                "mean_time": None,
                "mean_cpu_time": None,
            }
            
            # Using multiprocessing to speed up
            with Pool(processes=args.n_proc) as pool:
                # We use tqdm to monitor progress just like the notebook
                for inst_idx, cost, duration, cpu_duration in tqdm(pool.imap_unordered(eval_instance, pool_args), total=len(dataset), desc=f"tsp{problem_size}"):
                    objs[inst_idx] = cost
                    durations[inst_idx] = duration
                    cpu_durations[inst_idx] = cpu_duration
                    
                    # Log instance level metrics
                    output_json["results"][str(problem_size)]["instances"][str(inst_idx)] = {
                        "cost": cost,
                        "dt": duration,
                        "cpu_dt": cpu_duration,
                    }
                    
                    # Save progressively
                    with open(results_file, "w") as f:
                        json.dump(output_json, f, indent=4)
                    
            mean_obj = np.mean(objs).item()
            mean_optimal_obj = optimal_objs_dict[problem_size]
            gap = mean_obj / mean_optimal_obj - 1
            mean_time = np.mean(durations).item()
            mean_cpu_time = np.mean(cpu_durations).item()
            
            output_json["results"][str(problem_size)]["mean_cost"] = mean_obj
            output_json["results"][str(problem_size)]["mean_gap"] = gap
            output_json["results"][str(problem_size)]["mean_time"] = mean_time
            output_json["results"][str(problem_size)]["mean_cpu_time"] = mean_cpu_time
            
            # Final save per problem size
            with open(results_file, "w") as f:
                json.dump(output_json, f, indent=4)
            
            print(f"[*] Average for {problem_size}: {mean_obj:.6f} ({mean_optimal_obj:.6f})")
            print(f"[*] Optimality gap: {gap*100:.6f}%")
            print(f"[*] Total/Average duration: {sum(durations):.6f}s {sum(durations)/len(durations):.6f}s")
            print(f"[*] Total/Average CPU time: {sum(cpu_durations):.6f}s {sum(cpu_durations)/len(cpu_durations):.6f}s")
            print()
            
    print(f"\n[*] Finished! Results saved to {results_file}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Make sure multiprocessing works properly with our imported modules
    multiprocessing.set_start_method("fork", force=True)
    main()
