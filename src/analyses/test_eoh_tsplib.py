#!/usr/bin/env python3
import sys, os, pathlib, random, tempfile, uuid, time, math
import json
import numpy as np
import argparse
from multiprocessing import Pool

# Resolve paths
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "src"))

from llm4ad.task.optimization.tsp_gls_2O.get_instance import TSPInstance
from llm4ad.task.optimization.tsp_gls_2O.evaluation import solve_without_time

TSPLIB_DIR = ROOT / "data" / "tsplib_eoh"
N_PROC = 8

heuristic_src = '''
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    edge_count = np.zeros_like(edge_distance)
    for i in range(len(local_opt_tour) - 1):
        start = local_opt_tour[i]
        end = local_opt_tour[i + 1]
        edge_count[start][end] += 1
        edge_count[end][start] += 1
        # penalize local optimal route
    edge_n_used_max = np.max(edge_n_used)

    # calculate the average edge used
    decay_factor = 0.1 # decay fastor
    mean_distance = np.mean(edge_distance)
    # calculate the average distance
    for i in range(edge_distance.shape[0]):
        for j in range(edge_distance.shape[1]):
            if edge_count[i][j] > 0:
                noise_factor = (np.random.uniform(0.7, 1.3) / edge_count[i][j]) + ( edge_distance[i][j] / mean_distance) - (0.3 / edge_n_used_max) * edge_n_used[i][j]
                # calculate a hybrid noise factor
                updated_edge_distance[i][j] += noise_factor * (1 + edge_count[i][j]) - decay_factor * updated_edge_distance[i][j]

    return updated_edge_distance
'''
ns = {}
exec(heuristic_src, ns)
update_edge_distance = ns["update_edge_distance"]

def load_tsplib_instance(instance_name):
    tsp_path = TSPLIB_DIR / f"{instance_name}.tsp"
    if not tsp_path.exists():
        raise FileNotFoundError(f"Instance '{instance_name}' not found at {tsp_path}.")
    file_content = tsp_path.read_text(encoding='utf-8')
    lines = file_content.splitlines()
    coords = []
    reading_nodes = False
    for line in lines:
        line = line.strip()
        if line == "EOF" or not line:
            if reading_nodes: break
            continue
        if line == "NODE_COORD_SECTION":
            reading_nodes = True
            continue
        if reading_nodes:
            parts = line.split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)

def eval_instance(args):
    name, inst, opt_cost, global_seed = args
    
    # Isolate RNG across multiprocessing workers to ensure deterministic behavior per seed
    np.random.seed((global_seed * 1000 + inst._id) % (2**32))
    random.seed((global_seed * 1000 + inst._id) % (2**32))
    
    t0 = time.perf_counter()
    cost = solve_without_time(inst, update_edge_distance, seed=global_seed)
    dt = time.perf_counter() - t0
    
    # Express the best-known cost on the scaled instance so the gap reported is the gap to the BKS
    scaled_opt_cost = opt_cost / inst._scale_factor if opt_cost else None
    gap = (cost - scaled_opt_cost) / scaled_opt_cost * 100.0 if scaled_opt_cost else None
    
    return name, cost, gap, dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed to run (passed by SLURM array task ID)")
    args = parser.parse_args()

    tsplib_instances = [
        "eil51", "berlin52", "st70", "eil76", "pr76", "rat99", 
        "kroA100", "kroB100", "kroC100", "kroD100", "kroE100", 
        "rd100", "eil101", "lin105", "pr107", "pr124", "bier127", 
        "ch130", "pr136", "pr144", "ch150", "kroA150", "kroB150", 
        "pr152", "u159", "rat195", "d198", "kroA200", "kroB200"
    ]

    cache_file = TSPLIB_DIR / "opt_costs.json"
    if not cache_file.exists():
        raise FileNotFoundError("opt_costs.json not found! Run data/download_tsplib.py first.")
        
    with open(cache_file, "r") as f:
        ground_truth_costs = json.load(f)

    print("Loading instances and cached optimal costs...")
    pool_args = []
    
    for i, name in enumerate(tsplib_instances):
        coords = load_tsplib_instance(name)
        
        cmin, cmax = coords.min(), coords.max()
        scale = cmax - cmin
        if scale > 0:
            normalized_coords = (coords - cmin) / scale
        else:
            normalized_coords = coords - cmin
            scale = 1.0
            
        inst = TSPInstance(normalized_coords)
        inst._id = i
        inst._scale_factor = scale  # Save the scale factor to scale the BKS down later
        
        opt = ground_truth_costs.get(name)
        if opt is not None:
            pool_args.append((name, inst, opt, args.seed))
            print(f"{name:>12s}: {coords.shape[0]:>5d} cities | opt_cost = {opt}")
        else:
            print(f"{name:>12s}: {coords.shape[0]:>5d} cities | opt_cost = MISSING")
    
    print(f"\nLoaded {len(pool_args)} TSPLib instances.\n")

    seed = args.seed
    print(f"\nRunning seed {seed}")
    
    output_dir = ROOT / ".logs" / "tsplib_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"evaluation_results_EoH_s{seed}.json"

    seed_json = {}
    print(f"\nEvaluating Seed {seed} with {N_PROC} parallel workers...")

    with Pool(processes=N_PROC) as pool:
        for name, cost, gap, dt in pool.imap(eval_instance, pool_args):
            seed_json[name] = {
                "cost": float(cost),
                "gap": float(gap),
                "dt": float(dt)
            }
            print(f"{name:<10s}: gap = {gap:7.4f}%  time = {dt:6.1f}s")
            
            # Save progressively
            with open(results_file, "w") as f:
                json.dump(seed_json, f, indent=4)

    print(f"\nSeed {seed} finished. Saved to {results_file}...")

if __name__ == "__main__":
    main()
