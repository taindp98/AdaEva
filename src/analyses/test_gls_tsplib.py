#!/usr/bin/env python3
import sys, os, pathlib, random, tempfile, uuid, time, math
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "src"))

from llm4ad.task.optimization.tsp_gls_2O.get_instance import TSPInstance
from llm4ad.task.optimization.tsp_gls_2O.evaluation import solve_without_time

TSPLIB_DIR = ROOT / "data" / "tsplib_eoh"

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


def update_edge_distance_gls(edge_distance, local_opt_tour, edge_n_used):
    """Classic GLS penalty: augment tour edges proportional to d / (1 + times_penalised)."""
    aug = edge_distance.copy()
    n = len(local_opt_tour)
    for i in range(n):
        u = int(local_opt_tour[i])
        v = int(local_opt_tour[(i + 1) % n])
        delta = edge_distance[u, v] / (1.0 + edge_n_used[u, v])
        aug[u, v] += delta
        aug[v, u] += delta
    return aug

def main():
    tsplib_instances = [
        "eil51", "berlin52", "st70", "eil76", "pr76", "rat99", 
        "kroA100", "kroB100", "kroC100", "kroD100", "kroE100", 
        "rd100", "eil101", "lin105", "pr107", "pr124", "bier127", 
        "ch130", "pr136", "pr144", "ch150", "kroA150", "kroB150", 
        "pr152", "u159", "rat195", "d198", "kroA200", "kroB200"
    ]

    cache_file = TSPLIB_DIR / "opt_costs.json"
    import json
    if not cache_file.exists():
        raise FileNotFoundError("opt_costs.json not found! Run data/download_tsplib.py first.")
        
    with open(cache_file, "r") as f:
        ground_truth_costs = json.load(f)

    instances = []
    instance_names = []
    opt_costs = {}

    print("Loading instances and cached optimal costs...")
    for i, name in enumerate(tsplib_instances):
        coords = load_tsplib_instance(name)
        
        # Carefully normalize coordinates to [0, 1]
        # We MUST preserve the aspect ratio, so we shift to 0 and divide by a global max range
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
        
        instances.append(inst)
        instance_names.append(name)
        
        opt = ground_truth_costs.get(name)
        opt_costs[i] = opt
    
    print(f"\nLoaded {len(instances)} TSPLib instances.\n")

    baselines = {
        "GLS": update_edge_distance_gls
    }
    
    output_dir = ROOT / ".logs" / "tsplib_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for baseline_name, heuristic_func in baselines.items():
        print(f"\n========================================")
        print(f"Starting evaluation for baseline: {baseline_name}")
        print(f"========================================")
        
        output_json = {}
        results_file = output_dir / f"evaluation_results_{baseline_name}.json"
        
        for i, inst in enumerate(instances):
            t0 = time.perf_counter()
            cost = solve_without_time(inst, heuristic_func, seed=0)
            dt = time.perf_counter() - t0
            
            # Express the best-known cost on the scaled instance so the gap reported is the gap to the BKS
            scaled_opt_cost = opt_costs[i] / inst._scale_factor
            gap = (cost - scaled_opt_cost) / scaled_opt_cost * 100.0
            
            output_json[instance_names[i]] = {
                "cost": float(cost),
                "gap": float(gap),
                "dt": float(dt)
            }
            
            with open(results_file, "w") as f:
                json.dump(output_json, f, indent=4)

if __name__ == "__main__":
    main()
