#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import Evaluation
import sys

# Inject the exact EoH-generated heuristic string
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
    edge_n_used_max = np.max(edge_n_used)
    decay_factor = 0.1
    mean_distance = np.mean(edge_distance)
    for i in range(edge_distance.shape[0]):
        for j in range(edge_distance.shape[1]):
            if edge_count[i][j] > 0:
                noise_factor = (np.random.uniform(0.7, 1.3) / edge_count[i][j]) + (edge_distance[i][j] / mean_distance) - (0.3 / edge_n_used_max) * edge_n_used[i][j]
                updated_edge_distance[i][j] += noise_factor * (1 + edge_count[i][j]) * decay_factor * updated_edge_distance[i][j]
    return updated_edge_distance
'''

# Strictly mirror the Evaluation class but execute from string
from prob import solve_instance

class StringEvaluation(Evaluation):
    def __init__(self, dataset, n_test, heuristic_string, time_limit=10.0, ite_max=1000, perturbation_moves=1):
        super().__init__(dataset, n_test, time_limit, ite_max, perturbation_moves)
        self.heuristic_string = heuristic_string
        
    def evaluate(self):
        # Execute the heuristic string into a local namespace
        ns = {}
        exec(self.heuristic_string, ns)
        update_edge_distance_fn = ns["update_edge_distance"]
        
        gaps = np.zeros(self.n_test)
        for i in range(self.n_test):
            gaps[i] = solve_instance(
                self.opt_costs[i], self.instances[i],
                self.time_limit, self.ite_max,
                self.perturbation_moves, update_edge_distance_fn)
        return float(np.mean(gaps))

_TSPLIB_DIR = os.path.join(os.path.dirname(__file__), '..', 'TestingData', 'TSPLib')

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

def instance_size(name):
    return int(''.join(c for c in name if c.isdigit()))

def eval_instance(args):
    name, global_seed = args
    
    # Isolate RNG across multiprocessing workers to ensure deterministic behavior per seed
    inst_idx = list(BKS.keys()).index(name)
    np.random.seed((global_seed * 1000 + inst_idx) % (2**32))
    
    dis_matrix, scale = read_tsplib(os.path.join(_TSPLIB_DIR, f'{name}.tsp'))
    opt_cost = BKS[name] / scale
    
    dataset = {'distance_matrix': [dis_matrix], 'cost': [opt_cost]}
    
    eva = StringEvaluation(dataset, 1, heuristic_src, time_limit=TIME_LIMIT, ite_max=ITE_MAX)
    t0 = time.time()
    # evaluate() will dynamically execute the heuristic_string and run it
    gap = eva.evaluate()
    dt = time.time() - t0
    
    cost = (gap / 100.0 + 1.0) * opt_cost
    
    return name, gap, float(cost * scale), dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    args = parser.parse_args()
    seed = args.seed

    names = sorted(BKS.keys(), key=instance_size)
    
    # Automatically resolve .logs directory based on script location
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    output_dir = os.path.join(ROOT_DIR, ".logs", "tsplib_outputs")
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"evaluation_results_runEval_EoH_s{seed}.json")
    
    output_json = {}
    print(f"Starting parallel evaluation over {len(names)} instances for Seed {seed}...")
    
    pool_args = [(name, seed) for name in names]
    
    with Pool(processes=N_PROC) as pool:
        for name, gap, cost, dt in pool.imap(eval_instance, pool_args):
            output_json[name] = {
                "cost": cost,
                "gap": gap,
                "dt": dt
            }
            print(f"{name:<10s}: gap = {gap:7.4f}%  time = {dt:6.1f}s")
            
            # Write out iteratively
            with open(results_file, "w") as f:
                json.dump(output_json, f, indent=4)
                
    print(f"\nSeed {seed} finished! Results saved to {results_file}")

if __name__ == '__main__':
    main()
