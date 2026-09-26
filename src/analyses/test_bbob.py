import sys
import time
import json
import argparse
import pathlib
import numpy as np
import multiprocessing
import re
import os

from ioh import get_problem, logger, ProblemClass
import iohinspector
import polars as pl

def load_gen_heuristic(exp_path, has_incumbent, incumbent_mode="mean_rank"):
    """Load the FINAL best-so-far candidate from a SINGLE experiment directory: the
    last entry of the ``trajectory`` list (default) or, with ``--has-incumbent``,
    the FINAL validated incumbent from ``valid_trajectory_<incumbent_mode>.json``
    (``incumbent_mode`` in {``mean_rank``, ``mean_cost``}). Its source is resolved
    from heuristics.json. Returns ``(exp_name, cand_id, source, fitness)``.

    (One --exp per invocation: pass more experiments as more lines in the .txt task
    file and compare their AUCs by hand — this avoids auto-picking a single 'best'
    before evaluation.)"""
    exp_path = pathlib.Path(exp_path)
    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not heuristic_path.exists():
        raise FileNotFoundError(f"{exp_path} is missing heuristics.json")

    if has_incumbent:
        vt_path = exp_path / f"valid_trajectory_{incumbent_mode}.json"
        if not vt_path.exists():
            raise FileNotFoundError(
                f"{exp_path} is missing {vt_path.name}; run the racing validation "
                f"(_final_eval) to emit it.")
        with open(vt_path) as f:
            vt = json.load(f)
        cids = vt.get("cand_id")
        if not isinstance(cids, list) or not cids:
            raise KeyError(f"{vt_path} has no non-empty 'cand_id' list")
        i = len(cids) - 1
        cand_id = cids[i]
        scores = vt.get("score") or []
        fitness = scores[i] if i < len(scores) and scores[i] is not None else -np.inf
    else:
        if not trajectory_path.exists():
            raise FileNotFoundError(f"{exp_path} is missing trajectory.json")
        with open(trajectory_path) as f:
            traj_data = json.load(f)
        lst = traj_data.get("trajectory")
        if not isinstance(lst, list) or not lst:
            raise KeyError(
                f"trajectory.json at {trajectory_path} has no non-empty 'trajectory' list")
        last = lst[-1]
        cand_id = last["cand_id"]
        fitness = last.get("fitness", last.get("score", -np.inf))

    with open(heuristic_path) as f:
        heuristics_data = json.load(f)
    src_by_id = {h["cand_id"]: h["source"] for h in heuristics_data["heuristics"]}
    source = src_by_id.get(cand_id)
    if source is None:
        raise ValueError(f"no source for cand_id={cand_id!r} in {heuristic_path}")

    return exp_path.name, cand_id, source, fitness

def get_class_name(source: str) -> str:
    match = re.search(r"class\s+(\w+)", source)
    if match:
        return match.group(1)
    match_def = re.search(r"def\s+(\w+)", source)
    if match_def:
        return match_def.group(1)
    return "GeneratedAlgorithm"

_worker_source = None
_worker_output_dir = None
_worker_alg_name = None
# Instance indices (iids) each worker evaluates per (fid, dim). Defaults to the
# canonical 5 instances [1..5]; run_evaluation can override it (e.g. test_bbob_slice's
# --inst-indices). Kept as a module global so it reaches eval_fid in the pool workers.
_worker_inst_indices = [1, 2, 3, 4, 5]

def _worker_init(source, output_dir, alg_name, inst_indices=(1, 2, 3, 4, 5)):
    global _worker_source, _worker_output_dir, _worker_alg_name, _worker_inst_indices
    _worker_source = source
    _worker_output_dir = output_dir
    _worker_alg_name = alg_name
    _worker_inst_indices = list(inst_indices)

def eval_fid(fid):
    namespace = {"np": np, "math": __import__("math")}
    exec(_worker_source, namespace)
    AlgoClass = namespace.get(_worker_alg_name)
    if AlgoClass is None:
        raise ValueError(f"Class {_worker_alg_name} not found in source")
    
    # Evaluate for 5D, 10D, 20D
    for dim in [5, 10, 20]:
        budget = 2000 * dim

        for iid in _worker_inst_indices: # configurable instance indices (default [1..5])
            problem = get_problem(fid, dimension=dim, instance=iid, problem_class=ProblemClass.BBOB)

            # Setup logger to save to fid subfolder to avoid race conditions.
            # ioh suffixes duplicate folder names (fid_N, fid_N-1, ...); DataManager
            # pools them all. The Analyzer logs evaluations + y by default — no watch().
            l = logger.Analyzer(
                root=_worker_output_dir,
                folder_name=f"fid_{fid}",
                algorithm_name=_worker_alg_name,
                store_positions=True
            )
            problem.attach_logger(l)

            for rep in range(1, 6): # 5 independent runs per instance
                problem.reset()
                # Fresh algorithm per run: LLaMEA heuristics carry un-reset internal
                # optimization state (f_opt/x_opt/shrink/reinit_count/...), so reusing
                # one instance would contaminate runs 2..N. Training scored a fresh
                # `fn(budget, dim)` per evaluation — match that here.
                alg = AlgoClass(budget=budget, dim=dim)
                # Crash guard: an LLM-generated heuristic may raise on a specific
                # (dim, iid, rep, RNG) path that training never exercised (e.g. an
                # UnboundLocalError). Without this, the exception propagates through
                # the pool and aborts the ENTIRE run (no auc.json). Instead we log a
                # warning and move on — the logger keeps whatever evals completed
                # before the crash, so the AUC just reflects the partial/failed run
                # (a defective heuristic scores near-zero, as it should).
                try:
                    alg(problem)
                except Exception as e:
                    print(f"    [WARN] {_worker_alg_name} crashed on fid={fid} dim={dim} "
                          f"iid={iid} rep={rep}: {type(e).__name__}: {e}", flush=True)

            l.close()   # flush the run data before iohinspector reads it

    return fid


def run_evaluation(source, alg_name, output_dir, n_proc,
                   n_functions=24, inst_indices=(1, 2, 3, 4, 5)):
    """Evaluate ``source`` (class ``alg_name``) on the BBOB grid (``n_functions`` functions
    x {5,10,20}D, over ``inst_indices`` instances x 5 reps) in a process pool, logging ioh
    data under ``output_dir``. Shared by test_bbob.py and test_baselines_bbob.py so both use
    an identical evaluation protocol (fair comparison).

    ``n_functions`` (default 24) selects fids ``1..n_functions``; ``inst_indices`` (default
    the canonical 5 instances ``[1..5]``) selects which instance ids to score per (fid, dim).
    The defaults reproduce the original full BBOB-24 x 5-instance protocol byte-for-byte."""
    inst_indices = list(inst_indices)
    print(f"\nStarting evaluation in parallel across {n_functions} functions "
          f"(instances {inst_indices})...")
    start_time = time.time()
    fids = list(range(1, int(n_functions) + 1))
    with multiprocessing.Pool(processes=n_proc, initializer=_worker_init,
                              initargs=(source, str(output_dir), alg_name, inst_indices)) as pool:
        for fid in pool.imap_unordered(eval_fid, fids):
            print(f"Function {fid:<2} completed successfully.")
    print(f"Evaluation finished in {time.time() - start_time:.1f}s.")


def compute_auc(output_dir, alg_name):
    """Compute the LLaMEA-benchmark "Area under the EAF curve" AUC from the ioh data
    logged under ``output_dir`` and return ``(per_dim, overall)``. Shared by
    test_bbob.py and test_baselines_bbob.py so candidates and baselines score
    IDENTICALLY.

    It is computed exactly as in packages/LLaMEA/benchmarks/ma_bbob/GettingStarted.ipynb:
        df_eaf = transform_fval(df, 1e-8, 1e8)        # normalize precision -> 'eaf' col
        per_fn = get_aocc(df_eaf, budget, free_vars=["function_name","algorithm_name"])
        AUC    = mean of per-function AOCC over the 24 BBOB functions
    get_aocc integrates the normalized best-so-far LINEARLY over evaluations to the full
    2000*dim budget (scale_eval_log=False) — this IS the area under the EAF (== AOCC; the
    paper notes they are the same quantity). NO 1-x flip: already good-oriented.

    NORMALIZATION BOUNDS lower=1e-8, upper=1e8 — the Table I / iohinspector default, NOT
    the 1e2 used as the TRAINING fitness (racing/reprod). Verified: with 1e8 the
    per-function-mean AUC reproduces the reference (ERADS 5D -> 0.7137 vs table 0.733),
    whereas 1e2 gives 0.544. Computed PER DIMENSION (each dim has its own 2000*dim
    budget), then averaged for the overall score."""
    print("\nCalculating AUC with iohinspector...")
    manager = iohinspector.DataManager()
    manager.add_folder(str(output_dir))
    df = manager.load(True, True)   # polars DataFrame

    _AUC_LOWER, _AUC_UPPER = 1e-8, 1e8
    per_dim = {}
    for dim in [5, 10, 20]:
        df_dim = df.filter(pl.col("dimension") == dim)
        if df_dim.height == 0:
            print(f"--- {dim}D: no data ---")
            per_dim[dim] = None
            continue
        budget = 2000 * dim
        df_eaf = iohinspector.metrics.transform_fval(df_dim, _AUC_LOWER, _AUC_UPPER)
        per_fn = iohinspector.metrics.get_aocc(
            df_eaf, budget, free_vars=["function_name", "algorithm_name"])
        auc = float(np.asarray(per_fn["AOCC"], dtype=float).mean())
        per_dim[dim] = auc
        print(f"--- {dim}D (budget {budget}): AUC = {auc:.4f}  "
              f"(mean over {len(per_fn)} functions) ---")

    valid = [v for v in per_dim.values() if v is not None]
    overall = float(np.mean(valid)) if valid else None
    dims_used = [d for d in per_dim if per_dim[d] is not None]
    if overall is not None:
        print(f"\n=== {alg_name}: overall AUC = {overall:.4f} (mean over dims {dims_used}) ===")
    else:
        print(f"\n=== {alg_name}: overall AUC = n/a (no data) ===")
    return per_dim, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        help="A SINGLE experiment folder (its final trajectory/incumbent "
                             "candidate is evaluated). Pass more experiments as more lines "
                             "in the .txt task file and compare their AUCs by hand.")
    parser.add_argument("--has-incumbent", action="store_true", help="Use incumbents instead of trajectory")
    parser.add_argument("--incumbent-mode", type=str, default="mean_rank",
                        choices=["mean_rank", "mean_cost"],
                        help="With --has-incumbent, which validated incumbent series to load: "
                             "'mean_rank' -> valid_trajectory_mean_rank.json, "
                             "'mean_cost' -> valid_trajectory_mean_cost.json.")
    parser.add_argument("--n-proc", type=int, default=12, help="Number of worker processes")
    args = parser.parse_args()

    print(f"Loading final candidate from {args.exp} ...")
    best_exp_name, cand_id, source, best_fitness = load_gen_heuristic(
        args.exp, args.has_incumbent, args.incumbent_mode)

    print(f"Experiment: {best_exp_name}")
    print(f"Cand ID: {cand_id}, Fitness (AOCC in training): {best_fitness}")
    
    alg_name = get_class_name(source)
    print(f"Algorithm Class Name: {alg_name}")
    
    # Output directory
    output_dir = pathlib.Path(".logs/ioh_bbob_outputs") / best_exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {output_dir.resolve()}")
    
    # Save terminal output (stdout logging is useful)
    with open(output_dir / "terminal.txt", "w") as f:
        f.write(f"Experiment: {best_exp_name}\nCand ID: {cand_id}\nAlgorithm: {alg_name}\n")
    
    # Evaluate + score using the SHARED functions (identical to test_baselines_bbob.py).
    run_evaluation(source, alg_name, output_dir, args.n_proc)
    per_dim, overall = compute_auc(output_dir, alg_name)

    # Write a table row like tab1_llamea.json ({"ID", "AUC"}), plus the per-dim
    # breakdown and provenance.
    result_row = {
        "ID": alg_name,
        "AUC": (round(overall, 4) if overall is not None else None),
        "AUC_per_dim": {str(k): (round(v, 4) if v is not None else None)
                        for k, v in per_dim.items()},
        "exp": best_exp_name,
        "cand_id": cand_id,
        "train_fitness": best_fitness,
    }
    with open(output_dir / "auc.json", "w") as f:
        json.dump(result_row, f, indent=2)
    print(f"AUC row -> {output_dir / 'auc.json'}")
    print(f"Raw ioh data saved to {output_dir.resolve()}")

if __name__ == '__main__':
    main()
