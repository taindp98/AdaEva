"""Evaluate checkpoint-sliced populations from a UNIFORM (Fixed-K) LLaMEA BBOB run:
partial (uniform K-instance mean AOCC) vs full-pool (72 BBOB functions) mean AOCC for
ranking correlation analysis.

The BBOB counterpart of eval_racing_bbob_ranking.py for uniform LLaMEA-on-BBOB runs.

This script:
  1. reads trajectory.json to map each budget slice (20/40/60/80/100% of the final
     used_budget) to the checkpoint gen_id whose used_budget is closest;
  2. reconstructs the TRUE environmental-selection candidate pool for that checkpoint
     generation gen_id: entering survivors (survivors_by_gen[gen_id - 1] for gen_id > 1,
     or gen 1 initial population for gen_id == 1) UNION newly generated offspring
     (heuristics[gen_id]), matching the (mu+lambda) selection:
     survivors_by_gen[gen_id - 1] + heuristics[gen_id] -> survivors_by_gen[gen_id];
  3. records each candidate's stored partial score as the partial-eval score — raw
     mean AOCC in [0, 1] (higher is better; 1.0 = instant convergence); non-finite or
     crashed candidates (score <= 0.0) are dropped;
  4. evaluates the exact same candidates on the full instance pool (all 72 noiseless
     BBOB functions: 24 fids x 3 iids at dim 5, budget 10,000, 1 rep), caching by cand_id
     across checkpoints, saving both mean AOCCs and per-instance AOCCs;
  5. ranks the full-pool candidates by their AVERAGE performance (mean AOCC, higher better)
     and maps that order to rank weights via the same compute_rank_weights (no Friedman);
  6. saves full_pool_eval.json and ranking_eval_checkpoints.json with matching keys:
     candidates_ranked_race, candidates_ranked_full, full_mean_performance, full_mean_costs.

Usage:
    python src/analyses/eval_uniform_bbob_ranking.py \
        --exp .logs/tiny_llamea_bbob_vllm/2026-09-11/092508_0_fixinit_mg20_n_runs1_K32_nt2_nc20/ \
        --n-proc 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD"), str(ROOT / "packages" / "LLaMEA")]

from analyses.eval_bbob import (
    build_tasks,
    batch_scoring_parallel,
    _to_serialisable,
    _auto_eval_timeout,
    _N_FUNCTIONS,
)
from analyses._ranking_common import (
    index_heuristics,
    load_survivors,
    build_selection_pool_bbob,
    compute_rank_weights,
)


def load_trajectory(exp_path: pathlib.Path) -> list[dict]:
    """Load the incumbent trajectory rows (one per generation) from trajectory.json."""
    traj_file = exp_path / "trajectory.json"
    if not traj_file.exists():
        raise FileNotFoundError(f"Missing trajectory.json at: {traj_file}")
    with open(traj_file) as f:
        data = json.load(f)
    rows = data.get("trajectory", [])
    if not rows:
        raise ValueError(f"trajectory.json at {traj_file} has an empty 'trajectory'.")
    return rows


def get_sliced_checkpoints(
    traj_rows: list[dict], slices: list[float] = (0.2, 0.4, 0.6, 0.8, 1.0)
) -> list[tuple[str, float, dict]]:
    """Map each budget slice to the trajectory row (generation) whose ``used_budget`` is
    numerically closest to ``slice * final_budget`` (mirrors the racing script)."""
    final_budget = traj_rows[-1]["used_budget"]
    selected = []
    for s in slices:
        target = s * final_budget
        closest = min(traj_rows, key=lambda r: abs(r["used_budget"] - target))
        label = f"{int(round(s * 100))}%"
        selected.append((label, s, closest))
    return selected


def evaluate_candidates_on_full_pool(
    cand_ids: list[str],
    src_by_id: dict[str, str],
    tasks: list[tuple[int, int, int]],
    dim: int,
    budget: int,
    eval_timeout: float,
    cache_means: dict[str, float],
    cache_aoccs: dict[str, list[float]],
    n_cores: int = 20,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Evaluate candidates on the full 72-problem BBOB pool with caching across checkpoints
    using batch_scoring_parallel from analyses.eval_bbob.
    
    Returns ({cand_id: mean_AOCC}, {cand_id: [aocc_0, ..., aocc_71]}).
    """
    pending_ids = [cid for cid in cand_ids if cid not in cache_aoccs]
    if pending_ids:
        heuristics_to_eval = [
            {"cand_id": cid, "source": src_by_id[cid]} for cid in pending_ids
        ]
        scores, _rts, _pc, per_inst = batch_scoring_parallel(
            tasks,
            heuristics_to_eval,
            dim=dim,
            budget=budget,
            eval_timeout=eval_timeout,
            n_cores=n_cores,
            return_per_instance=True,
        )
        for cid, score, aoccs in zip(pending_ids, scores, per_inst):
            cache_means[cid] = float(score)
            cache_aoccs[cid] = [float(a) for a in aoccs]

    return (
        {cid: cache_means[cid] for cid in cand_ids},
        {cid: cache_aoccs[cid] for cid in cand_ids},
    )


def compute_full_pool_mean_ranking(
    cand_ids: list[str],
    cand_means: dict[str, float],
    src_by_id: dict[str, str],
    cost_tol: float = 1e-8,
) -> tuple[dict[str, float], list[str]]:
    """Rank the full-pool candidates by their AVERAGE performance (mean AOCC over the 72
    functions, HIGHER = better), then assign rank weights with the SAME
    ``compute_rank_weights`` used on the partial side (no per-instance Friedman).

      1. Order candidates best->worst by mean AOCC DESCENDING (tie-broken by cand_id).
      2. ``compute_rank_weights`` maps that order to weights in (0, 1] (top-1 = 1.0, last =
         1/N), grouping identical candidates (same source, or |mean_i - mean_j| < cost_tol).
         (compute_rank_weights' tie check is on |mean diff|, so AOCC direction is irrelevant
         there; only the best->worst ORDER above encodes 'higher is better'.)

    Returns:
      - full_rank_scores: {cid: float normalized rank weight in (0, 1]}
      - full_order: [cid in best-to-worst mean-AOCC order]
    """
    full_order = sorted(cand_ids, key=lambda cid: (-cand_means[cid], cid))
    full_rank_scores = compute_rank_weights(
        full_order, cand_means, src_by_id=src_by_id, cost_tol=cost_tol
    )
    return full_rank_scores, full_order


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint-sliced generation populations from a uniform (Fixed-K) "
                    "LLaMEA BBOB run on partial (K-instance mean AOCC) vs full-pool."
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment directory containing trajectory.json and heuristics.json.",
    )
    parser.add_argument(
        "--n-proc",
        "--n-cores",
        type=int,
        default=20,
        dest="n_proc",
        help="Number of parallel worker processes (default: 20).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=5,
        help="Problem dimension (default: 5, matching bbob_surrogate).",
    )
    parser.add_argument(
        "--budget-factor",
        type=int,
        default=2000,
        help="Budget factor: evaluations = budget_factor * dim (default: 2000 -> 10000 evals).",
    )
    parser.add_argument(
        "--n-rep",
        type=int,
        default=1,
        help="Repetitions per (fid, iid) pair (default: 1 -> 72 tasks total).",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=-1.0,
        help="Per-evaluation timeout in seconds (default: -1.0 = auto-compute).",
    )
    parser.add_argument(
        "--cost-tol",
        type=float,
        default=1e-8,
        help="Tolerance threshold for identical cost equivalence check (default: 1e-8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: verify checkpoints and candidates without running full-pool evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output combined JSON file (default: <exp>/ranking_eval_checkpoints.json).",
    )
    parser.add_argument(
        "--full-pool-output",
        type=str,
        default=None,
        help="Output JSON file for full-pool mean AOCCs (default: <exp>/full_pool_eval.json).",
    )
    args = parser.parse_args()

    exp_path = pathlib.Path(args.exp)
    if not exp_path.is_absolute():
        exp_path = (ROOT / exp_path).resolve()
    if not exp_path.exists():
        for suffix in ["_savpop", "_pmoriginal"]:
            cand = exp_path.parent / f"{exp_path.name}{suffix}"
            if cand.exists():
                print(f"Path '{exp_path}' not found, falling back to '{cand}'.")
                exp_path = cand
                break
        else:
            raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")

    heuristics_path = exp_path / "heuristics.json"
    if not heuristics_path.exists():
        raise FileNotFoundError(f"heuristics.json not found at: {heuristics_path}")

    out_file = (
        pathlib.Path(args.output)
        if args.output is not None
        else exp_path / "ranking_eval_checkpoints.json"
    )
    full_pool_file = (
        pathlib.Path(args.full_pool_output)
        if args.full_pool_output is not None
        else exp_path / "full_pool_eval.json"
    )

    budget = args.budget_factor * args.dim
    resolved_timeout = (
        _auto_eval_timeout(args.budget_factor, args.dim)
        if (args.eval_timeout is not None and args.eval_timeout < 0)
        else args.eval_timeout
    )

    print("=" * 70)
    print("Uniform BBOB Ranking Evaluation: Checkpoints Partial vs Full-Pool")
    print(f"Experiment: {exp_path}")
    print(f"Cores: {args.n_proc} | Dim: {args.dim} | Budget: {budget} (bf={args.budget_factor})")
    print(f"Repetitions: {args.n_rep} (72 * {args.n_rep} = {72 * args.n_rep} evals/cand)")
    print(f"Timeout per task: {resolved_timeout:.1f}s")
    print(f"Full-pool output: {full_pool_file}")
    print(f"Combined output:  {out_file}")
    print("=" * 70)

    t_start = time.time()

    # 1. Load heuristics, survivors, and trajectory; map budget slices -> checkpoint generations
    print("\n[Step 1/4] Loading trajectory, survivors, and heuristics...", flush=True)
    with open(heuristics_path) as f:
        hdata = json.load(f)
    heuristics = hdata.get("heuristics", [])
    by_id = index_heuristics(heuristics)
    print(f"  Loaded {len(heuristics)} sampled heuristics.")

    survivors_by_gen = load_survivors(exp_path)
    print(f"  Loaded survivors for {len(survivors_by_gen)} generations from survivors.json.")

    traj_rows = load_trajectory(exp_path)
    slices = [0.2, 0.4, 0.6, 0.8, 1.0]
    checkpoints = get_sliced_checkpoints(traj_rows, slices=slices)
    final_budget = traj_rows[-1]["used_budget"]
    print(f"  Found {len(traj_rows)} trajectory rows. Final budget: {final_budget}")

    if args.dry_run:
        print("\n[Dry Run] Verifying checkpoints and heuristics IDs...", flush=True)
        for slice_label, slice_frac, traj_row in checkpoints:
            used_b = traj_row["used_budget"]
            gen_id = traj_row["gen_id"]

            cand_ids, partial_scores, src_by_id, combined_gens = build_selection_pool_bbob(
                gen_id, by_id, survivors_by_gen
            )
            missing = [cid for cid in cand_ids if cid not in by_id]
            if missing:
                raise KeyError(f"Candidates {missing} from gen {gen_id} not found in heuristics.json")

            candidates_ranked_race = sorted(cand_ids, key=lambda cid: (-partial_scores[cid], cid))
            partial_rank_scores = compute_rank_weights(
                candidates_ranked_race, partial_scores, src_by_id=src_by_id, cost_tol=args.cost_tol
            )
            min_w = min(partial_rank_scores.values())
            max_w = max(partial_rank_scores.values())
            print(
                f"  Checkpoint {slice_label:>4s} (gen_id={gen_id:3d} pool=gens{combined_gens}, "
                f"budget={used_b:6d}): {len(cand_ids):2d} candidates in selection pool, "
                f"partial RankWeight range: [{min_w:.4f}, {max_w:.4f}]",
                flush=True,
            )
        print("\nDry run completed successfully! All candidates verified against heuristics.json.")
        return

    # 2. Build tasks for the full pool (all 72 noiseless BBOB functions: 24 fids x 3 iids)
    print("\n[Step 2/4] Building full-pool BBOB tasks...", flush=True)
    tasks = build_tasks(args.n_rep)
    print(f"  Full pool tasks ready: {len(tasks)} tasks (72 functions x {args.n_rep} rep)")

    # Resume from existing full_pool_eval.json if available (and the eval params match)
    full_eval_means: dict[str, float] = {}
    full_eval_aoccs: dict[str, list[float]] = {}
    if full_pool_file.exists():
        try:
            with open(full_pool_file) as f:
                prev_full = json.load(f)
            prev_params = (
                prev_full.get("dim"),
                prev_full.get("budget_factor"),
                prev_full.get("n_rep"),
            )
            cur_params = (args.dim, args.budget_factor, args.n_rep)
            if prev_params == cur_params:
                prev_scores = prev_full.get("mean_aoccs", prev_full.get("mean_costs", {}))
                prev_per_inst = prev_full.get("per_instance_aoccs", prev_full.get("per_instance_costs", {}))
                valid_resumed = {
                    k: float(v) for k, v in prev_scores.items()
                    if v is not None and k in prev_per_inst and prev_per_inst[k] is not None
                }
                full_eval_means.update(valid_resumed)
                full_eval_aoccs.update({
                    k: [float(x) for x in prev_per_inst[k]]
                    for k in valid_resumed
                })
                print(f"  Resumed {len(full_eval_means)} candidate mean AOCCs and "
                      f"{len(full_eval_aoccs)} per-instance AOCC vectors from {full_pool_file.name}")
            else:
                print(f"  NOTE: {full_pool_file.name} was built with different eval params "
                      f"{prev_params} != {cur_params}; ignoring its cache (re-evaluating).")
        except Exception as e:
            print(f"  Warning: failed to read existing {full_pool_file.name}: {e}")

    # 3. Evaluate each checkpoint
    print("\n[Step 3/4] Evaluating sliced checkpoints (partial per-generation vs full-pool)...", flush=True)
    dict_partial_eval: dict[str, dict[str, float]] = {}
    dict_full_eval: dict[str, dict[str, float]] = {}
    checkpoint_meta: dict[str, dict] = {}

    for slice_label, slice_frac, traj_row in checkpoints:
        t_cp = time.time()
        used_b = traj_row["used_budget"]
        gen_id = traj_row["gen_id"]

        # Form the TRUE environmental-selection pool for BBOB:
        # entering survivors (survivors_by_gen[gen_id - 1]) + newly generated offspring (heuristics[gen_id])
        cand_ids, partial_scores, src_by_id, combined_gens = build_selection_pool_bbob(
            gen_id, by_id, survivors_by_gen
        )

        # Candidates ranked by partial evaluation (best to worst: highest AOCC first)
        candidates_ranked_race = sorted(cand_ids, key=lambda cid: (-partial_scores[cid], cid))

        # Evaluate the exact same candidates on the full 72 BBOB pool
        sub_means, sub_aoccs = evaluate_candidates_on_full_pool(
            cand_ids=cand_ids,
            src_by_id=src_by_id,
            tasks=tasks,
            dim=args.dim,
            budget=budget,
            eval_timeout=resolved_timeout,
            cache_means=full_eval_means,
            cache_aoccs=full_eval_aoccs,
            n_cores=args.n_proc,
        )

        # Full-pool ranking by AVERAGE performance (mean AOCC, higher better) -> rank weights
        # via the same compute_rank_weights used on the partial side (no Friedman).
        full_scores, full_order = compute_full_pool_mean_ranking(
            cand_ids=cand_ids,
            cand_means=sub_means,
            src_by_id=src_by_id,
            cost_tol=args.cost_tol,
        )

        # Normalized rank index for partial evaluation based on candidates_ranked_race
        partial_rank_scores = compute_rank_weights(
            candidates_ranked_race, partial_scores, src_by_id=src_by_id, cost_tol=args.cost_tol
        )

        dict_partial_eval[slice_label] = partial_rank_scores
        dict_full_eval[slice_label] = full_scores
        checkpoint_meta[slice_label] = {
            "slice_frac": slice_frac,
            "gen_id": gen_id,
            "combined_gens": combined_gens,
            "used_budget": used_b,
            "n_candidates": len(cand_ids),
            "candidates_ranked_race": candidates_ranked_race,
            "candidates_ranked_full": full_order,
            "full_mean_performance": sub_means,        # mean AOCC (higher is better)
            "full_mean_aoccs": sub_means,              # alias (mean AOCC, higher=better)
            "partial_mean_scores": partial_scores,     # partial mean AOCC (higher=better)
            # NEGATED-AOCC (lower = better) views, matching the racing-bbob convention so the
            # notebook Recall@k / Top-k can rank partial & full ASCENDING uniformly.
            "full_mean_costs": {cid: -float(sub_means[cid]) for cid in cand_ids},
            "partial_mean_costs": {cid: -float(partial_scores[cid]) for cid in cand_ids},
        }

        print(
            f"  Checkpoint {slice_label:>4s} (gen_id={gen_id:3d} pool=gens{combined_gens}, "
            f"budget={used_b:6d}): {len(cand_ids):2d} candidates in selection pool "
            f"(unique evaluated: {len(full_eval_means)}, step time: {time.time()-t_cp:.1f}s)",
            flush=True,
        )

    # 4. Save results to JSON
    # A) Dedicated full-pool evaluation JSON file in exp_path
    print(f"\n[Step 4/4] Saving full-pool evaluations to {full_pool_file}...", flush=True)
    full_pool_data = {
        "exp": str(exp_path),
        "metric": "mean_AOCC (higher is better; in [0,1], 1.0 = optimum)",
        "dim": args.dim,
        "budget_factor": args.budget_factor,
        "n_rep": args.n_rep,
        "n_functions": _N_FUNCTIONS,
        "eval_timeout": resolved_timeout,
        "total_unique_candidates": len(full_eval_means),
        "mean_aoccs": full_eval_means,
        "per_instance_costs": full_eval_aoccs,
        "per_instance_aoccs": full_eval_aoccs,
    }
    with open(full_pool_file, "w") as f:
        json.dump(full_pool_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved full-pool mean AOCCs and per-instance AOCCs -> {full_pool_file}")

    # B) Combined ranking checkpoints evaluation JSON file
    print(f"Saving combined ranking checkpoint evaluations to {out_file}...", flush=True)
    output_data = {
        "exp": str(exp_path),
        "mode": "uniform",
        "metric": "rank_weight (higher is better; in (0, 1], 1.0 = top-1 candidate)",
        "slices": slices,
        "dim": args.dim,
        "budget_factor": args.budget_factor,
        "n_rep": args.n_rep,
        "n_functions": _N_FUNCTIONS,
        "eval_timeout": resolved_timeout,
        "total_unique_candidates": len(full_eval_means),
        "checkpoints": checkpoint_meta,
        "partial_eval": dict_partial_eval,
        "full_eval": dict_full_eval,
    }
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved combined data -> {out_file}")

    # Summary verification
    print("\n" + "=" * 70)
    print("Verification Summary:")
    for slice_frac in slices:
        lbl = f"{int(round(slice_frac * 100))}%"
        p_dict = dict_partial_eval[lbl]
        f_dict = dict_full_eval[lbl]
        assert set(p_dict.keys()) == set(f_dict.keys()), f"Key mismatch in {lbl}"  # sets: partial & full dicts share candidates but differ in order (partial-cost vs full-mean ranked)
        print(f"  [{lbl}] Candidates: {len(p_dict):2d} | 1:1 ID Alignment: TRUE")
        sample_cid = list(p_dict.keys())[0]
        print(f"       Sample {sample_cid}: Partial AOCC={p_dict[sample_cid]:.4f}, Full AOCC={f_dict[sample_cid]:.4f}")

    print(f"\nTotal elapsed time: {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
