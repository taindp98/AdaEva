"""Evaluate checkpoint-sliced populations from a UNIFORM (non-racing) eoh/tiny OBP run:
partial (uniform, per-generation) score vs full-pool (25 instances) score for ranking
correlation analysis.

Counterpart of eval_racing_obp_ranking.py for uniform (fixed-K) OBP runs.

This script:
  1. reads ``trajectory.json`` to map each budget slice (20/40/60/80/100% of the final
     ``used_budget``) to the checkpoint ``gen_id`` whose ``used_budget`` is closest;
  2. reconstructs the TRUE environmental-selection candidate pool for that
     checkpoint generation gen_id: entering survivors (survivors_by_gen[gen_id + 1])
     UNION newly generated offspring (heuristics[gen_id]), matching the selection:
     survivors_by_gen[gen_id + 1] + heuristics[gen_id] -> survivors_by_gen[gen_id + 2];
  3. uses each candidate's stored partial score as the partial-eval score — the
     heuristics.json score is NEGATED (cost = -score); penalty/crash candidates
     (cost >= 1e5) are dropped;
  4. re-evaluates the exact same candidates on the full instance pool (25 instances),
     caching by cand_id across checkpoints, saving both mean costs and per-instance costs;
  5. ranks the full-pool candidates by their AVERAGE performance (mean cost, lower better)
     and maps that order to rank weights via the same compute_rank_weights (no Friedman);
  6. saves full_pool_eval.json and ranking_eval_checkpoints.json with matching keys:
     candidates_ranked_race, candidates_ranked_full, full_mean_performance, full_mean_costs.

Usage:
    python src/analyses/eval_uniform_obp_ranking.py \
        --exp .logs/tiny_eoh_obp_vllm/2026-09-07/122427_0_fixinit_.../ --n-proc 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD")]

from analyses.eval_obp import load_instances, batch_scoring_parallel, _to_serialisable
from analyses._ranking_common import index_heuristics, load_survivors, build_selection_pool, compute_rank_weights

# Cost threshold above which a candidate is treated as a crash/penalty and dropped.
# Matches the racing pipeline's filter and base.py's "clean candidate" convention
# (BIG_PENALTY = 1e6, threshold 1e5); catches the -1e6 fix-init + -5000 sentinels.
PENALTY_COST = 1e5


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
    instances: list,
    cache_means: dict[str, float],
    cache_costs: dict[str, list[float]],
    n_cores: int = 20,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Evaluate candidates on the full instance pool with caching across checkpoints using
    batch_scoring_parallel from analyses.eval_obp.
    
    Returns ({cand_id: mean_cost}, {cand_id: [cost_0, ..., cost_24]}).
    """
    pending_ids = [cid for cid in cand_ids if cid not in cache_costs]
    if pending_ids:
        heuristics_to_eval = [
            {"cand_id": cid, "source": src_by_id[cid]} for cid in pending_ids
        ]
        scores, per_inst = batch_scoring_parallel(instances, heuristics_to_eval, n_cores=n_cores)
        for cid, score, inst_costs in zip(pending_ids, scores, per_inst):
            cache_means[cid] = float(score)
            cache_costs[cid] = [float(c) for c in inst_costs]

    return (
        {cid: cache_means[cid] for cid in cand_ids},
        {cid: cache_costs[cid] for cid in cand_ids},
    )


def compute_full_pool_mean_ranking(
    cand_ids: list[str],
    cand_means: dict[str, float],
    src_by_id: dict[str, str],
    cost_tol: float = 1e-8,
) -> tuple[dict[str, float], list[str]]:
    """Rank the full-pool candidates by their AVERAGE performance (mean cost over the 25
    instances, lower = fewer bins = better), then assign rank weights with the SAME
    ``compute_rank_weights`` used on the partial side.

    This replaces the earlier per-instance Friedman mean-rank mechanism: instead of ranking
    each instance and averaging ranks, we rank candidates directly by their mean cost.

      1. Order candidates best->worst by mean cost ascending (tie-broken by cand_id).
      2. ``compute_rank_weights`` maps that order to weights in (0, 1] (top-1 = 1.0, last =
         1/N), grouping identical candidates (same source, or |mean_i - mean_j| < cost_tol)
         onto a shared average weight.

    Returns:
      - full_rank_scores: {cid: float normalized rank weight in (0, 1]}
      - full_order: [cid in best-to-worst mean-performance order]
    """
    full_order = sorted(cand_ids, key=lambda cid: (cand_means[cid], cid))
    full_rank_scores = compute_rank_weights(
        full_order, cand_means, src_by_id=src_by_id, cost_tol=cost_tol
    )
    return full_rank_scores, full_order


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint-sliced generation populations from a uniform OBP run "
                    "on partial (per-generation) score vs full-pool."
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment directory containing trajectory.json, heuristics.json, and survivors.json.",
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
        "--n-instances",
        type=int,
        default=25,
        help="Number of full-pool instances (default: 25).",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=5000,
        help="Number of items per OBP instance (default: 5000).",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=100,
        help="Bin capacity (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for full-pool instances (default: 1, matching _final_eval).",
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
        help="Output JSON file for full-pool mean costs (default: <exp>/full_pool_eval.json).",
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
        raise FileNotFoundError(f"Missing heuristics.json at: {heuristics_path}")

    out_file = (
        pathlib.Path(args.output)
        if args.output
        else exp_path / "ranking_eval_checkpoints.json"
    )
    full_pool_file = (
        pathlib.Path(args.full_pool_output)
        if args.full_pool_output
        else exp_path / "full_pool_eval.json"
    )

    print("=" * 70)
    print("Uniform OBP Ranking Evaluation: Checkpoints Partial vs Full-Pool")
    print(f"Experiment: {exp_path}")
    print(f"Cores: {args.n_proc} | Instances: {args.n_instances} (items={args.n_items}, cap={args.capacity}, seed={args.seed})")
    print(f"Full-pool output: {full_pool_file}")
    print(f"Combined output:  {out_file}")
    print("=" * 70)

    t_start = time.time()

    # 1. Load heuristics and trajectory; map budget slices -> checkpoint generations
    print("\n[Step 1/4] Loading trajectory and heuristics...", flush=True)
    with open(heuristics_path) as f:
        hdata = json.load(f)
    heuristics = hdata["heuristics"]
    by_id = index_heuristics(heuristics)
    print(f"  Loaded {len(heuristics)} sampled heuristics.")

    # survivors.json: per-gen environmental-selection survivor cand_ids (REQUIRED).
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

            cand_ids, partial_costs, src_by_id, combined_gens = build_selection_pool(
                gen_id, by_id, survivors_by_gen, penalty_cost=PENALTY_COST
            )
            missing = [cid for cid in cand_ids if cid not in by_id]
            if missing:
                raise KeyError(f"Candidates {missing} from gen {gen_id} not found in heuristics.json")

            candidates_ranked_race = sorted(cand_ids, key=lambda cid: (partial_costs[cid], cid))
            partial_rank_scores = compute_rank_weights(
                candidates_ranked_race, partial_costs, src_by_id=src_by_id, cost_tol=args.cost_tol
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

    # 2. Load full-pool instances
    print("\n[Step 2/4] Generating / loading full-pool OBP instances...", flush=True)
    avg_lb, full_instances = load_instances(
        n_instances=args.n_instances,
        n_items=args.n_items,
        capacity=args.capacity,
        seed=args.seed,
    )
    print(f"  Full pool ready: {len(full_instances)} instances (avg lower bound = {avg_lb:.4f})")

    # Resume from existing full_pool_eval.json if available (and the instance params match).
    full_eval_means: dict[str, float] = {}
    full_eval_costs: dict[str, list[float]] = {}
    if full_pool_file.exists():
        try:
            with open(full_pool_file) as f:
                prev_full = json.load(f)
            prev_params = (
                prev_full.get("n_instances"), prev_full.get("n_items"),
                prev_full.get("capacity"), prev_full.get("seed"),
            )
            cur_params = (args.n_instances, args.n_items, args.capacity, args.seed)
            if prev_params == cur_params:
                prev_means = prev_full.get("mean_costs", {})
                prev_per_inst = prev_full.get("per_instance_costs", {})
                valid_resumed = {
                    k: float(v) for k, v in prev_means.items()
                    if v is not None and k in prev_per_inst and prev_per_inst[k] is not None
                }
                full_eval_means.update(valid_resumed)
                full_eval_costs.update({
                    k: [float(x) for x in prev_per_inst[k]]
                    for k in valid_resumed
                })
                print(f"  Resumed {len(full_eval_means)} candidate mean costs and "
                      f"{len(full_eval_costs)} per-instance cost vectors from {full_pool_file.name}")
            else:
                print(f"  NOTE: {full_pool_file.name} was built with different instance params "
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

        # Form the TRUE environmental-selection pool: entering survivors (survivors_by_gen[gen_id + 1])
        # + newly generated offspring (heuristics[gen_id]), dedup by cand_id.
        # Each candidate's partial cost = its own stored score (cost = -score).
        cand_ids, partial_costs, src_by_id, combined_gens = build_selection_pool(
            gen_id, by_id, survivors_by_gen, penalty_cost=PENALTY_COST
        )

        # Candidates ranked by partial evaluation (best to worst: lowest cost first)
        candidates_ranked_race = sorted(cand_ids, key=lambda cid: (partial_costs[cid], cid))

        # Evaluate the exact same candidates on the full instance pool.
        sub_means, sub_costs = evaluate_candidates_on_full_pool(
            cand_ids=cand_ids,
            src_by_id=src_by_id,
            instances=full_instances,
            cache_means=full_eval_means,
            cache_costs=full_eval_costs,
            n_cores=args.n_proc,
        )

        # Full-pool ranking by AVERAGE performance (mean cost, lower better) -> rank weights
        # via the same compute_rank_weights used on the partial side (no Friedman).
        full_scores, full_order = compute_full_pool_mean_ranking(
            cand_ids=cand_ids,
            cand_means=sub_means,
            src_by_id=src_by_id,
            cost_tol=args.cost_tol,
        )

        # Normalized rank index for partial evaluation based on candidates_ranked_race
        partial_rank_scores = compute_rank_weights(
            candidates_ranked_race, partial_costs, src_by_id=src_by_id, cost_tol=args.cost_tol
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
            "full_mean_performance": sub_means,
            "full_mean_costs": sub_means,
            "partial_mean_costs": partial_costs,
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
        "metric": "rank_weight (higher is better; in (0, 1], 1.0 = top-1 candidate)",
        "n_instances": args.n_instances,
        "n_items": args.n_items,
        "capacity": args.capacity,
        "seed": args.seed,
        "avg_lb": avg_lb,
        "total_unique_candidates": len(full_eval_means),
        "mean_costs": full_eval_means,
        "per_instance_costs": full_eval_costs,
    }
    with open(full_pool_file, "w") as f:
        json.dump(full_pool_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved full-pool mean costs and per-instance costs -> {full_pool_file}")

    # B) Combined ranking checkpoints evaluation JSON file
    print(f"Saving combined ranking checkpoint evaluations to {out_file}...", flush=True)
    output_data = {
        "exp": str(exp_path),
        "mode": "uniform",
        "metric": "rank_weight (higher is better; in (0, 1], 1.0 = top-1 candidate)",
        "slices": slices,
        "n_instances_full": args.n_instances,
        "n_items": args.n_items,
        "capacity": args.capacity,
        "seed": args.seed,
        "avg_lb": avg_lb,
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
        print(f"       Sample {sample_cid}: Partial RankWeight={p_dict[sample_cid]:.4f}, Full Mean-Rank Weight={f_dict[sample_cid]:.4f}")

    print(f"\nTotal elapsed time: {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
