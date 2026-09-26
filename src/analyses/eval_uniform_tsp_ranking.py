"""Evaluate checkpoint-sliced populations from a UNIFORM (non-racing) eoh/tiny TSP-GLS run:
partial (uniform, per-generation) score vs full-pool (64 instances) score for ranking
correlation analysis. The TSP counterpart of eval_uniform_obp_ranking.py.

Like the OBP uniform script, the tiny/reprod TSP runner evaluates every candidate in a
generation on the SAME number of instances (K), so the partial score is already fair within
a generation. This script:

  1. reads ``trajectory.json`` to map each budget slice (20/40/60/80/100% of the final
     ``used_budget``) to the checkpoint ``gen_id`` whose ``used_budget`` is closest, and
     recovers the RUN's training mean_opt (= score / (1 + gap); trajectory ``gap`` is a
     FRACTION) for the partial gap%;
  2. reconstructs the TRUE environmental-selection candidate pool for that
     checkpoint generation gen_id: entering survivors (survivors_by_gen[gen_id + 1])
     UNION newly generated offspring (heuristics[gen_id]), matching the selection:
     survivors_by_gen[gen_id + 1] + heuristics[gen_id] -> survivors_by_gen[gen_id + 2];
  3. uses each candidate's stored partial score as the partial-eval cost — the
     heuristics.json score is NEGATED (cost = -score; failed heuristics store -inf ->
     +inf); penalty/non-finite candidates are dropped. Partial gap% uses the training
     mean_opt;
  4. re-evaluates the exact same candidates on the full instance pool
     (64 instances, problem_size=100, seed=2024 — the same pool evaluate_single/_final_eval
     validate on), caching raw cost by cand_id across checkpoints. Full gap% uses the
     full-pool (64-instance) mean_opt.
  5. computes Friedman mean ranks on the full pool and stores candidate rankings for
     both race (partial) and full-pool evaluation.

TSP full-pool scoring is EXPENSIVE (Concorde optima + GLS per instance), so caching by
cand_id and resuming full_pool_eval.json matter.

Usage:
    python src/analyses/eval_uniform_tsp_ranking.py \\
        --exp .logs/tiny_eoh_tsp_gls_vllm/2026-08-05/150704_0_mg10_K5_.../ --n-proc 20
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

from analyses.eval_tsp import load_instances, batch_scoring_parallel, _to_serialisable
from analyses._ranking_common import index_heuristics, load_survivors, build_selection_pool, compute_rank_weights

# Cost threshold above which a candidate is treated as a crash/penalty and dropped.
# TSP failed heuristics store score=-inf -> cost=+inf (caught by the isfinite check);
# a large finite guard is kept for parity with the OBP pipeline's convention.
PENALTY_COST = 1e5


def _gap_pct(cost: float, mean_opt: float) -> float | None:
    """Percentage optimality gap: (cost - mean_opt) / mean_opt * 100. None if not
    computable (non-finite cost or non-positive mean_opt)."""
    if cost is None or not np.isfinite(cost) or not np.isfinite(mean_opt) or mean_opt <= 0:
        return None
    return (cost - mean_opt) / mean_opt * 100.0


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


def recover_train_mean_opt(traj_rows: list[dict]) -> float:
    """Recover the run's TRAINING mean_opt from a finite-gap trajectory row:
    trajectory ``gap`` is a FRACTION with gap = (score - mean_opt)/mean_opt, so
    mean_opt = score / (1 + gap). Returns nan if no finite-gap row exists."""
    for r in traj_rows:
        g, s = r.get("gap"), r.get("score")
        if (isinstance(g, (int, float)) and isinstance(s, (int, float))
                and g == g and s == s and abs(g) < 1e18 and (1.0 + g) != 0):
            return float(s) / (1.0 + float(g))
    return float("nan")


def get_sliced_checkpoints(
    traj_rows: list[dict], slices: list[float] = (0.2, 0.4, 0.6, 0.8, 1.0)
) -> list[tuple[str, float, dict]]:
    """Map each budget slice to the trajectory row (generation) whose ``used_budget`` is
    numerically closest to ``slice * final_budget`` (mirrors the OBP uniform script)."""
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
    """Evaluate candidates (raw mean cost and per-instance costs) on the full instance
    pool with caching across checkpoints using batch_scoring_parallel from analyses.eval_tsp.

    Returns ({cand_id: mean_cost}, {cand_id: [cost_0, ..., cost_63]}).
    """
    pending_ids = [cid for cid in cand_ids if cid not in cache_costs]
    if pending_ids:
        heuristics_to_eval = [
            {"cand_id": cid, "source": src_by_id[cid]} for cid in pending_ids
        ]
        scores, _runtimes, per_inst = batch_scoring_parallel(
            instances, heuristics_to_eval, n_cores=n_cores
        )
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
    """Rank the full-pool candidates by their AVERAGE performance (mean cost over the
    instances, lower = shorter tour = better), then assign rank weights with the SAME
    ``compute_rank_weights`` used on the partial side (no per-instance Friedman).

      1. Order candidates best->worst by mean cost ascending (tie-broken by cand_id).
      2. ``compute_rank_weights`` maps that order to weights in (0, 1] (top-1 = 1.0, last =
         1/N), grouping identical candidates (same source, or |mean_i - mean_j| < cost_tol).

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
        description="Evaluate checkpoint-sliced generation populations from a uniform TSP-GLS "
                    "run on partial (per-generation) score vs full-pool."
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
        "--n-instances",
        type=int,
        default=64,
        help="Number of full-pool instances (default: 64, matching _final_eval).",
    )
    parser.add_argument(
        "--problem-size",
        type=int,
        default=100,
        help="TSP problem size / number of cities (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="RNG seed for full-pool instances (default: 2024, matching eval_tsp).",
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
    parser.add_argument(
        "--cost-tol",
        type=float,
        default=1e-8,
        help="Tolerance threshold for identical cost equivalence check (default: 1e-8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint parsing and heuristics without executing expensive evaluations.",
    )
    args = parser.parse_args()

    exp_path = pathlib.Path(args.exp)
    if not exp_path.is_absolute():
        exp_path = (ROOT / exp_path).resolve()
    if not exp_path.exists():
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
    print(f"Uniform TSP-GLS Ranking Evaluation: Checkpoints Partial vs Full-Pool")
    print(f"Experiment: {exp_path}")
    print(f"Cores: {args.n_proc} | Instances: {args.n_instances} (problem_size={args.problem_size}, seed={args.seed})")
    print(f"Full-pool output: {full_pool_file}")
    print(f"Combined output:  {out_file}")
    print("=" * 70)

    # 1. Load heuristics and trajectory; map budget slices -> checkpoint generations.
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
    train_mean_opt = recover_train_mean_opt(traj_rows)
    slices = [0.2, 0.4, 0.6, 0.8, 1.0]
    checkpoints = get_sliced_checkpoints(traj_rows, slices=slices)
    final_budget = traj_rows[-1]["used_budget"]
    print(f"  Found {len(traj_rows)} trajectory rows. Final budget: {final_budget}")
    print(f"  Training mean_opt (recovered from trajectory) = {train_mean_opt:.4f}")

    if args.dry_run:
        print("\n[Dry Run] Verifying checkpoints and heuristics IDs...")
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

    # 2. Load full-pool instances (Concorde optima). full-pool mean_opt used for full gap%.
    t_start = time.time()
    print("\n[Step 2/4] Generating / loading full-pool TSP instances (Concorde optima)...", flush=True)
    full_mean_opt, full_instances = load_instances(
        n_instances=args.n_instances,
        problem_size=args.problem_size,
        seed=args.seed,
    )
    print(f"  Full pool ready: {len(full_instances)} instances (mean_opt = {full_mean_opt:.4f})")

    # Resume from existing full_pool_eval.json if the instance params match.
    full_eval_means: dict[str, float] = {}
    full_eval_costs: dict[str, list[float]] = {}
    if full_pool_file.exists():
        try:
            with open(full_pool_file) as f:
                prev_full = json.load(f)
            prev_params = (
                prev_full.get("n_instances"), prev_full.get("problem_size"),
                prev_full.get("seed"),
            )
            cur_params = (args.n_instances, args.problem_size, args.seed)
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
    dict_partial_cost: dict[str, dict[str, float]] = {}
    dict_partial_gap: dict[str, dict[str, float | None]] = {}
    dict_full_cost: dict[str, dict[str, float]] = {}
    dict_full_gap: dict[str, dict[str, float | None]] = {}
    dict_partial_eval: dict[str, dict[str, float]] = {}
    dict_full_eval: dict[str, dict[str, float]] = {}
    checkpoint_meta: dict[str, dict] = {}

    for slice_label, slice_frac, traj_row in checkpoints:
        t_cp = time.time()
        used_b = traj_row["used_budget"]
        gen_id = traj_row["gen_id"]

        # Form the TRUE environmental-selection pool: entering survivors (survivors_by_gen[gen_id + 1])
        # + newly generated offspring (heuristics[gen_id]), dedup by cand_id.
        cand_ids, partial_costs, src_by_id, combined_gens = build_selection_pool(
            gen_id, by_id, survivors_by_gen, penalty_cost=PENALTY_COST)
        partial_gaps = {cid: _gap_pct(c, train_mean_opt) for cid, c in partial_costs.items()}

        # Candidates ranked by partial evaluation (best to worst: lowest cost first)
        candidates_ranked_race = sorted(cand_ids, key=lambda cid: (partial_costs[cid], cid))

        # Normalized rank index for partial evaluation based on candidates_ranked_race
        partial_rank_scores = compute_rank_weights(
            candidates_ranked_race, partial_costs, src_by_id=src_by_id, cost_tol=args.cost_tol
        )

        # Evaluate the exact same candidates on the full instance pool.
        sub_means, sub_costs = evaluate_candidates_on_full_pool(
            cand_ids=cand_ids,
            src_by_id=src_by_id,
            instances=full_instances,
            cache_means=full_eval_means,
            cache_costs=full_eval_costs,
            n_cores=args.n_proc,
        )
        full_gaps = {cid: _gap_pct(c, full_mean_opt) for cid, c in sub_means.items()}

        # Full-pool ranking by AVERAGE performance (mean cost, lower better) -> rank weights
        # via the same compute_rank_weights used on the partial side (no Friedman).
        full_scores, full_order = compute_full_pool_mean_ranking(
            cand_ids=cand_ids,
            cand_means=sub_means,
            src_by_id=src_by_id,
            cost_tol=args.cost_tol,
        )

        dict_partial_cost[slice_label] = partial_costs
        dict_partial_gap[slice_label] = partial_gaps
        dict_full_cost[slice_label] = sub_means
        dict_full_gap[slice_label] = full_gaps
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
        "problem_size": args.problem_size,
        "seed": args.seed,
        "mean_opt": full_mean_opt,
        "total_unique_candidates": len(full_eval_means),
        "mean_costs": full_eval_means,
        "per_instance_costs": full_eval_costs,
        "by_checkpoint_cost": dict_full_cost,
        "by_checkpoint_gap": dict_full_gap,
    }
    with open(full_pool_file, "w") as f:
        json.dump(full_pool_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved full-pool mean costs -> {full_pool_file}")

    # B) Combined ranking checkpoints evaluation JSON file
    print(f"Saving combined ranking checkpoint evaluations to {out_file}...", flush=True)
    output_data = {
        "exp": str(exp_path),
        "mode": "uniform",
        "problem": "tsp_gls",
        "metric": "rank_weight (higher is better; in (0, 1], 1.0 = top-1 candidate)",
        "slices": slices,
        "n_instances_full": args.n_instances,
        "problem_size": args.problem_size,
        "seed": args.seed,
        "train_mean_opt": train_mean_opt,
        "full_mean_opt": full_mean_opt,
        "total_unique_candidates": len(full_eval_means),
        "checkpoints": checkpoint_meta,
        "partial_eval": dict_partial_eval,
        "partial_eval_cost": dict_partial_cost,
        "partial_eval_gap_pct": dict_partial_gap,
        "full_eval": dict_full_eval,
        "full_eval_cost": dict_full_cost,
        "full_eval_gap_pct": dict_full_gap,
    }
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved combined data -> {out_file}")

    # Summary verification
    print("\n" + "=" * 70)
    print("Verification Summary:")
    for slice_frac in slices:
        lbl = f"{int(round(slice_frac * 100))}%"
        p_cost = dict_partial_cost[lbl]
        f_cost = dict_full_cost[lbl]
        assert set(p_cost.keys()) == set(f_cost.keys()), f"Key mismatch in {lbl}"
        print(f"  [{lbl}] Candidates: {len(p_cost):2d} | 1:1 ID Alignment: TRUE")
        sample_cid = list(p_cost.keys())[0]
        pg = dict_partial_gap[lbl][sample_cid]
        fg = dict_full_gap[lbl][sample_cid]
        pg_s = f"{pg:.3f}%" if pg is not None else "n/a"
        fg_s = f"{fg:.3f}%" if fg is not None else "n/a"
        print(f"       Sample {sample_cid}: Partial cost={p_cost[sample_cid]:.4f} (gap={pg_s}), "
              f"Full cost={f_cost[sample_cid]:.4f} (gap={fg_s})")

    print(f"\nTotal elapsed time: {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
