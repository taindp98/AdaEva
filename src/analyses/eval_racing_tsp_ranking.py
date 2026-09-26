"""Evaluate checkpoint-sliced candidates from race_log.jsonl on TSP-GLS:
partial evaluation (rank-weighted values from race_log.jsonl settled candidate ranking)
vs full-pool (64 instances) Friedman mean-rank ordering for ranking correlation analysis.

Full parity counterpart of eval_racing_fssp_ranking.py for TSP-GLS racing runs.

This script:
  1. loads all race objects from race_log.jsonl;
  2. picks 5 checkpoint races corresponding to budget checkpoints [0.2, 0.4, 0.6, 0.8, 1.0]
     of the final used_budget in phase_after;
  3. for each checkpoint race, takes the phase_after candidate list, filters out crashed / penalty
     candidates (mean_cost is None or >= 1e5), and assigns normalized rank weights based on
     the race's settled ranking order: weight = (N - idx) / N, with DSU identical candidate grouping;
  4. evaluates the exact same candidates on the full instance pool (64 TSP instances), caching
     per-instance tour costs across checkpoints;
  5. computes the full-pool Friedman mean-rank ordering across the 64 instances matching
     AdaEva-R's racing mechanism (rankdata(cost, method='average') on each instance, then mean
     rank across instances, tie-broken by mean tour cost);
  6. saves full_pool_eval.json and ranking_eval_checkpoints.json.

TSP full-pool scoring is EXPENSIVE (Concorde optima + GLS per instance), so caching by
cand_id and resuming full_pool_eval.json matter.

Usage:
    python src/analyses/eval_racing_tsp_ranking.py --exp .logs/racing_eoh_tsp_gls_vllm/... --n-proc 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import numpy as np
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD")]

from analyses.eval_tsp import load_instances, batch_scoring_parallel, _to_serialisable
from analyses._ranking_common import compute_survivor_partial_mean_costs

PENALTY_COST = 1e5


def load_race_log_entries(exp_path: pathlib.Path) -> list[dict]:
    """Load all race entries from race_log.jsonl inside exp_path."""
    race_log_file = exp_path / "race_log.jsonl"
    if not race_log_file.exists():
        raise FileNotFoundError(f"Missing race_log.jsonl at: {race_log_file}")
    with open(race_log_file) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        raise ValueError(f"race_log.jsonl at {race_log_file} is empty.")
    return entries


def get_sliced_checkpoints(
    entries: list[dict], slices: list[float] = (0.2, 0.4, 0.6, 0.8, 1.0)
) -> list[tuple[str, float, dict]]:
    """Pick 5 race objects in race_log.jsonl corresponding to budget checkpoints
    [0.2, 0.4, 0.6, 0.8, 1.0] of the final budget.
    """
    final_budget = entries[-1]["phase_after"]["used_budget"]
    selected = []
    for s in slices:
        target = s * final_budget
        closest = min(entries, key=lambda e: abs(e["phase_after"]["used_budget"] - target))
        label = f"{int(round(s * 100))}%"
        selected.append((label, s, closest))
    return selected


def compute_partial_eval(
    race_entry: dict,
    src_by_id: dict[str, str] | None = None,
    cost_tol: float = 1e-8,
    penalty_cost: float = PENALTY_COST,
) -> tuple[list[str], dict[str, float]]:
    """Take the phase_after candidate list, drop crashed candidates (mean_cost is None or
    >= penalty_cost), preserving the race's own settled ranking order.

    Assign each valid candidate a normalized rank-based weighted value:
        weight = (N - idx) / N
    where index 0 (the race's top-1 candidate) receives 1.0 (highest value), and the lowest-ranked
    candidate receives 1/N.

    Identical Candidate Rules (shared rank score):
      1. Source Code Check: Candidates sharing identical source code in src_by_id.
      2. Mean Cost Check: Candidates whose difference in mean_cost is less than cost_tol (1e-8).
    All candidates identified as identical share the identical average rank score of their positions:
        avg_idx = mean(indices in race ranking)
        weight = (N - avg_idx) / N
    """
    cands = race_entry["phase_after"]["candidates"]
    valid = [
        c for c in cands
        if c.get("mean_cost") is not None and c["mean_cost"] < penalty_cost
    ]
    if not valid:
        raise ValueError(
            f"No valid candidates found in race_idx {race_entry.get('race_idx')}"
        )
    N = len(valid)
    ordered_cand_ids = [c["cand_id"] for c in valid]

    # Disjoint Set Union (Union-Find) over candidate indices 0..N-1
    parent = list(range(N))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Rule 1: Heuristic source code identity check
    if src_by_id is not None:
        code_to_idx: dict[str, int] = {}
        for idx, c in enumerate(valid):
            cid = c["cand_id"]
            code_key = src_by_id.get(cid, cid).strip()
            if code_key in code_to_idx:
                union(code_to_idx[code_key], idx)
            else:
                code_to_idx[code_key] = idx

    # Rule 2: Mean cost difference check (< cost_tol)
    for i in range(N):
        ci = valid[i].get("mean_cost")
        if ci is None:
            continue
        for j in range(i + 1, N):
            cj = valid[j].get("mean_cost")
            if cj is None:
                continue
            if abs(ci - cj) < cost_tol:
                union(i, j)

    # Collect grouped indices by root
    groups: dict[int, list[int]] = {}
    for idx in range(N):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    # Assign each candidate its group's shared average rank weight
    partial_scores: dict[str, float] = {}
    for idx, c in enumerate(valid):
        cid = c["cand_id"]
        idxs = groups[find(idx)]
        avg_idx = float(sum(idxs)) / float(len(idxs))
        partial_scores[cid] = float(N - avg_idx) / float(N)

    return ordered_cand_ids, partial_scores


def evaluate_candidates_on_full_pool(
    cand_ids: list[str],
    src_by_id: dict[str, str],
    instances: list,
    cache_means: dict[str, float],
    cache_costs: dict[str, list[float]],
    n_cores: int = 20,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Evaluate candidates on the full instance pool with caching across checkpoints using
    batch_scoring_parallel from analyses.eval_tsp.

    Returns ({cand_id: mean_cost}, {cand_id: [cost_0, ..., cost_63]}).
    """
    pending_ids = [cid for cid in cand_ids if cid not in cache_costs]
    if pending_ids:
        heuristics_to_eval = [
            {"cand_id": cid, "source": src_by_id[cid]} for cid in pending_ids
        ]
        # eval_tsp.batch_scoring_parallel ALWAYS returns per-instance costs (no flag).
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


def compute_full_pool_friedman_ranking(
    cand_ids: list[str],
    cand_costs: dict[str, list[float]],
    cand_means: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Compute Friedman mean-rank ordering on the full 64-instance pool matching
    AdaEva-R's racing ranking mechanism.

    For the checkpoint's candidate roster:
      1. Assemble (N x 64) matrix of per-instance tour costs.
      2. For each instance j in 0..63:
         Compute ranks using rankdata(costs[:, j], method='average')
         (lower tour cost = Rank 1, best; ties receive average rank).
      3. Compute Friedman mean rank across the 64 instances: mean_rank = np.mean(rank_mat, axis=1).
      4. Sort candidates by (mean_rank, mean_cost) ascending.
      5. Assign normalized full-pool rank weights:
         score = (N + 1 - rank_of_candidate) / N
         (so Rank 1 gets 1.0, lowest gets 1/N, and ties share identical average weights).

    Returns:
      - cand_friedman_ranks: {cid: float mean rank}
      - full_rank_scores: {cid: float normalized rank weight in (0, 1]}
      - full_order: [cid in best-to-worst Friedman order]
    """
    N = len(cand_ids)
    cost_mat = np.array([cand_costs[cid] for cid in cand_ids], dtype=float)  # shape: (N, 64)
    rank_mat = np.zeros_like(cost_mat)
    for j in range(cost_mat.shape[1]):
        rank_mat[:, j] = rankdata(cost_mat[:, j], method="average")

    mean_ranks = np.mean(rank_mat, axis=1)  # shape: (N,)
    cand_friedman_ranks = {cid: float(mean_ranks[i]) for i, cid in enumerate(cand_ids)}

    # Best-to-worst ordering on full pool (tie-broken by lower mean tour cost)
    full_order = sorted(cand_ids, key=lambda cid: (cand_friedman_ranks[cid], cand_means[cid]))

    # Rank positions (1 to N) with average tie-breaking based on Friedman mean ranks
    ranks_1_to_n = rankdata(mean_ranks, method="average")
    full_rank_scores = {
        cid: float(N + 1 - ranks_1_to_n[i]) / float(N)
        for i, cid in enumerate(cand_ids)
    }

    return cand_friedman_ranks, full_rank_scores, full_order


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint-sliced candidates from race_log.jsonl on TSP-GLS: "
                    "race-settled rank weights vs full-pool Friedman mean-rank weights."
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment directory containing race_log.jsonl and heuristics.json.",
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
        "--cost-tol",
        type=float,
        default=1e-8,
        help="Cost difference threshold below which candidates are treated as identical (default: 1e-8).",
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
        help="Output JSON file for full-pool evaluations (default: <exp>/full_pool_eval.json).",
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
    print("TSP-GLS Ranking Evaluation: Race-Settled Rank vs Full-Pool Friedman Mean Rank")
    print(f"Experiment: {exp_path}")
    print(f"Cores: {args.n_proc} | Instances: {args.n_instances} "
          f"(problem_size={args.problem_size}, seed={args.seed})")
    print(f"Full-pool output: {full_pool_file}")
    print(f"Combined output:  {out_file}")
    print("=" * 70)

    t_start = time.time()
    # 1. Load heuristics and race records; map budget slices -> checkpoints
    print("\n[Step 1/4] Loading race records and heuristics...", flush=True)
    with open(heuristics_path) as f:
        hdata = json.load(f)
    src_by_id = {h["cand_id"]: h["source"] for h in hdata.get("heuristics", [])}
    print(f"  Loaded {len(src_by_id)} candidate heuristic sources.")

    entries = load_race_log_entries(exp_path)
    slices = [0.2, 0.4, 0.6, 0.8, 1.0]
    checkpoints = get_sliced_checkpoints(entries, slices=slices)
    final_budget = entries[-1]["phase_after"]["used_budget"]
    print(f"  Found {len(entries)} race records. Final budget: {final_budget}")

    if args.dry_run:
        print("\n[Dry Run] Verifying checkpoints and heuristics IDs...", flush=True)
        for slice_label, slice_frac, race_obj in checkpoints:
            used_b = race_obj["phase_after"]["used_budget"]
            race_idx = race_obj.get("race_idx")
            gen_id = race_obj.get("gen_id")

            cand_ids, partial_scores = compute_partial_eval(
                race_obj, src_by_id=src_by_id, cost_tol=args.cost_tol, penalty_cost=PENALTY_COST
            )
            missing = [cid for cid in cand_ids if cid not in src_by_id]
            if missing:
                raise KeyError(f"Candidates {missing} from race_idx {race_idx} not found in heuristics.json")

            min_w = min(partial_scores.values())
            max_w = max(partial_scores.values())
            print(
                f"  Checkpoint {slice_label:>4s} (race_idx={race_idx:2d}, gen_id={gen_id if gen_id is not None else -1:2d}, "
                f"budget={used_b:6d}): {len(cand_ids):2d} valid candidates, "
                f"partial RankWeight range: [{min_w:.4f}, {max_w:.4f}]",
                flush=True,
            )
        print("\nDry run completed successfully! All candidates verified against heuristics.json.")
        return

    # 2. Load full-pool instances (Concorde optima) & resume cache
    print("\n[Step 2/4] Generating / loading full-pool TSP instances (Concorde optima)...", flush=True)
    full_mean_opt, full_instances = load_instances(
        n_instances=args.n_instances,
        problem_size=args.problem_size,
        seed=args.seed,
    )
    print(f"  Full pool ready: {len(full_instances)} instances (mean_opt = {full_mean_opt:.4f})")

    # Resume from existing full_pool_eval.json if available
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
    print("\n[Step 3/4] Evaluating sliced checkpoints (settled partial rank vs full-pool Friedman rank)...", flush=True)
    dict_partial_eval: dict[str, dict[str, float]] = {}
    dict_full_eval: dict[str, dict[str, float]] = {}
    checkpoint_meta: dict[str, dict] = {}

    for slice_label, slice_frac, race_obj in checkpoints:
        t_cp = time.time()
        used_b = race_obj["phase_after"]["used_budget"]
        race_idx = race_obj["race_idx"]
        gen_id = race_obj["gen_id"]

        cand_ids, partial_scores = compute_partial_eval(
            race_obj, src_by_id=src_by_id, cost_tol=args.cost_tol, penalty_cost=PENALTY_COST
        )

        missing = [cid for cid in cand_ids if cid not in src_by_id]
        if missing:
            raise KeyError(f"Candidates {missing} from race_idx {race_idx} not found in heuristics.json")

        # Evaluate the exact same candidates on the full instance pool
        sub_means, sub_costs = evaluate_candidates_on_full_pool(
            cand_ids=cand_ids,
            src_by_id=src_by_id,
            instances=full_instances,
            cache_means=full_eval_means,
            cache_costs=full_eval_costs,
            n_cores=args.n_proc,
        )

        # Compute full-pool Friedman mean ranking
        cand_friedman_ranks, full_scores, full_order = compute_full_pool_friedman_ranking(
            cand_ids=cand_ids,
            cand_costs=sub_costs,
            cand_means=sub_means,
        )

        # Survivor-common-block partial mean cost (AdaEva-R Top-1 metric source) — INDEPENDENT
        # of the rank-weight partial_eval fields above; does NOT interfere with them.
        survivor_partial_means, survivor_partial_ranked, n_common_partial = \
            compute_survivor_partial_mean_costs(race_obj, penalty_cost=PENALTY_COST)

        dict_partial_eval[slice_label] = partial_scores
        dict_full_eval[slice_label] = full_scores
        checkpoint_meta[slice_label] = {
            "slice_frac": slice_frac,
            "race_idx": race_idx,
            "gen_id": gen_id,
            "used_budget": used_b,
            "n_candidates": len(cand_ids),
            "candidates_ranked_race": cand_ids,
            "candidates_ranked_full": full_order,
            "full_friedman_mean_ranks": cand_friedman_ranks,
            "full_mean_costs": sub_means,
            "partial_mean_costs": survivor_partial_means,
            "candidates_ranked_partial_mean": survivor_partial_ranked,
            "n_common_partial_mean": n_common_partial,
        }

        print(
            f"  Checkpoint {slice_label:>4s} (race_idx={race_idx:2d}, gen_id={gen_id:2d}, budget={used_b:5d}): "
            f"{len(cand_ids):2d} candidates (unique evaluated: {len(full_eval_means)}, step time: {time.time()-t_cp:.1f}s)",
            flush=True,
        )

    # 4. Save results to JSON
    # A) Dedicated full-pool evaluation JSON file in exp_path
    print(f"\n[Step 4/4] Saving full-pool evaluations to {full_pool_file}...", flush=True)
    full_pool_data = {
        "exp": str(exp_path),
        "metric": "mean_tour_cost (lower is better)",
        "n_instances": args.n_instances,
        "problem_size": args.problem_size,
        "seed": args.seed,
        "mean_opt": full_mean_opt,
        "total_unique_candidates": len(full_eval_means),
        "mean_costs": full_eval_means,
        "per_instance_costs": full_eval_costs,
    }
    with open(full_pool_file, "w") as f:
        json.dump(full_pool_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved full-pool mean tour costs and per-instance costs -> {full_pool_file}")

    # B) Combined ranking checkpoints evaluation JSON file
    print(f"Saving combined ranking checkpoint evaluations to {out_file}...", flush=True)
    output_data = {
        "exp": str(exp_path),
        "mode": "racing",
        "problem": "tsp_gls",
        "metric": "rank_weight (higher is better; in (0, 1], 1.0 = top-1 candidate)",
        "slices": slices,
        "n_instances_full": args.n_instances,
        "problem_size": args.problem_size,
        "seed": args.seed,
        "mean_opt": full_mean_opt,
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
        assert list(p_dict.keys()) == list(f_dict.keys()), f"Key mismatch in {lbl}"
        print(f"  [{lbl}] Candidates: {len(p_dict):2d} | 1:1 ID Alignment: TRUE")
        sample_cid = list(p_dict.keys())[0]
        print(f"       Sample {sample_cid}: Partial RankWeight={p_dict[sample_cid]:.4f}, Full Friedman Weight={f_dict[sample_cid]:.4f}")

    print(f"\nTotal elapsed time: {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
