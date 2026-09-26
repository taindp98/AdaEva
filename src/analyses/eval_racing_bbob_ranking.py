"""Evaluate checkpoint-sliced candidates from race_log.jsonl on BBOB:
partial evaluation (rank-weighted values from race_log.jsonl settled candidate ranking)
vs full-pool (72 BBOB functions) average-performance (mean AOCC) rank ordering for ranking correlation analysis.

The BBOB counterpart of eval_racing_obp_ranking.py / eval_racing_fssp_ranking.py for
LLaMEA-on-BBOB adaptive racing runs (racing/llamea_bbob.py).

This script:
  1. loads all race objects from race_log.jsonl;
  2. picks 5 checkpoint races corresponding to budget checkpoints [0.2, 0.4, 0.6, 0.8, 1.0]
     of the final used_budget in phase_after;
  3. for each checkpoint race, takes the phase_after candidate list, filters out crashed
     candidates (mean_cost is None or >= 0.0), and assigns normalized rank weights based on
     the race's settled ranking order: weight = (N - idx) / N;
  4. evaluates the exact same candidates on the full instance pool (all 72 noiseless
     BBOB functions: 24 fids x 3 iids at dim 5, budget 10,000, 1 rep), caching per-instance
     AOCCs across checkpoints;
  5. computes the full-pool average-performance (mean AOCC) rank ordering across the 72 instances,
     AdaEva-R's racing mechanism (rankdata(-AOCC, method='average') on each instance, then mean
     rank across instances, tie-broken by mean AOCC);
  6. saves full_pool_eval.json and ranking_eval_checkpoints.json.

Usage:
    python src/analyses/eval_racing_bbob_ranking.py \
        --exp .logs/racing_llamea_bbob_vllm/2026-09-11/132109_0_fixinit_tf5_te1_savpop_dwcpenalty_cp0.0_elimit2_nt2_nc20_pmoriginal/ \
        --n-proc 32
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages" / "LLM4AD"), str(ROOT / "packages" / "LLaMEA")]

from analyses.eval_bbob import (
    build_tasks,
    batch_scoring_parallel,
    _to_serialisable,
    _auto_eval_timeout,
    _N_FUNCTIONS,
)
from analyses._ranking_common import compute_survivor_partial_mean_costs


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
) -> tuple[list[str], dict[str, float]]:
    """Take the phase_after candidate list, drop crashed candidates (mean_cost is None or
    >= 0.0, i.e. non-positive AOCC), preserving the race's own settled ranking order.

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
        if c.get("mean_cost") is not None and c["mean_cost"] < 0.0  # crash / non-positive AOCC -> drop
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
    cand_aoccs: dict[str, list[float]],
    cand_means: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Rank candidates on the full 72-instance pool by their AVERAGE PERFORMANCE (mean AOCC),
    replacing the earlier per-instance Friedman mean-rank mechanism. The rank NORMALIZATION
    is unchanged.

    For the checkpoint's candidate roster:
      1. Average performance = ``cand_means[cid]`` = mean AOCC over the 72 tasks (higher is
         better). (No per-instance rank matrix is built; ``cand_aoccs`` is accepted for a
         stable signature but unused.)
      2. Rank candidates by mean AOCC DESCENDING via
         ``ranks_1_to_n = rankdata(-mean_AOCC, method='average')`` (highest AOCC = Rank 1,
         best; ties receive the average rank).
      3. Assign normalized full-pool rank weights (SAME as before):
         ``score = (N + 1 - rank_of_candidate) / N``
         (so Rank 1 gets 1.0, lowest gets 1/N, and ties share identical average weights).

    Returns:
      - cand_mean_perf_ranks: {cid: float rank position by mean AOCC (1 = best)}
      - full_rank_scores: {cid: float normalized rank weight in (0, 1]}
      - full_order: [cid in best-to-worst mean-AOCC order]
    """
    N = len(cand_ids)
    means = np.array([cand_means[cid] for cid in cand_ids], dtype=float)

    # Rank positions (1 to N) by mean AOCC descending (higher AOCC -> lower rank -> better).
    ranks_1_to_n = rankdata(-means, method="average")
    cand_mean_perf_ranks = {cid: float(ranks_1_to_n[i]) for i, cid in enumerate(cand_ids)}

    # Best-to-worst ordering on full pool (tie-broken by higher mean AOCC).
    full_order = sorted(cand_ids, key=lambda cid: (cand_mean_perf_ranks[cid], -cand_means[cid]))

    # Rank normalization UNCHANGED: score = (N + 1 - rank) / N.
    full_rank_scores = {
        cid: float(N + 1 - ranks_1_to_n[i]) / float(N)
        for i, cid in enumerate(cand_ids)
    }

    return cand_mean_perf_ranks, full_rank_scores, full_order


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint-sliced candidates from race_log.jsonl: "
                    "race-settled rank weights vs full-pool average-performance (mean AOCC) rank weights."
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment directory containing race_log.jsonl and heuristics.json.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=5,
        help="Problem dimension (default: 5).",
    )
    parser.add_argument(
        "--budget-factor",
        type=int,
        default=2000,
        help="Per-instance func-eval budget = budget_factor * dim (default: 2000 => 10,000 evals).",
    )
    parser.add_argument(
        "--n-rep",
        type=int,
        default=1,
        help="Repeated seeded runs per (fid, iid) function (default: 1). Total evals/candidate = 72 * n_rep.",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=-1.0,
        help="Per-(candidate, instance) wall-clock cap (s). -1 (default) = auto-scale "
             "(= 60 + budget_factor*dim/100); 0 = disabled; >0 = fixed.",
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
        "--output",
        type=str,
        default=None,
        help="Output combined JSON file (default: <exp>/ranking_eval_checkpoints.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: verify checkpoints and candidates without running full-pool evaluation.",
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
        fallback_path = exp_path.parent / f"{exp_path.name}_pmoriginal"
        if fallback_path.exists():
            print(f"Path '{exp_path}' not found, falling back to '{fallback_path}'.")
            exp_path = fallback_path
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
    print("Racing BBOB Ranking Evaluation: Race-Settled Rank vs Full-Pool Mean-Performance Rank")
    print(f"Experiment: {exp_path}")
    print(f"Cores: {args.n_proc} | Dim: {args.dim} | Budget: {budget} (bf={args.budget_factor})")
    print(f"Repetitions: {args.n_rep} (72 * {args.n_rep} = {72 * args.n_rep} evals/cand)")
    print(f"Timeout per task: {resolved_timeout:.1f}s")
    print(f"Full-pool output: {full_pool_file}")
    print(f"Combined output:  {out_file}")
    print("=" * 70)

    # 1. Build tasks for the full pool (all 72 noiseless BBOB functions: 24 fids x 3 iids)
    t_start = time.time()
    print("\n[Step 1/4] Building full-pool BBOB tasks...", flush=True)
    tasks = build_tasks(args.n_rep)
    print(f"  Full pool tasks ready: {len(tasks)} tasks (72 functions x {args.n_rep} rep)")

    # 2. Load heuristics and race_log.jsonl; map budget slices -> checkpoint races
    print("\n[Step 2/4] Loading race_log.jsonl and heuristics...", flush=True)
    with open(heuristics_path) as f:
        hdata = json.load(f)
    src_by_id = {h["cand_id"]: h["source"] for h in hdata.get("heuristics", [])}
    print(f"  Loaded {len(src_by_id)} sampled heuristics.")

    race_entries = load_race_log_entries(exp_path)
    slices = [0.2, 0.4, 0.6, 0.8, 1.0]
    checkpoints = get_sliced_checkpoints(race_entries, slices=slices)
    final_budget = race_entries[-1]["phase_after"]["used_budget"]
    print(f"  Found {len(race_entries)} race entries. Final budget: {final_budget}")

    # Resume from existing full_pool_eval.json if available
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
                full_eval_means.update({k: float(v) for k, v in prev_scores.items() if v is not None})
                prev_aoccs = prev_full.get("per_instance_aoccs", {})
                full_eval_aoccs.update({k: [float(x) for x in v] for k, v in prev_aoccs.items() if v is not None})
                print(f"  Resumed {len(full_eval_means)} candidate mean AOCCs and "
                      f"{len(full_eval_aoccs)} per-instance AOCC vectors from {full_pool_file.name}")
            else:
                print(f"  NOTE: {full_pool_file.name} was built with different eval params "
                      f"{prev_params} != {cur_params}; ignoring its cache (re-evaluating).")
        except Exception as e:
            print(f"  Warning: failed to read existing {full_pool_file.name}: {e}")

    if args.dry_run:
        print("\n[Dry Run] Verifying checkpoints and heuristics IDs...", flush=True)
        for slice_label, slice_frac, race_entry in checkpoints:
            used_b = race_entry["phase_after"]["used_budget"]
            race_idx = race_entry.get("race_idx")
            gen_id = race_entry.get("gen_id")

            cand_ids, partial_scores = compute_partial_eval(race_entry, src_by_id=src_by_id)
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

    # 3. Evaluate each checkpoint
    print("\n[Step 3/4] Evaluating sliced checkpoints (settled partial rank vs full-pool mean-performance rank)...", flush=True)
    dict_partial_eval: dict[str, dict[str, float]] = {}
    dict_full_eval: dict[str, dict[str, float]] = {}
    checkpoint_meta: dict[str, dict] = {}

    for slice_label, slice_frac, race_entry in checkpoints:
        t_cp = time.time()
        used_b = race_entry["phase_after"]["used_budget"]
        race_idx = race_entry.get("race_idx")
        gen_id = race_entry.get("gen_id")

        cand_ids, partial_scores = compute_partial_eval(race_entry, src_by_id=src_by_id)

        # Check for missing sources
        missing = [cid for cid in cand_ids if cid not in src_by_id]
        if missing:
            raise KeyError(f"Candidates {missing} from race_idx {race_idx} not found in heuristics.json")

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

        # Compute full-pool ranking by AVERAGE PERFORMANCE (mean AOCC), rank normalization
        # unchanged ((N+1-rank)/N).
        cand_mean_perf_ranks, full_scores, full_order = compute_full_pool_mean_ranking(
            cand_ids=cand_ids,
            cand_aoccs=sub_aoccs,
            cand_means=sub_means,
        )

        # Survivor-common-block partial mean cost (AdaEva-R Top-1 metric source) — INDEPENDENT
        # of the rank-weight partial_eval fields above; does NOT interfere with them. BBOB's
        # per_instance/mean_cost are NEGATED AOCC (lower = better), so the valid filter is
        # mean_cost < 0.0 (penalty_cost=0.0) and the ascending rank gives best->worst.
        survivor_partial_means, survivor_partial_ranked, n_common_partial = \
            compute_survivor_partial_mean_costs(race_entry, penalty_cost=0.0)

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
            "full_mean_perf_ranks": cand_mean_perf_ranks,
            "full_mean_aoccs": sub_means,
            # full_mean_costs = NEGATED AOCC (lower = better), matching partial_mean_costs'
            # convention so downstream (notebook Recall@k) can rank both ASCENDING uniformly.
            "full_mean_costs": {cid: -float(sub_means[cid]) for cid in cand_ids},
            "partial_mean_costs": survivor_partial_means,
            "candidates_ranked_partial_mean": survivor_partial_ranked,
            "n_common_partial_mean": n_common_partial,
        }

        print(
            f"  Checkpoint {slice_label:>4s} (race_idx={race_idx:2d}, gen_id={gen_id if gen_id is not None else -1:2d}, "
            f"budget={used_b:6d}): {len(cand_ids):2d} candidates "
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
        "per_instance_aoccs": full_eval_aoccs,
    }
    with open(full_pool_file, "w") as f:
        json.dump(full_pool_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved full-pool mean AOCCs and per-instance AOCCs -> {full_pool_file}")

    # B) Combined ranking checkpoints evaluation JSON file
    print(f"Saving combined ranking checkpoint evaluations to {out_file}...", flush=True)
    output_data = {
        "exp": str(exp_path),
        "mode": "racing",
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
        assert list(p_dict.keys()) == list(f_dict.keys()), f"Key mismatch in {lbl}"
        print(f"  [{lbl}] Candidates: {len(p_dict):2d} | 1:1 ID Alignment: TRUE")
        sample_cid = list(p_dict.keys())[0]
        print(f"       Sample {sample_cid}: Partial RankWeight={p_dict[sample_cid]:.4f}, Full Mean-Rank Weight={f_dict[sample_cid]:.4f}")

    print(f"\nTotal elapsed time: {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
