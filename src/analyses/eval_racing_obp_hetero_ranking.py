"""Evaluate checkpoint-sliced candidates from race_log.jsonl on HETEROGENEOUS OBP:
partial evaluation (rank-weighted values from race_log.jsonl settled candidate ranking)
vs full-pool (128 pre-generated instances) Friedman mean-rank ordering for ranking
correlation analysis.

The heterogeneous-OBP counterpart of eval_racing_obp_ranking.py (and the racing counterpart
of eval_uniform_obp_hetero_ranking.py). Differs from the homogeneous racing-OBP script in the
full instance pool and the full-pool metric:

  * the full pool is LOADED from a pre-generated pickle (``--data-file``, default the run's
    args.yaml ``data_file`` if present, else the EoH-S training set of 128 instances), not
    generated from (n_items, capacity, seed);
  * eval_obp_hetero.batch_scoring_parallel returns THREE full-pool quantities per candidate:
    mean relative gap, mean BINS, and per-instance GAPS. All full-pool candidates share the
    same 128 instances with fixed per-instance lb, so per-instance gap = (bins - lb)/lb is a
    strictly monotonic transform of per-instance bins -> the Friedman ranks are IDENTICAL
    whether computed on gaps or bins. We feed the per-instance GAPS (the only per-instance
    vector eval_obp_hetero exposes) into the Friedman ranking, tie-broken by mean bins.
  * to keep the partial and full sides in the SAME unit, ``full_mean_costs`` stores mean BINS
    (matching the race_log partial side, whose mean_cost / per_instance / survivor common-block
    are raw BINS). ``full_mean_gaps`` / ``full_mean_bins`` are also emitted for completeness.

This script:
  1. loads all race objects from race_log.jsonl;
  2. picks 5 checkpoint races corresponding to budget checkpoints [0.2, 0.4, 0.6, 0.8, 1.0]
     of the final used_budget in phase_after;
  3. for each checkpoint race, takes the phase_after candidate list, filters out crashed / penalty
     candidates (mean_cost is None or >= 1e5), and assigns normalized rank weights based on
     the race's settled ranking order: weight = (N - idx) / N, with DSU identical candidate grouping;
  4. evaluates the exact same candidates on the full 128-instance pool (mean gap, mean bins,
     per-instance gaps), caching across checkpoints;
  5. computes the full-pool Friedman mean-rank ordering across the 128 instances on the
     per-instance gaps (== bins ranking), tie-broken by mean bins;
  6. saves full_pool_eval.json and ranking_eval_checkpoints.json.

Usage:
    python src/analyses/eval_racing_obp_hetero_ranking.py --exp .logs/racing_eoh_obp_hetero_vllm/... --n-proc 20
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

from analyses.eval_obp_hetero import (
    load_training_instances,
    batch_scoring_parallel,
    _to_serialisable,
    DEFAULT_DATA_FILE,
)
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

    (race_log mean_cost is the raw BIN count for hetero-OBP; the DSU grouping and rank order use
    it only as an ordering key, so the unit does not affect the produced rank weights.)
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
    lbs: list,
    cache_means: dict[str, float],
    cache_bins: dict[str, float],
    cache_costs: dict[str, list[float]],
    n_cores: int = 20,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
    """Evaluate candidates on the full instance pool (mean relative gap, mean bins, and
    per-instance gaps) with caching across checkpoints, using batch_scoring_parallel from
    analyses.eval_obp_hetero.

    Returns ({cand_id: mean_gap}, {cand_id: mean_bins}, {cand_id: [gap_0, ..., gap_127]}).
    """
    pending_ids = [cid for cid in cand_ids if cid not in cache_costs]
    if pending_ids:
        heuristics_to_eval = [
            {"cand_id": cid, "source": src_by_id[cid]} for cid in pending_ids
        ]
        mean_gaps, mean_bins, per_inst = batch_scoring_parallel(
            instances, lbs, heuristics_to_eval, n_cores=n_cores, return_per_instance=True
        )
        for cid, gap, mbins, inst_gaps in zip(pending_ids, mean_gaps, mean_bins, per_inst):
            cache_means[cid] = float(gap)
            cache_bins[cid] = float(mbins)
            cache_costs[cid] = [float(g) for g in inst_gaps]

    return (
        {cid: cache_means[cid] for cid in cand_ids},
        {cid: cache_bins[cid] for cid in cand_ids},
        {cid: cache_costs[cid] for cid in cand_ids},
    )


def compute_full_pool_friedman_ranking(
    cand_ids: list[str],
    cand_costs: dict[str, list[float]],
    cand_means: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Compute Friedman mean-rank ordering on the full 128-instance pool matching
    AdaEva-R's racing ranking mechanism.

    For the checkpoint's candidate roster:
      1. Assemble (N x 128) matrix of per-instance GAPS.
      2. For each instance j:
         Compute ranks using rankdata(costs[:, j], method='average')
         (lower gap = fewer excess bins = Rank 1, best; ties receive average rank).
         (Per-instance gap is monotonic in per-instance bins for a fixed instance lb, so this
         is identical to ranking per-instance bins.)
      3. Compute Friedman mean rank across the 128 instances: mean_rank = np.mean(rank_mat, axis=1).
      4. Sort candidates by (mean_rank, mean_cost) ascending — cand_means here is mean BINS,
         so ties break by lower mean bins.
      5. Assign normalized full-pool rank weights:
         score = (N + 1 - rank_of_candidate) / N
         (so Rank 1 gets 1.0, lowest gets 1/N, and ties share identical average weights).

    Returns:
      - cand_friedman_ranks: {cid: float mean rank}
      - full_rank_scores: {cid: float normalized rank weight in (0, 1]}
      - full_order: [cid in best-to-worst Friedman order]
    """
    N = len(cand_ids)
    cost_mat = np.array([cand_costs[cid] for cid in cand_ids], dtype=float)  # shape: (N, 128)
    rank_mat = np.zeros_like(cost_mat)
    for j in range(cost_mat.shape[1]):
        rank_mat[:, j] = rankdata(cost_mat[:, j], method="average")

    mean_ranks = np.mean(rank_mat, axis=1)  # shape: (N,)
    cand_friedman_ranks = {cid: float(mean_ranks[i]) for i, cid in enumerate(cand_ids)}

    # Best-to-worst ordering on full pool (tie-broken by lower mean bins)
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
        description="Evaluate checkpoint-sliced candidates from race_log.jsonl on HETERO-OBP: "
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
        "--data-file",
        type=str,
        default=None,
        help="Pickle of pre-generated heterogeneous OBP instances for the full pool. "
             "Default: the run's args.yaml 'data_file' if present, else the EoH-S training set "
             f"({DEFAULT_DATA_FILE.name}).",
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

    # Resolve the full-pool data file: --data-file > run's args.yaml 'data_file' > default.
    data_file = args.data_file
    if data_file is None:
        args_yaml = exp_path / "args.yaml"
        if args_yaml.exists():
            try:
                import yaml
                a = yaml.safe_load(open(args_yaml)) or {}
                if a.get("data_file"):
                    data_file = a["data_file"]
            except Exception as e:
                print(f"  Warning: could not read data_file from {args_yaml}: {e}")
    if data_file is None:
        data_file = str(DEFAULT_DATA_FILE)
    data_file = pathlib.Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Full-pool data file does not exist: {data_file}")

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
    print("HETERO-OBP Ranking Evaluation: Race-Settled Rank vs Full-Pool Friedman Mean Rank")
    print(f"Experiment: {exp_path}")
    print(f"Cores: {args.n_proc} | Metric: mean bins (lower is better)")
    print(f"Full-pool data file: {data_file}")
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

    # 2. Load full-pool instances (pre-generated heterogeneous set + per-instance lb) & resume cache
    print("\n[Step 2/4] Loading full-pool heterogeneous OBP instances...", flush=True)
    full_instances, lbs, mean_lb = load_training_instances(data_file)
    print(f"  Full pool ready: {len(full_instances)} instances (mean lb (round) = {mean_lb:.4f})")

    # Resume from existing full_pool_eval.json if the data file matches. A resumable entry
    # needs mean_gap, mean_bins, and per-instance gaps (all three are re-saved below).
    full_eval_means: dict[str, float] = {}
    full_eval_bins: dict[str, float] = {}
    full_eval_costs: dict[str, list[float]] = {}
    if full_pool_file.exists():
        try:
            with open(full_pool_file) as f:
                prev_full = json.load(f)
            if prev_full.get("data_file") == str(data_file):
                prev_gaps = prev_full.get("mean_gaps", {})
                prev_bins = prev_full.get("mean_bins", {})
                prev_per_inst = prev_full.get("per_instance_costs", {})
                valid_resumed = {
                    k: float(v) for k, v in prev_gaps.items()
                    if v is not None and k in prev_bins and prev_bins[k] is not None
                    and k in prev_per_inst and prev_per_inst[k] is not None
                }
                full_eval_means.update(valid_resumed)
                full_eval_bins.update({k: float(prev_bins[k]) for k in valid_resumed})
                full_eval_costs.update({
                    k: [float(x) for x in prev_per_inst[k]]
                    for k in valid_resumed
                })
                print(f"  Resumed {len(full_eval_means)} candidate mean gaps/bins and "
                      f"{len(full_eval_costs)} per-instance gap vectors from {full_pool_file.name}")
            else:
                print(f"  NOTE: {full_pool_file.name} was built with a different data_file "
                      f"({prev_full.get('data_file')} != {data_file}); ignoring its cache (re-evaluating).")
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

        # Evaluate the exact same candidates on the full 128-instance pool
        # (mean relative gap, mean bins, per-instance gaps).
        sub_means, sub_bins, sub_costs = evaluate_candidates_on_full_pool(
            cand_ids=cand_ids,
            src_by_id=src_by_id,
            instances=full_instances,
            lbs=lbs,
            cache_means=full_eval_means,
            cache_bins=full_eval_bins,
            cache_costs=full_eval_costs,
            n_cores=args.n_proc,
        )

        # Full-pool Friedman mean ranking on per-instance GAPS (== bins ranking; tie-break by
        # mean bins). full_mean_costs is stored as mean BINS to match the partial (bins) side.
        cand_friedman_ranks, full_scores, full_order = compute_full_pool_friedman_ranking(
            cand_ids=cand_ids,
            cand_costs=sub_costs,     # per-instance gaps
            cand_means=sub_bins,      # tie-break by mean bins
        )

        # Survivor-common-block partial mean cost (AdaEva-R Top-1 metric source) — INDEPENDENT
        # of the rank-weight partial_eval fields above; does NOT interfere with them. For
        # hetero-OBP the race_log per_instance / mean_cost are raw BINS, so this is a bins mean
        # over the survivors' common block — the SAME unit as full_mean_costs below.
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
            "full_mean_costs": sub_bins,        # mean BINS (matches partial-side bins unit)
            "full_mean_bins": sub_bins,         # explicit alias
            "full_mean_gaps": sub_means,        # mean relative gap (for completeness)
            "partial_mean_costs": survivor_partial_means,   # survivor common-block mean BINS
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
        "metric": "mean_bins (lower is better)",
        "data_file": str(data_file),
        "n_instances": len(full_instances),
        "mean_lb": mean_lb,
        "total_unique_candidates": len(full_eval_means),
        "mean_gaps": full_eval_means,
        "mean_bins": full_eval_bins,
        "per_instance_costs": full_eval_costs,
    }
    with open(full_pool_file, "w") as f:
        json.dump(full_pool_data, f, indent=2, default=_to_serialisable)
    print(f"Successfully saved full-pool mean gaps/bins and per-instance gaps -> {full_pool_file}")

    # B) Combined ranking checkpoints evaluation JSON file
    print(f"Saving combined ranking checkpoint evaluations to {out_file}...", flush=True)
    output_data = {
        "exp": str(exp_path),
        "mode": "racing",
        "problem": "obp_hetero",
        "metric": "rank_weight (higher is better; in (0, 1], 1.0 = top-1 candidate)",
        "slices": slices,
        "data_file": str(data_file),
        "n_instances_full": len(full_instances),
        "mean_lb": mean_lb,
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
