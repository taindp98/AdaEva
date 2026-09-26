"""Shared helpers for the uniform-run ranking scripts (eval_uniform_{obp,tsp,fssp}_ranking).

Centralizes the checkpoint-selection-pool reconstruction so all three problems behave
identically: the checkpoint population at generation g is the TRUE environmental-selection
pool EoH's survival() compared — the survivors entering gen g (survivors_by_gen[g+1])
UNION gen g's offspring (heuristics[g]) — not a single generation's offspring alone.

Environmental selection operates:
  survivors_by_gen[g+1] + heuristics[g] -> survivors_by_gen[g+2]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def index_heuristics(heuristics: list[dict]) -> dict[str, dict]:
    """Index heuristics.json by cand_id -> entry (cand_id, gen_id, score, source)."""
    return {h["cand_id"]: h for h in heuristics}


def load_survivors(exp_path: pathlib.Path) -> dict[int, list[str]]:
    """Load per-generation environmental-selection survivor cand_ids from survivors.json
    (written by the tiny/eoh runner's _save_survivors). Returns {gen_id(int) -> [cand_id]}
    (survivors kept AFTER that generation's survival()).

    REQUIRED: older runs predate this log; raise a clear error so the user re-runs the
    experiment rather than silently falling back to a less faithful selection pool."""
    surv_file = exp_path / "survivors.json"
    if not surv_file.exists():
        raise FileNotFoundError(
            f"Missing survivors.json at: {surv_file}\n"
            f"  This run predates the survivor-identity logging. The checkpoint selection "
            f"pool (survivors entering gen + offspring) cannot be reconstructed faithfully — "
            f"re-run the experiment with the updated tiny/eoh runner to produce survivors.json."
        )
    with open(surv_file) as f:
        data = json.load(f)
    out: dict[int, list[str]] = {}
    for gid_str, rows in data.get("survivors_by_gen", {}).items():
        out[int(gid_str)] = [r["cand_id"] for r in rows if r.get("cand_id") is not None]
    return out


def build_selection_pool(
    checkpoint_gen: int,
    by_id: dict[str, dict],
    survivors_by_gen: dict[int, list[str]],
    penalty_cost: float = 1e5,
) -> tuple[list[str], dict[str, float], dict[str, str], list[int]]:
    """Form the TRUE environmental-selection pool at checkpoint_gen = the survivors
    entering that generation UNION that generation's offspring.
    Mirrors EoH's survival():
      survivors_by_gen[checkpoint_gen + 1] + heuristics[checkpoint_gen] -> survivors_by_gen[checkpoint_gen + 2]
    so the ranking is measured over exactly that combined pool, not one generation's offspring alone.

    - offspring(g) = heuristics.json candidates with gen_id == g.
    - survivors entering gen g = survivors_by_gen[g + 1] (keys start at 2; absent for g == 0 -> pool is just gen 0).
    Deduped by cand_id (a carried-over elite appears once, under its ORIGINAL cand_id).
    Each candidate's partial cost = -score (its own stored score, the value survival()
    ranked it by). Penalty/non-finite candidates are dropped.

    Returns (ordered_cand_ids, partial_costs, source_by_id, combined_gens).
    """
    combined_gens = [checkpoint_gen]
    pool_ids: list[str] = []
    seen: set[str] = set()

    # Environmental selection operates:
    # survivors_by_gen[gen_id + 1] + heuristics[gen_id] -> survivors_by_gen[gen_id + 2]
    # Therefore, parents entering checkpoint_gen are logged under key checkpoint_gen + 1.
    entering_key = checkpoint_gen + 1
    if entering_key in survivors_by_gen:
        combined_gens = [checkpoint_gen - 1, checkpoint_gen]
        for cid in survivors_by_gen[entering_key]:
            if cid not in seen:
                seen.add(cid)
                pool_ids.append(cid)

    # Offspring generated in checkpoint_gen
    for cid, h in by_id.items():
        if h.get("gen_id") == checkpoint_gen and cid not in seen:
            seen.add(cid)
            pool_ids.append(cid)

    ordered_cand_ids: list[str] = []
    partial_costs: dict[str, float] = {}
    src_by_id: dict[str, str] = {}
    for cid in pool_ids:
        h = by_id.get(cid)
        if h is None:
            continue
        score = h.get("score")
        if score is None:
            continue
        cost = -float(score)            # heuristics.json stores the NEGATED cost
        if not np.isfinite(cost) or cost >= penalty_cost:
            continue                    # crash / penalty candidate -> drop
        ordered_cand_ids.append(cid)
        partial_costs[cid] = cost
        src_by_id[cid] = h["source"]
    if not ordered_cand_ids:
        raise ValueError(f"No valid (non-penalty) candidates in selection pool for "
                         f"gen {checkpoint_gen} (combined gens {combined_gens})")
    return ordered_cand_ids, partial_costs, src_by_id, combined_gens


def build_selection_pool_bbob(
    checkpoint_gen: int,
    by_id: dict[str, dict],
    survivors_by_gen: dict[int, list[str]],
    min_score: float = 0.0,
) -> tuple[list[str], dict[str, float], dict[str, str], list[int]]:
    """Form the TRUE environmental-selection pool for BBOB (1-indexed LLaMEA runs)
    at checkpoint_gen = the survivors entering that generation UNION that generation's offspring.

    LLaMEA environmental selection operates:
      survivors_by_gen[g - 1] + heuristics[g] -> survivors_by_gen[g] (for g > 1)
      gen 1 initial population -> survivors_by_gen[1] (for g == 1)

    - For g == 1: pool is gen 1 offspring (heuristics[1]).
    - For g > 1: survivors entering gen g are survivors_by_gen[g - 1], combined with offspring heuristics[g].
    Deduped by cand_id.
    Each candidate's partial score = its own stored score (positive AOCC in [0, 1], higher is better).
    Crashed / non-finite / <= min_score candidates are dropped.

    Returns (ordered_cand_ids, partial_scores, source_by_id, combined_gens).
    """
    combined_gens = [checkpoint_gen]
    pool_ids: list[str] = []
    seen: set[str] = set()

    if checkpoint_gen > 1:
        entering_key = checkpoint_gen - 1
        if entering_key in survivors_by_gen:
            combined_gens = [checkpoint_gen - 1, checkpoint_gen]
            for cid in survivors_by_gen[entering_key]:
                if cid not in seen:
                    seen.add(cid)
                    pool_ids.append(cid)

    # Offspring generated in checkpoint_gen
    for cid, h in by_id.items():
        if h.get("gen_id") == checkpoint_gen and cid not in seen:
            seen.add(cid)
            pool_ids.append(cid)

    ordered_cand_ids: list[str] = []
    partial_scores: dict[str, float] = {}
    src_by_id: dict[str, str] = {}
    for cid in pool_ids:
        h = by_id.get(cid)
        if h is None:
            continue
        score = h.get("score")
        if score is None:
            continue
        score_val = float(score)
        if not np.isfinite(score_val) or score_val <= min_score:
            continue                    # crash / non-finite / non-positive candidate -> drop
        ordered_cand_ids.append(cid)
        partial_scores[cid] = score_val
        src_by_id[cid] = h["source"]
    if not ordered_cand_ids:
        raise ValueError(f"No valid (positive AOCC) candidates in selection pool for "
                         f"gen {checkpoint_gen} (combined gens {combined_gens})")
    return ordered_cand_ids, partial_scores, src_by_id, combined_gens


def compute_rank_weights(
    ordered_cand_ids: list[str],
    metric_by_id: dict[str, float],
    src_by_id: dict[str, str] | None = None,
    cost_tol: float = 1e-8,
) -> dict[str, float]:
    """Compute normalized rank weights in (0, 1] for an already-ordered list of candidate IDs,
    where index 0 (top-1) receives 1.0 and index N-1 receives 1/N.

    Identical Candidate Equivalence (shared average rank weight via DSU):
      1. Source Code Check: Candidates sharing identical source code (src_by_id[cid].strip()).
      2. Score/Cost Difference Check: Candidates whose |score_i - score_j| < cost_tol.
    Tied / identical candidates share the average rank index of their group:
      avg_idx = mean(indices)
      weight = (N - avg_idx) / N
    """
    N = len(ordered_cand_ids)
    if N == 0:
        return {}

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
        for idx, cid in enumerate(ordered_cand_ids):
            code = src_by_id.get(cid, "").strip()
            if code:
                if code in code_to_idx:
                    union(idx, code_to_idx[code])
                else:
                    code_to_idx[code] = idx

    # Rule 2: Metric difference tolerance check (|val_i - val_j| < cost_tol)
    for i in range(N):
        cid_i = ordered_cand_ids[i]
        vi = metric_by_id.get(cid_i)
        if vi is None:
            continue
        for j in range(i + 1, N):
            cid_j = ordered_cand_ids[j]
            vj = metric_by_id.get(cid_j)
            if vj is None:
                continue
            if abs(vi - vj) < cost_tol:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(N):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    group_avg_idx = {
        root: float(np.mean(members)) for root, members in groups.items()
    }

    return {
        ordered_cand_ids[idx]: float(N - group_avg_idx[find(idx)]) / float(N)
        for idx in range(N)
    }


def compute_survivor_partial_mean_costs(
    race_entry: dict,
    penalty_cost: float = 1e5,
) -> tuple[dict[str, float], list[str], int]:
    """Partial mean cost over the SURVIVORS' shared common instance block, for the AdaEva-R
    Top-1 metric — INDEPENDENT of the rank-weight partial fields (does NOT touch them).

    Mechanism (user spec):
      1. take the race's ``phase_after`` candidate list;
      2. keep only SURVIVED candidates (``survived == True``) that are valid (mean_cost
         finite and < penalty_cost);
      3. common block = the intersection of the survivors' ``instances`` (1-based idxs);
      4. each survivor's partial mean = mean of its ``per_instance`` values over that common
         block. The stored value is the RAW race_log cost (OBP bins / FSSP makespan /
         BBOB negated-AOCC) — all lower-is-better, so ranking ascending gives best->worst.

    Fallback: if the common block is empty (survivors share no instance, or a single
    survivor with its own set), each survivor falls back to its own ``phase_after``
    mean_cost over its full instance set (never errors).

    Returns:
      - partial_mean_costs: {cid: float mean over common block (or own mean_cost fallback)}
      - ranked_cand_ids: survivor cand_ids sorted best->worst (ascending mean, tie-break cid)
      - n_common: size of the survivor common block (0 when the fallback was used)
    """
    cands = race_entry.get("phase_after", {}).get("candidates", [])
    survivors = [
        c for c in cands
        if c.get("survived") is True
        and c.get("mean_cost") is not None and c["mean_cost"] < penalty_cost
    ]
    if not survivors:
        return {}, [], 0

    inst_sets = [set(c.get("instances", []) or []) for c in survivors]
    common = set.intersection(*inst_sets) if all(inst_sets) else set()
    common_keys = sorted(common)

    partial_mean_costs: dict[str, float] = {}
    for c in survivors:
        cid = c["cand_id"]
        if common_keys:
            per = c.get("per_instance", {}) or {}
            vals = [float(per[str(k)]) for k in common_keys if str(k) in per]
            partial_mean_costs[cid] = float(sum(vals) / len(vals)) if vals else float(c["mean_cost"])
        else:
            # Fallback: survivor's own mean_cost over its full instance set.
            partial_mean_costs[cid] = float(c["mean_cost"])

    ranked_cand_ids = sorted(partial_mean_costs, key=lambda cid: (partial_mean_costs[cid], cid))
    return partial_mean_costs, ranked_cand_ids, len(common_keys)
