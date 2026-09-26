import math
from typing import Optional

import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel

try:
    import scikit_posthocs as sp

    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False


def friedman_eliminate(
    rank_mat: np.ndarray, alpha: float = 0.05, posthoc_test_type: str = "conover"
) -> tuple[int, list[bool], Optional[float]]:
    """Rank-based elimination using Friedman's test + post-hoc test.

    Returns ``(best_idx, keep_mask, p_value)`` where ``p_value`` is the Friedman
    omnibus p (``None`` when no omnibus test ran, e.g. < 2 configs/instances).

    Supports two post-hoc test variants:
        - "conover" (default): Original R irace formula from Conover (1999, pp. 369-371).
          Compares each config's rank sum specifically against the best config using
          Student's t-distribution critical difference without all-pairs Tukey penalty.
        - "nemenyi": Pairwise Nemenyi post-hoc test via scikit-posthocs (all-pairs
          Tukey Studentized Range adjustment across all candidate pairs).
    """
    n_configs, n_instances = rank_mat.shape
    sum_ranks = rank_mat.sum(axis=1)
    best_idx = int(np.argmin(sum_ranks))

    if n_configs < 2 or n_instances < 2:
        return best_idx, [True] * n_configs, None

    try:
        stat, p = friedmanchisquare(*[rank_mat[i] for i in range(n_configs)])
    except ValueError:
        return best_idx, [True] * n_configs, None

    if p >= alpha:
        return best_idx, [True] * n_configs, float(p)

    print(f"  [stat-test] friedman omnibus p={p:.4e} < {alpha}: conducting post-hoc test (posthoc_test_type='{posthoc_test_type}')")

    keep = [True] * n_configs

    if posthoc_test_type == "conover":
        from scipy.stats import t as t_dist
        # Formula from Conover (1999), pages 369-371 / R irace race.R:161-165
        A = float(np.sum(rank_mat ** 2))
        R_sq_sum = float(np.sum(sum_ranks ** 2))
        df = (n_instances - 1) * (n_configs - 1)
        if df > 0:
            t_crit = t_dist.ppf(1 - alpha / 2, df=df)
            diff_threshold = t_crit * np.sqrt(max(0.0, 2 * (n_instances * A - R_sq_sum) / df))
            order = np.argsort(sum_ranks)
            keep_set = {best_idx}
            for idx in order:
                if abs(sum_ranks[idx] - sum_ranks[best_idx]) <= diff_threshold:
                    keep_set.add(idx)
                else:
                    break
            keep = [i in keep_set for i in range(n_configs)]
    elif posthoc_test_type == "nemenyi" and HAS_POSTHOC and n_configs >= 3:
        pvals = sp.posthoc_nemenyi_friedman(rank_mat.T).values
        for i in range(n_configs):
            if i == best_idx:
                continue
            if pvals[best_idx, i] < alpha and sum_ranks[i] > sum_ranks[best_idx]:
                keep[i] = False
    else:
        threshold = np.percentile(sum_ranks, 75)
        for i in range(n_configs):
            if i != best_idx and sum_ranks[i] > threshold:
                keep[i] = False

    keep[best_idx] = True
    return best_idx, keep, float(p)


def ttest_eliminate(
    cost_mat: np.ndarray, alpha: float = 0.05
) -> tuple[int, list[bool], Optional[float]]:
    """Mean-cost elimination using a paired t-test of each config vs the best.

    R reference: `packages/irace/R/race.R:255-290` (`aux_ttest`).

    Features:
        - Best = config with the lowest mean cost across observed instances.
        - For every other config, runs `scipy.stats.ttest_rel` against the
          best on the shared instances.
        - Eliminates a config when (a) p < alpha AND (b) its mean cost is
          strictly worse than the best's.
        - The best config is always kept; configs whose difference vector
          is exactly zero are kept (no signal to test).
        - Faithfully handles constant/dominant difference vectors and crash penalties.

    Arguments:
        cost_mat: Cost matrix of shape (n_configs, n_instances), as produced
            by `ranking.cost_matrix`. Lower = better.
        alpha: Significance level for the per-pair t-test.

    Returns:
        (best_idx, keep_mask): Index of the best config and a boolean list
        where keep_mask[i] = True means config i survives.
    """
    n_configs, n_instances = cost_mat.shape
    mean_costs = cost_mat.mean(axis=1)
    best_idx = int(np.argmin(mean_costs))

    if n_configs < 2 or n_instances < 2:
        return best_idx, [True] * n_configs, None

    keep = [True] * n_configs
    best = cost_mat[best_idx]
    min_p: Optional[float] = None                # smallest vs-best p seen (a proxy omnibus)
    for i in range(n_configs):
        if i == best_idx:
            continue
        diff = cost_mat[i] - best
        if np.allclose(diff, 0):
            continue
        try:
            res = ttest_rel(cost_mat[i], best)
            p = float(res.pvalue)
        except Exception:
            p = float("nan")

        # Handle constant/dominant difference vectors where scipy returns NaN
        if not math.isfinite(p):
            if mean_costs[i] > mean_costs[best_idx]:
                if np.all(cost_mat[i] >= best) and np.any(cost_mat[i] > best):
                    p = 0.0
                else:
                    p = 1.0
            else:
                p = 1.0

        if math.isfinite(p):
            min_p = p if min_p is None else min(min_p, p)
        if p < alpha and mean_costs[i] > mean_costs[best_idx]:
            keep[i] = False

    keep[best_idx] = True
    return best_idx, keep, min_p
