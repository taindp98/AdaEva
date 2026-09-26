import numpy as np
from scipy.stats import rankdata


def rank_costs(costs: list[float]) -> list[float]:
    """Convert a list of per-instance costs to ranks (lower cost = lower/better rank).

    Features:
        - Lower cost gets rank 1 (best), highest cost gets rank N (worst).
        - Ties receive the *average* of the ranks they span, matching irace's
          convention and producing unbiased Friedman test inputs.

    Arguments:
        costs: One cost per alive config on a single instance.

    Returns:
        List of float ranks in the same order as `costs`.

    Example:
        >>> rank_costs([10.0, 5.0, 7.0, 5.0])
        [4.0, 1.5, 3.0, 1.5]
    """
    return rankdata(costs, method="average").tolist()


def rank_matrix(configs, n_instances: int) -> np.ndarray:
    """Build a (configs x instances) rank matrix from each config's stored ranks.

    Features:
        - Pulls the last `n_instances` ranks from every config, so callers can
          window the matrix to the instances seen since a particular point.
        - Used as the input to the Friedman test in `tests.friedman_eliminate`.

    Arguments:
        configs: Iterable of `Config` objects, typically the currently-alive set.
        n_instances: How many trailing rank entries to include per row.

    Returns:
        A float ndarray of shape (len(configs), n_instances). Row i =
        configs[i].ranks[-n_instances:].

    Example:
        >>> from utils.config import Config
        >>> a = Config(id='a', params={}, ranks=[1, 1, 2])
        >>> b = Config(id='b', params={}, ranks=[2, 2, 1])
        >>> rank_matrix([a, b], 3).tolist()
        [[1.0, 1.0, 2.0], [2.0, 2.0, 1.0]]
    """
    return np.array([c.ranks[-n_instances:] for c in configs], dtype=float)


def cost_matrix(configs, n_instances: int) -> np.ndarray:
    """Build a (configs x instances) cost matrix from each config's stored costs.

    Features:
        - Symmetric counterpart of `rank_matrix` over raw costs.
        - Used as the input to the paired t-test in `tests.ttest_eliminate`.

    Arguments:
        configs: Iterable of `Config` objects, typically the currently-alive set.
        n_instances: How many trailing cost entries to include per row.

    Returns:
        A float ndarray of shape (len(configs), n_instances).

    Example:
        >>> from utils.config import Config
        >>> a = Config(id='a', params={}, costs=[1.0, 2.0])
        >>> b = Config(id='b', params={}, costs=[3.0, 4.0])
        >>> cost_matrix([a, b], 2).tolist()
        [[1.0, 2.0], [3.0, 4.0]]
    """
    return np.array([c.costs[-n_instances:] for c in configs], dtype=float)
