import math
import numpy as np


def _obp_lower_bound(items, capacity) -> int:
    return math.ceil(float(np.sum(items)) / float(capacity))
