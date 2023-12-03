from functools import lru_cache

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


@lru_cache
def laplace_function(x: np.float64) -> np.float64:
    return norm.cdf(x)


@lru_cache
def laplace_reverse(x: np.float64) -> np.float64:
    return norm.ppf(x)


def quantil_solve(data_sample: np.array) -> np.array:
    data_sample = np.array(sorted(data_sample))
    ind1, ind2 = data_sample.size // 4, 3 * data_sample.size // 4
    xp1, xp2 = data_sample[ind1], data_sample[ind2]
    p1, p2 = np.float64(ind1 / data_sample.size), np.float64(ind2 / data_sample.size)

    def quantil_generate(x: np.array) -> list:
        return [(xp1 - x[0]) / x[1] - laplace_reverse(p1), (xp2 - x[0]) / x[1] - laplace_reverse(p2)]

    return fsolve(quantil_generate, np.array([1, 1]))
