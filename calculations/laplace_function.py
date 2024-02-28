from functools import lru_cache

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


@lru_cache
def laplace_function(x: np.float64) -> np.float64:
    """
    :param x: np.float64 - аргумент для вычисления функции Лапласа
    :return: np.float64 - значение функции Лапласа
    """
    return norm.cdf(x)


@lru_cache
def laplace_reverse(x: np.float64) -> np.float64:
    """
    :param x: np.float64 - аргумент для вычисления обратной функции Лапласа
    :return: np.float64 - значение обратной функции Лапласа
    """
    return norm.ppf(x)


def quantil_solve(data_sample: np.array) -> np.array:
    """
    Определение оценки параметров закона распределения - математического ожидания и ср. кв. отклонения.
    Используется метод квантилей, составляется два уравнения вида Ф((x_p1 - m) / sigma) + 0.5 = p.
    В функции берутся квартили - элементы с порядковыми номерами n/4 и 3n/4.
    :param data_sample: np.array - выборка
    :return: np.ndarray - вычисленные значения мат. ожидания и ср. кв. отклонения
    """
    data_sample = np.array(sorted(data_sample))
    ind1, ind2 = data_sample.size // 4, 3 * data_sample.size // 4
    xp1, xp2 = data_sample[ind1], data_sample[ind2]
    p1, p2 = np.float64(ind1 / data_sample.size), np.float64(ind2 / data_sample.size)

    def quantile_generate(x: np.array) -> list:
        return [(xp1 - x[0]) / x[1] - laplace_reverse(p1), (xp2 - x[0]) / x[1] - laplace_reverse(p2)]
    return fsolve(quantile_generate, np.array([1, 1]))
