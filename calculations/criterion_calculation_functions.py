from scipy.stats import chi2, t, binom

from calculations.laplace_function import *
from calculations.constants import *


@lru_cache
def get_chi2(alpha: np.float64, k: int) -> np.float64:
    """
    Возвращает критическое значение хи-квадрат распределения для заданного уровня значимости alpha и степеней свободы k.
    :param alpha: уровень значимости
    :param k: число степеней свободы
    """
    return chi2.ppf(1 - alpha, k)


@lru_cache
def calculate_students_coef(q: np.float64, n: int) -> np.float64:
    """
    Вычисляет коэффициент Стьюдента для доверительного интервала с уровнем значимости q и количеством наблюдений n.
    :param q: уровень значимости
    :param n: количество наблюдений
    """
    return t.ppf(1 - q / 2, n - 1)


def trusted_intervals(samples: np.array, q: np.float64) -> list[tuple]:
    """
    Возвращает доверительные интервалы для среднего значения и стандартного отклонения на основе выборки samples с уровнем значимости q
    :param samples: выборка чисел
    :param q: уровень значимости
    """
    x_mean = np.mean(samples)
    s = np.sqrt(np.mean(np.power(samples, 2)) - x_mean ** 2)
    n = samples.size
    return [loc_trust_interval(x_mean, q, s, n), std_trust_interval(q, n, s)]


def loc_trust_interval(x_mean: np.float64, q: np.float64, s: np.float64, n: int) -> tuple[np.float64, np.float64]:
    """
    Вычисляет доверительный интервал для мат. ожидания на основе среднего x_mean, среднего квадратического отклонения s, количества наблюдений n и уровня значимости q
    :param x_mean: среднее значение
    :param q: уровень значимости
    :param s: среднее квадратическое отклонение
    :param n: количество наблюдений
    """
    return (np.round(x_mean - calculate_students_coef(q, n) * s / np.sqrt(n - 1), precision),
            np.round(x_mean + calculate_students_coef(q, n) * s / np.sqrt(n - 1), precision))


def std_trust_interval(q: np.float64, n: int, s: np.float64) -> tuple[np.float64, np.float64]:
    """
    Вычисляет доверительный интервал для стандартного отклонения на основе среднего квадратического отклонения s, количества наблюдений n и уровня значимости q
    :param q: уровень значимости
    :param n: число наблюдений
    :param s: среднее квадратическое отклонение
    """
    return (np.round(np.sqrt(n) * s / np.sqrt(get_chi2(q / 2, n - 1)), precision),
            np.round(np.sqrt(n) * s / np.sqrt(get_chi2(1 - q / 2, n - 1)), precision))


def chi2_value(interval_list: list[tuple], frequencies: np.array, data_sample: np.array, mean_value: np.float64,
               std: np.float64):
    """
    Вычисляет значение хи-квадрат статистики для проверки соответствия наблюдаемых частот ожидаемым частотам.
    :param interval_list: список интервалов
    :param frequencies: частота в интервалах
    :param data_sample: числовая выборка
    :param mean_value: среднее значение
    :param std: среднее квадратическое отклонение
    :return:
    """
    probability_intervals = np.array([probability_in_interval(i, mean_value, std) for i in interval_list])
    return np.sum(np.power(frequencies - data_sample.size * probability_intervals, 2) /
                  (data_sample.size * probability_intervals))


@lru_cache
def probability_in_interval(interval: tuple[np.float64, np.float64], mean_value: np.float64,
                            std: np.float64) -> np.float64:
    """
    Возвращает вероятность попадания случайной величины в заданный интервал на основе среднего значения и среднего квадратического отклонения
    :param interval: интервал в виде кортежа
    :param mean_value: среднее значение
    :param std: среднее квадратическое отклонение
    :return:
    """
    return np.float64(laplace_function(np.float64((interval[1] - mean_value) / std)) -
                      laplace_function(np.float64(((interval[0]) - mean_value) / std)))


@lru_cache
def calculate_sign_checker(alpha, n):
    """
    Вычисляет критическое значение для знакового теста с уровнем значимости alpha и количеством наблюдений n.
    :param alpha: уровень значимости
    :param n: число наблюдений
    """
    return int(binom.ppf(alpha / 2, n, 0.5))
