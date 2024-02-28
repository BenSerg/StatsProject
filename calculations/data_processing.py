import numpy as np

from calculations.laplace_function import laplace_reverse


def sign_check(data_sample: np.array, count=20) -> int:
    """
    Подсчитывает количество отрицательных и положительных чисел в выборке размером count
    из генеральной совокупности data_sample
    :param data_sample: np.array - генеральная совокупность
    :param count: int - размер выборки
    :return: min(pos.size, neg.size): int
    """
    first_twenty_samples = data_sample[:count]
    last_twenty_samples = data_sample[-count:]
    diffs = first_twenty_samples - last_twenty_samples
    positives = diffs[diffs > 0]
    negatives = diffs[diffs < 0]
    return min(positives.size, negatives.size)


def inversion_check(data_sample: np.array, count=20) -> tuple[np.float64, np.float64, int]:
    """
    Подсчет инверсий для Критерия Вилкоксона
    :param data_sample: np.array - генеральная совокупность
    :param count: int - объем выборки, default=20
    :return: tuple - мат. ожидание числа инверсий, дисперсия числа инверсий, число инверсий
    """
    first_sample = data_sample[:count]
    last_sample = data_sample[-count:]

    def inv_calculate(arr1, arr2):
        arr1 = sorted(arr1)
        total = 0
        for i in range(len(arr1)):
            for j in range(len(arr2)):
                if arr1[i] > arr2[j]:
                    total += 1
        return total

    inv_count = inv_calculate(first_sample, last_sample)
    M = first_sample.size * last_sample.size / 2
    D = first_sample.size * last_sample.size / 12 * (first_sample.size + last_sample.size + 1)
    return np.float64(M), np.float64(D), inv_count


def critical_district(q: np.float64, M: np.float64, std_inv: np.float64) -> tuple[np.float64, np.float64, np.float64]:
    """
    Построение критической области для числа инверсий исходя из соотношения
    q = 1 - Ф(t)
    :param q: np.float64 - уровень значимости
    :param M: np.float64 - мат. ожидание числа инверсий
    :param std_inv: np.float64 - ср. кв. отклонение числа инверсий
    :return:
    """
    t_i = laplace_reverse(np.float64((1 - q) / 2 + 0.5))
    return np.float64(t_i), np.float64(M - std_inv * t_i), np.float64(M + t_i * std_inv)
