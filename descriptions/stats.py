import numpy as np


class DescriptiveStats:
    """
    Класс, представляющий статистические показатели описательной статистики.

    Поля:\n
    - min_value: Минимальное значение.
    - max_value: Максимальное значение.
    - mean_value: Среднее значение.
    - linspace: Массив значений.
    - median: Медиана.
    - std: Стандартное отклонение.
    - skewness: Коэффициент асимметрии.
    - kurtosis: Коэффициент эксцесса.

    """
    def __init__(self, min_value: np.float64, max_value: np.float64, mean_value: np.float64, linspace: np.array,
                 median: np.float64, std: np.float64, skewness: np.float64, kurtosis: np.float64):
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.median = median
        self.std = std
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.linspace = linspace


class HistoGramStats:
    """
    Класс, представляющий статистические показатели для гистограммы
    Поля:\n
    - hist: Массив значений гистограммы.
    - bins: Массив корзин.
    """
    def __init__(self, hist: np.array, bins: np.array):
        self.hist = hist
        self.bins = bins
