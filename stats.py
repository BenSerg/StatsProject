import scipy
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.stats import norm, chi2, t

from data_processing import sign_check, inversion_check, critical_district
from laplace_function import *

precision = 3
interval_count = 11
freedom_degrees = 7

def filter_vals(interval: tuple, sample: np.array) -> np.array:
    return sample[np.logical_and(interval[0] <= sample, sample < interval[1])]


class Stats:
    def __init__(self, min_value: np.float64, max_value: np.float64, mean_value: np.float64, linspace: np.array,
                 median: np.float64, std: np.float64, hist: np.array, bins: np.array, skewness: np.float64,
                 kurtosis: np.float64):
        self.min_value, self.max_value, self.mean_value, = min_value, max_value, mean_value
        self.linspace, self.median, self.std = linspace, median, std
        self.hist, self.bins, self.skewness, self.kurtosis = hist, bins, skewness, kurtosis


def create_report(file_path: str, data_sample: np.array, stats: Stats) -> None:
    min_value, max_value, mean_value, linspace, median, std, hist, bins, skewness, kurtosis = stats.__dict__.values()
    with open(file=file_path, mode='w', encoding='utf-8') as stats_file:
        table = PrettyTable()
        table.field_names = ['Interval N', 'Begin', 'End', 'Xi', 'Fi', 'Fi / N', 'acc Fi / N']
        interval_list = list(
            (np.round(linspace[i - 1], 1), np.round(linspace[i], 1)) for i in range(1, interval_count + 1))
        mid_list = np.array(list(np.mean(i) for i in interval_list))
        frequencies = np.array(list(filter_vals(i, data_sample).size for i in interval_list))
        accumulation_frequency = np.add.accumulate(frequencies / data_sample.size)
        table_rows = [
            [i + 1, interval_list[i][0], interval_list[i][1], np.round(mid_list[i], precision), frequencies[i],
             np.round(frequencies[i] / data_sample.size, 3), np.round(accumulation_frequency[i], 3)]
            for i in range(interval_count)
        ]
        xi_square = chi2_value(interval_list, frequencies, data_sample, mean_value, std)
        print(f'Max: {max_value:.{precision}f}\n'
              f'Min: {min_value:.{precision}f}\n'
              f'Range: {max_value - min_value:.{precision}f}\n'
              f'Delta: {(max_value - min_value) / interval_count:.{precision}f}\n'
              f'Begin point: {min_value:.{precision}f}\n'
              f'X_mean: {mean_value:.{precision}f}\n'
              f'Std: {std:.{precision}f}\n'
              f'Median: {median:.{precision}f}\n'
              f'Skewness: {skewness:.{precision}f}\n'
              f'Kurtosis: {kurtosis:.{precision}f}\n'
              f'Xi_square: {xi_square:.{precision}f}\n\n', file=stats_file)
        table.add_rows(table_rows)
        print(table, file=stats_file)
        draw_polygon(mid_list, frequencies / data_sample.size)
        draw_histogram(bins, hist / data_sample.size)
        draw_accumulation(bins, accumulation_frequency)
        m, sigma = quantil_solve(data_sample)
        print(f'\n\n'
              f'Quantile approximation:\n'
              f'M: {m:.{precision}f}\n'
              f'σ: {sigma:.{precision}f}\n\n', file=stats_file)
        q = [0.01, 0.05, 0.1]
        sign_checkers = [4, 6, 5]
        for i in range(3):
            print(f'Xi square value with q = {q[i]}: {get_chi2(q[i], freedom_degrees):.{precision}f}\n\n', file=stats_file)
            elms = data_sample[:20]
            m_trusted_interval, sigma_trusted_interval = trusted_intervals(elms, np.float64(q[i]))
            val = sign_check(data_sample, count=20)
            print(f'M confidence interval: {tuple(map(lambda x: np.round(x, precision), m_trusted_interval))}\n'
                  f'σ confidence interval: {tuple(map(lambda x: np.round(x, precision), sigma_trusted_interval))}',
                  file=stats_file)
            if val > sign_checkers[i]:
                print(f'min(k_i) = {val} > {sign_checkers[i]}\n', file=stats_file)
            else:
                print(f'min(k_i) = {val} < {sign_checkers[i]}\n\n', file=stats_file)
            print('Inversions\n', file=stats_file)
            M, D, inv_count = inversion_check(data_sample, count=20)
            print(f'M[u] = {M:.{precision}f}\n'
                  f'D[u] = {D:.{precision}f}\n'
                  f'σ[u] = {np.sqrt(D):.{precision}f}\n'
                  f'Inversions_count = {inv_count}\n', file=stats_file)
            t_i, *u_interval = critical_district(np.float64(q[i]), M, np.sqrt(D))
            print(f't = {t_i:.{precision}f}\n'
                  f'u <= {u_interval[0]:.{precision}f}\n'
                  f'u >= {u_interval[1]:.{precision}f}\n\n', file=stats_file)


def create_stats(data_sample: np.array) -> Stats:
    min_value, max_value, mean_value = data_sample.min(), data_sample.max(), data_sample.mean()
    linspace = np.linspace(min_value, max_value + 1, num=interval_count + 1)
    median = data_sample.median()
    std = data_sample.std()

    hist, bins = np.histogram(data_sample, linspace)
    skewness = scipy.stats.skew(data_sample)
    kurtosis = scipy.stats.kurtosis(data_sample)
    return Stats(min_value, max_value, mean_value, linspace, median, std, hist, bins, skewness, kurtosis)


def draw_polygon(x: np.array, y: np.array, color='r') -> None:
    plt.figure()
    plt.plot(x, y, color)
    plt.xlabel('Xi')
    plt.ylabel('Fi / N')
    plt.grid()
    plt.savefig('stats_analysis/pictures/polygon.png')


def draw_accumulation(bins: np.array, accumulation: np.array, color='lightgreen') -> None:
    plt.figure()
    plt.bar(bins[:-1], accumulation, width=(bins[1] - bins[0]), color=color, edgecolor='black', alpha=0.5)
    plt.xlabel('Xi')
    plt.ylabel('acc Fi / N')
    plt.title('Accumulation frequency')
    plt.grid(axis='y')
    plt.savefig('stats_analysis/pictures/accumulation.png')


def draw_histogram(bins: np.array, data: np.array, color='yellow') -> None:
    plt.figure()
    plt.bar(bins[:-1], data, width=(bins[1] - bins[0]), color=color, edgecolor='black', alpha=0.5)
    plt.xlabel('Xi')
    plt.ylabel('Fi / N')
    plt.title('Histogram')
    plt.grid(axis='y')
    plt.savefig('stats_analysis/pictures/histogram.png')


@lru_cache
def get_chi2(alpha: float, k: int) -> np.float64:
    return chi2.ppf(1 - alpha, k)


@lru_cache
def calculate_students_coef(q: np.float64, n: int) -> np.float64:
    return t.ppf(1 - q / 2, n - 1)


def trusted_intervals(samples: np.array, q: np.float64) -> list[tuple]:
    x_mean = np.mean(samples)
    s = np.sqrt(np.mean(np.power(samples, 2)) - x_mean ** 2)
    n = samples.size
    return [exp_trust_interval(x_mean, q, s, n), std_trust_interval(q, n, s)]


def exp_trust_interval(x_mean: np.float64, q: np.float64, s: np.float64, n: int) -> tuple[np.float64, np.float64]:
    return (x_mean - calculate_students_coef(q, n) * s / np.sqrt(n - 1),
            x_mean + calculate_students_coef(q, n) * s / np.sqrt(n - 1))


def std_trust_interval(q: np.float64, n: int, s: np.float64) -> tuple[np.float64, np.float64]:
    return (np.sqrt(n) * s / np.sqrt(get_chi2(q / 2, n - 1)),
            np.sqrt(n) * s / np.sqrt(get_chi2(1 - q / 2, n - 1)))


def chi2_value(interval_list: list[tuple], frequencies: np.array, data_sample: np.array, mean_value: np.float64,
               std: np.float64):
    probability_intervals = np.array([probability_in_interval(i, mean_value, std) for i in interval_list])
    return np.sum(
            np.power(frequencies - data_sample.size * probability_intervals, 2) /
            (data_sample.size * probability_intervals))


@lru_cache
def probability_in_interval(interval: tuple[np.float64, np.float64], mean_value: np.float64,
                            std: np.float64) -> np.float64:
    return np.float64(laplace_function(np.float64((interval[1] - mean_value) / std)) -
                      laplace_function(np.float64(((interval[0]) - mean_value) / std)))
