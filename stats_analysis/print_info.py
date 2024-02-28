from typing import TextIO
from matplotlib import pyplot as plt
import scipy
from descriptions import stats

from descriptions.StatsTable import StatsTable
from calculations.criterion_calculation_functions import *
from calculations.data_processing import sign_check, inversion_check, critical_district
from calculations.laplace_function import *


def print_statistics(descr_stats: stats.DescriptiveStats, chi_square: np.float64, file: TextIO):
    print(f'Max: {descr_stats.max_value:.{precision}f}\n'
          f'Min: {descr_stats.min_value:.{precision}f}\n'
          f'Range: {descr_stats.max_value - descr_stats.min_value:.{precision}f}\n'
          f'Delta: {(descr_stats.max_value - descr_stats.min_value) / interval_count:.{precision}f}\n'
          f'Begin point: {descr_stats.min_value:.{precision}f}\n'
          f'X_mean: {descr_stats.mean_value:.{precision}f}\n'
          f'Std: {descr_stats.std:.{precision}f}\n'
          f'Median: {descr_stats.median:.{precision}f}\n'
          f'Skewness: {descr_stats.skewness:.{precision}f}\n'
          f'Kurtosis: {descr_stats.kurtosis:.{precision}f}\n'
          f'Chi square: {chi_square:.{precision}f}\n\n', file=file)


def print_quantile_approximation(loc: np.float64, std: np.float64, file: TextIO):
    print(f'\n\n'
          f'Quantile approximation:\n'
          f'M: {loc:.{precision}f}\n'
          f'σ: {std:.{precision}f}\n\n', file=file)


def print_inversion_check(loc: np.float64, var: np.float64, inv_count: int, file: TextIO):
    print(f'M[u] = {loc:.{precision}f}\n'
          f'D[u] = {var:.{precision}f}\n'
          f'σ[u] = {np.sqrt(var):.{precision}f}\n'
          f'Inversions_count = {inv_count}\n', file=file)


def print_chi2_with_alpha(alpha: np.float64, file: TextIO):
    print(f'Chi square value with q = {alpha}: {get_chi2(alpha, freedom_degrees):.{precision}f}\n\n',
          file=file)


def print_confidence_intervals(data_sample, count, alpha, sign_checker, file):
    elms = data_sample[:count]
    m_trusted_interval, sigma_trusted_interval = trusted_intervals(elms, np.float64(alpha))
    val = sign_check(data_sample, count=count)
    print(f'M confidence interval: {m_trusted_interval}\n'
          f'σ confidence interval: {sigma_trusted_interval}',
          file=file)
    if val > sign_checker:
        print(f'min(k_i) = {val} > {sign_checker}\n', file=file)
    else:
        print(f'min(k_i) = {val} < {sign_checker}\n\n', file=file)


def print_inverse_confidence_interval(alpha, loc, std, file):
    print('Inversions\n', file=file)
    t_i, *u_interval = critical_district(np.float64(alpha), loc, std)
    print(f't = {t_i:.{precision}f}\n'
          f'u <= {u_interval[0]:.{precision}f}\n'
          f'u >= {u_interval[1]:.{precision}f}\n\n', file=file)


def create_report(file_path: str, data_sample: np.array,
                  statistics: tuple[stats.DescriptiveStats, stats.HistoGramStats]) -> None:
    min_value, max_value, mean_value, median, std, skewness, kurtosis, linspace = statistics[0].__dict__.values()
    hist, bins = statistics[1].__dict__.values()
    with open(file=file_path, mode='w', encoding='utf-8') as stats_file:
        table = StatsTable(data_sample, descr_stats=statistics[0])
        chi_square = chi2_value(table.interval_list, table.frequencies, data_sample, mean_value, std)
        print_statistics(descr_stats=statistics[0], chi_square=chi_square, file=stats_file)
        print(table, file=stats_file)
        draw_polygon(table.mid_list, table.frequencies / data_sample.size)
        draw_histogram(bins, hist / data_sample.size)
        draw_accumulation(bins, table.accumulation_frequency)
        m, sigma = quantil_solve(data_sample)
        print_quantile_approximation(m, sigma, stats_file)
        M, D, inv_count = inversion_check(data_sample, count=20)
        print_inversion_check(M, D, inv_count, stats_file)

        q = [0.01, 0.05, 0.1]
        sample_count = 20
        sign_checkers = [calculate_sign_checker(i, sample_count) for i in q]

        for i in range(len(q)):
            print_chi2_with_alpha(q[i], stats_file)
            print_confidence_intervals(data_sample, sample_count, q[i], sign_checkers[i], stats_file)
            print_inverse_confidence_interval(q[i], M, np.sqrt(D), stats_file)


def create_stats(data_sample: np.array) -> tuple[stats.DescriptiveStats, stats.HistoGramStats]:
    min_value, max_value, mean_value = data_sample.min(), data_sample.max(), data_sample.mean()
    linspace = np.linspace(min_value, max_value + 1, num=interval_count + 1)
    median = data_sample.median()
    std = data_sample.std()

    hist, bins = np.histogram(data_sample, linspace)
    skewness = scipy.stats.skew(data_sample)
    kurtosis = scipy.stats.kurtosis(data_sample)
    return (stats.DescriptiveStats(min_value, max_value, mean_value, linspace, median, std, skewness, kurtosis),
            stats.HistoGramStats(hist, bins))


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
