from prettytable import PrettyTable

from descriptions import stats
from calculations.constants import *
from calculations.laplace_function import *


class StatsTable:
    @staticmethod
    def filter_vals(interval: tuple, sample: np.array) -> np.array:
        return sample[np.logical_and(interval[0] <= sample, sample < interval[1])]

    @staticmethod
    def calculate_interval_list(linspace):
        return list((np.round(linspace[i - 1], 1), np.round(linspace[i], 1)) for i in range(1, interval_count + 1))

    @staticmethod
    def calculate_cumulative_function(interval_list, mean_value, std):
        return [norm.cdf(interval_list[i][1], loc=mean_value, scale=std) -
                norm.cdf(interval_list[i][0], loc=mean_value, scale=std) for i in range(interval_count)]

    def set_table_rows(self, data_sample, norm_accumulate):
        self.table_rows = [
            [i + 1, self.interval_list[i][0], self.interval_list[i][1], np.round(self.mid_list[i], precision),
             self.frequencies[i],
             np.round(self.frequencies[i] / data_sample.size, 3), np.round(self.accumulation_frequency[i], 3),
             np.round(norm_accumulate[i], 3),
             np.round(np.abs(np.round(norm_accumulate[i], 3) - np.round(self.accumulation_frequency[i], 3)), 3)]
            for i in range(interval_count)
        ]

    def __str__(self):
        return self.table.__str__()

    def __init__(self, data_sample: np.array, descr_stats: stats.DescriptiveStats):
        self.table_rows = None
        self.table = PrettyTable()
        self.table.field_names = ['Interval N', 'Begin', 'End', 'Xi', 'Fi', 'Fi / N', 'acc Fi / N', 'norm acc Fi/N',
                                  'diff']
        linspace = descr_stats.linspace
        self.interval_list = self.calculate_interval_list(linspace)
        self.mid_list = np.array(list(np.mean(i) for i in self.interval_list))
        self.frequencies = np.array(list(self.filter_vals(i, data_sample).size for i in self.interval_list))
        norm_cdf_values = self.calculate_cumulative_function(self.interval_list, descr_stats.mean_value,
                                                             descr_stats.std)
        self.accumulation_frequency = np.add.accumulate(self.frequencies / data_sample.size)
        self.set_table_rows(data_sample, np.add.accumulate(norm_cdf_values))

        self.table.add_rows(self.table_rows)
