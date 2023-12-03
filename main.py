import pandas as pd

from stats import create_stats, create_report


def main():
    file_name = 'stats_data.xlsx'
    try:
        data = pd.read_excel(file_name, sheet_name='data')
        val_column_name = data.columns[-1]
        data_sample = data[val_column_name].array
        sorted_data = pd.DataFrame({
            'sorted': data[val_column_name].sort_values()
        })
        with pd.ExcelWriter('stats_data.xlsx', mode='a', if_sheet_exists='replace') as writer:
            sorted_data.to_excel(writer, sheet_name='sorted_data', engine='openpyxl', index=False)
        stats = create_stats(data_sample)
        report_file_name = 'stats_analysis/data_info.txt'
        create_report(report_file_name, data_sample, stats)
    except FileNotFoundError:
        print(f'Файл {file_name} не найден')


if __name__ == '__main__':
    main()
