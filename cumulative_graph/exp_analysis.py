import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter
from scipy.stats import kstest
from cumulative_graph.detect_abrupt import detect_abrupt_changes
from cumulative_graph.detect_abrupt import detect_abrupt_changes_cusum
from cumulative_graph.manage_lines import least_squares_line
from cumulative_graph.manage_lines import get_line_equation
from cumulative_graph.analyze_results import calculate_aic
from cumulative_graph.analyze_results import calculate_error_metrics
from cumulative_graph.forecast_time_series import forecast_with_least_squares
from cumulative_graph.manage_lines import create_report
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates



# Example usage
if __name__ == "__main__":

    value = '2'
    department_name = 'CT'
    output_file_path = '../results/CV/2023/2_category/results.txt'
    output_df_file_path = '../results/CV/2023/2_category/df.csv'
    histogram_file_path = '../results/CV/2023/2_category/histogram.pdf'
    detect_changes_file_path = '../results/CV/2023/2_category/detect_changes.pdf'
    output_report_file_path = '../results/CV/2023/2_category/report_df.xlsx'
    output_time_diff_df = '../results/CV/2023/2_category/time_diff_df.csv'
    df = pd.read_csv('../sources/CT_2024.csv', header=None)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)


    #get optimal settings
    print('Оптимальные настройки')
    settings_df = pd.read_json('optimal_settings.json')
    settings_df['category'] = settings_df['category'].astype(str)
    settings_df['department'] = settings_df['department'].astype(str)
    settings_df.query("(department == @department_name) and (category == @value)", inplace=True)
    settings_df = settings_df.reset_index(drop=True)
    print(settings_df.info())
    print(settings_df.to_string())
    hist_param = settings_df['hist'].iloc[0]
    penalty_param = settings_df['penalty'].iloc[0]

    # Create a sample DataFrame with float64 time series data
    #df = pd.read_csv('../sources/CSH_2024.csv', header=None)
    df.columns = ['START_TIME','CATEGORY']
    df['START_TIME'] = pd.to_datetime(df['START_TIME'], format='%d-%m-%Y %H:%M:%S')
    df['CATEGORY'] = df['CATEGORY'].astype(str)
    print(df.head())
    print(len(df))
    print(df['START_TIME'].iloc[0])
    if value == 'all':
        print('Use Dataframe without filters\n')
    else:
        df.query("CATEGORY == @value", inplace=True)
        df = df.reset_index(drop=True)
        print("After filtering get a new DataFrame:\n")
        print(df.head())
        print('DATAFRAME LENGHT:\n')
        print(len(df))
    print(df['START_TIME'].iloc[0])
    df['START_TIME'] = pd.to_datetime(df['START_TIME'])
    print(df.info())
    df['DELTA_TIME'] = df['START_TIME'] - df['START_TIME'].iloc[0]
    print(df.info())
    df['DELTA_MINUTES'] = df['DELTA_TIME'].dt.total_seconds() / 60
    print(df.head())
    print(df.info())
    df['TIME_DIFF'] = df['DELTA_MINUTES'].diff()
    df.loc[df.index[0], 'TIME_DIFF'] = 0.0
    print(df.head())
    print(df.info())
    df['INDEX'] = (1 / len(df)) * (df.index)
    print(df.head())
    print(df.tail())
    print(df.info())
    df.to_csv(output_df_file_path, index=True)

# Q-Q Plot function
    stats.probplot(df['TIME_DIFF'], dist="expon", plot=plt)
    plt.title("Q-Q Plot (Exponential Distribution)")
    plt.grid(True)
    plt.show()

# P-P Plot function
    # Примерные данные (замени на свои)
    pp_df = df['TIME_DIFF']
    pp_df = np.sort(pp_df)
    # Эмпирическая CDF
    ecdf = np.arange(1, len(pp_df) + 1) / len(pp_df)
    # Теоретическая CDF для экспоненциального распределения
    # Параметр scale = 1 / λ (возможно, нужно подогнать под свои данные)
    cdf_theoretical = stats.expon.cdf(pp_df, scale=np.mean(pp_df))
    # Построение P-P Plot
    plt.figure(figsize=(6, 6))
    plt.plot(cdf_theoretical, ecdf, 'o', label='P-P points')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y = x)')
    plt.xlabel("Теоретические вероятности")
    plt.ylabel("Эмпирические вероятности")
    plt.title("P-P Plot (Exponential Distribution)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Get best
    f = Fitter(pp_df, distributions=['norm', 'lognorm', 'expon', 'gamma', 't'])
    f.fit()
    f.summary()  # таблица результатов
    print(f.get_best())  # {'norm': (mu, sigma)}
    print(f.fitted_param['expon'])  # параметры логнормального распределения

# Тест Колмогорова-Смирнова
    ks_stat, p_value = kstest(pp_df, 'expon', args=(0, 3160.036))
    print(f"KS statistic: {ks_stat}, p-value: {p_value}")
