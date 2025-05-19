import os
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from cumulative_graph.detect_abrupt import detect_abrupt_changes
from cumulative_graph.detect_abrupt import detect_abrupt_changes_cusum
from fitter import Fitter
from scipy.stats import expon, kstest
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

    MINUTES_IN_MONTH = 43829.1  # среднее количество минут в месяце
    value = '1'
    department_name = 'CV'
    output_file_path = 'results/CV/2024/1_category/results.txt'
    output_df_file_path = 'results/CV/2024/1_category/df.csv'
    histogram_file_path = 'results/CV/2024/1_category/histogram.pdf'
    detect_changes_file_path = 'results/CV/2024/1_category/detect_changes.pdf'
    output_report_file_path = 'results/CV/2024/1_category/report_df.xlsx'
    exp_fit_results_path = 'results/CV/2024/1_category/exp_fit_results.xlsx'
    pp_plot_dir = 'results/CV/2024/1_category/pp_plots'  # ← здесь можно изменить путь
    os.makedirs(pp_plot_dir, exist_ok=True)
    df = pd.read_csv('../sources/CV_2024.csv', header=None)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    #get optimal settings
    print('Оптимальные настройки')
    settings_df = pd.read_json('settings/optimal_settings.json')
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


    df = df.dropna()
    # Define bin edges to match data
    bin_edges = np.arange(df['TIME_DIFF'].min(), df['TIME_DIFF'].max() + 1, hist_param)  # Step size of 100 sec
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['TIME_DIFF'], bins=bin_edges, edgecolor='black', alpha=0.7)
    # Remove empty space by adjusting x-axis limits
    plt.xlim(df['TIME_DIFF'].min(), df['TIME_DIFF'].max())
    # Labels and title
    plt.xlabel('Интервалы между сбойными событиями, мин')
    plt.ylabel('Частота')
    plt.title('Гистограмма интервалов между сбойными событиями')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Show plot
    plt.savefig(histogram_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Detect abrupt changes
    detected_changes = detect_abrupt_changes(df, start_event="START_TIME", time_column="DELTA_MINUTES", value_column="INDEX", model="rbf", pen=penalty_param)
    print("Detected Change Points with Coordinates:\n", detected_changes)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df["START_TIME"], df["INDEX"], label="График накопленного числа событий")

    for _, row in detected_changes.iterrows():
        plt.axvline(row["START_TIME"], color="red", linestyle="--", label=f"Точка изменения {row['START_TIME']}")
    plt.legend()
    plt.title("График накопленного числа событий")
    plt.xlabel("x,мин")
    plt.ylabel("Нормированное значение")
    plt.xlim(df['START_TIME'].iloc[0], df['START_TIME'].iloc[len(df)-1])
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(detect_changes_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    change_points_df = detect_abrupt_changes_cusum(df, value_column="INDEX", time_column="DELTA_MINUTES", threshold=40,
                                                   drift=0.5)


    #======================================================Новый функционал====================================================

    # Output detected change points as DataFrame
    print("Change points detected:")
    print(detected_changes)

    #separate parts of dataframe
    # Найдём индексы в df, где значения совпадают с контрольными
    checkpoints = df[df['DELTA_MINUTES'].isin(detected_changes['DELTA_MINUTES'])].index.tolist()

    #Разбиваем по этим индексам
    dfs = []
    prev_idx = 0
    for idx in checkpoints:
        dfs.append(df.iloc[prev_idx:idx])
        prev_idx = idx
    dfs.append(df.iloc[prev_idx:])  # Последняя часть

    #Применить fitter
    results = []
    for i, part in enumerate(dfs):
        #Проверка на пустой dataframe
        if part.empty:
            print(f"\nЧасть {i} пуста — пропускаем.")
            continue
        #Извлечь поле TIME_DIFF
        data = part['TIME_DIFF'].dropna().values
        #Проверка что длина dataframe не меньше 10
        if len(data) < 10:
            print(f"\nЧасть {i} слишком мала для анализа (n={len(data)}) — пропускаем.")
            continue
        #Прменение fitter dataframe для общего анализа
        print(f"\n Анализ части {i} (n={len(data)}):")
        f = Fitter(
            data,
            distributions=['norm', 'lognorm', 'expon', 'gamma', 'beta'],
            timeout=10
        )
        f.fit()
        #f.summary()
        #Определение параметра для экспоненциального распределения
        try:
            f = Fitter(data, distributions=['expon'], timeout=10)
            f.fit()
            params = f.fitted_param['expon']
            loc, scale = params
            lambda_est = 1 / scale
            lambda_est_month = lambda_est * MINUTES_IN_MONTH

            print(f"  ➤ Параметры экспоненциального распределения:")
            print(f"     loc = {loc:.4f}, scale = {scale:.4f}")
            print(f"     λ (в минуту)      ≈ {lambda_est:.10f}")
            print(f"     λ (в месяц)       ≈ {lambda_est_month:.4f}")

            # Тест Колмогорова–Смирнова
            ks_stat, ks_pvalue = kstest(data, 'expon', args=(0, 1 / lambda_est))
            ks_hypothesis = ks_pvalue >= 0.05  # True = гипотеза не отклоняется

            #Добавление результатов обработки в результирующий dataframe
            results.append({
                'dataframe_index': i,
                'start_time': part['START_TIME'].iloc[0],
                'end_time': part['START_TIME'].iloc[-1],wa
                'loc': loc,
                'scale': scale,
                'lambda_est': lambda_est,
                'lambda_est_month': lambda_est_month,
                'ks_statistic': round(ks_stat, 5),
                'ks_pvalue': round(ks_pvalue, 5),
                'ks_hypothesis': ks_hypothesis
            })

            #Построение P-P plot сравнения идеального эсконенциального распределения и изначального dataframe
            sorted_data = np.sort(data)
            n = len(sorted_data)
            empirical_cdf = np.arange(1, n + 1) / n
            theoretical_cdf = expon.cdf(sorted_data, loc=0, scale=1 / lambda_est)

            plt.figure(figsize=(6, 6))
            plt.plot(theoretical_cdf, empirical_cdf, 'o', label='P–P точки')
            plt.plot([0, 1], [0, 1], 'r--', label='Идеальное совпадение')
            plt.xlabel('Теоретическая CDF (expon)')
            plt.ylabel('Эмпирическая CDF')
            plt.title(f'P–P Plot (часть {i})')
            plt.legend()
            plt.grid(True)
            plt.axis('square')
            plt.show()
            file_path = os.path.join(pp_plot_dir, f'pp_plot_part_{i}.png')
            plt.savefig(file_path)
            plt.close()

        except Exception as e:
            print(f"  ⚠️ Ошибка при анализе части {i}: {e}")

    df_result = pd.DataFrame(results)
    df_result.to_excel(exp_fit_results_path,index=True)
    print(df_result)


    # # Plot the time series with detected change points
    # plt.figure(figsize=(10, 6))
    # plt.plot(df["DELTA_MINUTES"], df["INDEX"], label="Time Series")
    # for _, row in change_points_df.iterrows():
    #     plt.axvline(row["DELTA_MINUTES"], color="red", linestyle="--", label=f"Change Point at {row['DELTA_MINUTES']:.2f}")
    # plt.legend()
    # plt.title("Time Series Change Point Detection (CUSUM)")
    # plt.xlabel("DELTA_MINUTES")
    # plt.ylabel("INDEX")
    # plt.xlim(0, df['DELTA_MINUTES'].iloc[len(df)-11])
    # plt.ylim(0, 1)
    # plt.grid()
    # plt.show()




    # # Get the equations of the lines between change points using least squares
    # with open(output_file_path, "w") as f:
    #     line_equations = get_line_equation(detected_changes, df)
    #     f.write("\nEquations of the best fit lines between change points (Least Squares):\n")
    #     for i, (m, b) in enumerate(line_equations):
    #         f.write(f"Line {i + 1}: y = {m:.8f}x + ({b:.2f})\n")
    #     if not detected_changes.empty:
    #         last_change_point = detected_changes["DELTA_MINUTES"].iloc[-1]
    #     else:
    #         last_change_point = df["DELTA_MINUTES"].iloc[-1]
    #     f.write(f"Last change point: {last_change_point}")
    # f = open(output_file_path, "r")
    # print(f.read())
    #
    # print('CHECK_REPORT')
    # create_report(df, detected_changes, line_equations).to_excel(output_report_file_path,index=True)
    # # line_equations = get_line_equation(detected_changes, df)
    # # print("\nEquations of the best fit lines between change points (Least Squares):")
    # # for i, (m, b) in enumerate(line_equations):
    # #     print(f"Line {i+1}: y = {m:.8f}x + {b:.2f}")
    #
    # # Get the last change point
    # # if not detected_changes.empty:
    # #     last_change_point = detected_changes["DELTA_MINUTES"].iloc[-1]
    # # else:
    # #     last_change_point = df["DELTA_MINUTES"].iloc[-1]
    # #
    # # print(f"Last change point: {last_change_point}")
    #
    # # Fit Least Squares (Linear Regression) model
    # filtered_df = df[df["DELTA_MINUTES"] >= last_change_point].copy()
    # X = filtered_df["DELTA_MINUTES"].values.reshape(-1, 1)
    # y = filtered_df["INDEX"].values
    #
    # model = LinearRegression()
    # model.fit(X, y)
    #
    # # Calculate AIC for the model
    # aic = calculate_aic(model, X, y)
    # print(f"Akaike's Information Criterion (AIC): {aic}")
    #
    # # Forecast future values using Least Squares (Linear Regression)
    # forecast_period = 50000
    # forecast_df = forecast_with_least_squares(df.copy(), value_column="INDEX", forecast_period=forecast_period,
    #                                           last_change_point=last_change_point)
    #
    # # Evaluate metrics on existing data
    # y_true = filtered_df["INDEX"].values
    # y_pred = model.predict(filtered_df["DELTA_MINUTES"].values.reshape(-1, 1))
    # metrics = calculate_error_metrics(y_true, y_pred)
    #
    # print("\nError Metrics for the Model:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.8f}")
    #
    # print("\nForecasted Values using Least Squares:")
    # print(forecast_df)
    #
    # # Plot the original time series, change points, and forecasts
    # plt.figure(figsize=(12, 6))
    # plt.plot(df["DELTA_MINUTES"], df["INDEX"], label="Original Time Series")
    # plt.scatter(detected_changes["DELTA_MINUTES"], detected_changes["INDEX"], color="red", label="Change Points")
    # plt.plot(forecast_df["DELTA_MINUTES"], forecast_df["FORECAST"], label="Least Squares Forecast", linestyle="--")
    # plt.legend()
    # plt.title("Time Series with Detected Change Points and Forecast (Least Squares)")
    # plt.xlabel("DELTA_MINUTES")
    # plt.ylabel("INDEX")
    # plt.xlim(0)
    # plt.ylim(0)
    # plt.grid()
    # plt.show()