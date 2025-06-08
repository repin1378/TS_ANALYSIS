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
from utils import generate_segmented_exponential_dataset, add_start_times, make_objective, detect_cusum_changes
import optuna
import seaborn as sns
from tqdm import tqdm

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
    output_df_exp_file_path = 'results/CV/2024/1_category/df_exp.csv'
    histogram_exp_file_path = 'results/CV/2024/1_category/histogram_exp.pdf'
    nce_plot_file_path = 'results/CV/2024/1_category/nce_plot.pdf'
    diff_plot_file_path = 'results/CV/2024/1_category/diff_plot.pdf'

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
                'end_time': part['START_TIME'].iloc[-1],
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
    indices_to_multiply = [0, 2]
    df_result.loc[indices_to_multiply, 'lambda_est'] *= 10
    df_result.to_excel(exp_fit_results_path,index=True)
    print(df_result)

#=================================Генерация Dataframe c экспоненциальном распределением====================================

    # Список сегментов: (кол-во событий, индекс строки в df_result)
    segments = [(500, 1), (200, 0), (500, 1), (200, 0), (500, 1), (200, 2), (500, 1), (200, 0), (500, 1), (200, 2), (500, 1)]

    # Генерация
    df_gen = generate_segmented_exponential_dataset(df_result, segments, seed=None)

    # Добавление временных меток
    df_gen = add_start_times(df_gen, start_time_0='2023-01-01 00:00:00')

    # Просмотр результата
    print(df_gen.head())

    # Подготовка Dataframe
    df_gen.loc[df_gen.index[0], 'TIME_DIFF'] = 0.0
    df_gen['INDEX'] = (1 / len(df_gen)) * (df_gen.index)
    print(df.head())
    print(df.tail())
    print(df.info())
    df_gen.to_csv(output_df_exp_file_path, index=True)
    df_gen = df_gen.dropna()


    # Гистограмма
    bin_edges = np.arange(df_gen['TIME_DIFF'].min(), df_gen['TIME_DIFF'].max() + 1, hist_param)  # Step size of 100 sec
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df_gen['TIME_DIFF'], bins=bin_edges, edgecolor='black', alpha=0.7)
    # Remove empty space by adjusting x-axis limits
    plt.xlim(df_gen['TIME_DIFF'].min(), df_gen['TIME_DIFF'].max())
    # Labels and title
    plt.xlabel('Интервалы между сбойными событиями, мин')
    plt.ylabel('Частота')
    plt.title('Гистограмма интервалов между сбойными событиями')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Show plot
    plt.savefig(histogram_exp_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    #График накопленного числа событий
    # === Подготовка данных таблицы ===
    segment_lengths = [length for length, _ in segments]
    cumulative_lengths = np.cumsum(segment_lengths)[:-1]
    switch_times = df_gen.loc[cumulative_lengths, 'START_TIME'].values

    table_data = []
    for i, (switch_idx, t) in enumerate(zip(cumulative_lengths, switch_times)):
        t_pd = pd.to_datetime(t)
        lam_val = df_gen.loc[switch_idx, 'lambda_est']
        table_data.append([
            i + 1,
            switch_idx,
            t_pd.strftime("%Y-%m-%d %H:%M"),
            f"{lam_val:.6f}"
        ])

    # === Поиск минимального значения lambda_est ===
    min_lambda = df_result['lambda_est'].min()
    min_lambda_rows = df_gen[df_gen['lambda_est'] == min_lambda]
    if not min_lambda_rows.empty:
        min_lambda_index = min_lambda_rows.index[0]
        min_lambda_time = df_gen.loc[min_lambda_index, 'START_TIME']
    else:
        min_lambda_index = None
        min_lambda_time = None

    # === Создание общей фигуры ===
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    # === График накопленного числа событий ===
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_gen["START_TIME"], df_gen["INDEX"], label="Накопленное число событий")

    # 🔴 Красные линии — смены λ
    for idx, (switch_idx, t) in enumerate(zip(cumulative_lengths, switch_times)):
        lam_val = df_gen.loc[switch_idx, 'lambda_est']
        if lam_val == min_lambda:
            continue  # Пропускаем, если это уже min λ

        t_pd = pd.to_datetime(t)
        ax1.axvline(x=t_pd, color='red', linestyle='--', linewidth=1, label='Смена λ' if idx == 0 else None)

    # # 🔵 Синие линии — сигналы CUSUM
    # for j, (_, row) in enumerate(alerts.iterrows()):
    #     ax1.axvline(x=row['START_TIME'], color='blue', linestyle='-', linewidth=1, label='CUSUM' if j == 0 else None)

    # ✅ 🟢 Зелёная линия — переход к min(lambda_est)
    for idx, (switch_idx, t) in enumerate(zip(cumulative_lengths, switch_times)):
        lam_val = df_gen.loc[switch_idx, 'lambda_est']
        if lam_val != min_lambda:
            continue  # Пропускаем, если это уже min λ

        min_lambda_time = pd.to_datetime(t)
        ax1.axvline(x=min_lambda_time, color='green', linestyle='--', linewidth=1, label='Переход к min λ' if idx == 1 else None)

    # Оформление графика
    ax1.set_title("График накопленного числа событий")
    ax1.set_xlabel("Время события")
    ax1.set_ylabel("Нормированное значение")
    ax1.set_xlim(df_gen['START_TIME'].iloc[0], df_gen['START_TIME'].iloc[-1])
    ax1.set_ylim(0, 1)
    ax1.grid()
    ax1.legend()

    # === Таблица смен λ ===
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    column_labels = ["Смена №", "Индекс", "Время", "λ"]
    table = ax2.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)

    # === Сохранение и показ ===
    plt.tight_layout()
    plt.savefig(nce_plot_file_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Построение списка истинных точек смены λ
    true_change_points = np.cumsum([length for length, _ in segments])[:-1]

    # Выбор минимального значения λ для CUSUM
    lambda_0 = df_result['lambda_est'].min()
    print(f"lambda_0 (наименьшая интенсивность): {lambda_0}")

    # Построение objective-функции с параметрами
    k_range=(5000, 10000)
    h_range = (140, 200)
    tolerance = 2
    n_trials = 100

    objective_fn = make_objective(
        df_gen,
        lambda_0,
        true_change_points,
        k_range=k_range,
        h_range=h_range,
        tolerance=tolerance
    )

    # Оптимизация через Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn, n_trials)

    # === 6. Запуск Optuna с прогрессбаром
    study = optuna.create_study(direction="minimize")
    for _ in tqdm(range(n_trials), desc="Optimizing trials"):
        study.optimize(objective_fn, n_trials=1, catch=(Exception,))

    # Сбор и отображение результатов
    results = []
    for trial in study.trials:
        results.append({
            "k": trial.params["k"],
            "h": trial.params["h"],
            "matches": -trial.value
        })
    df_res = pd.DataFrame(results)

    # Тепловая карта результатов
    pivot = df_res.pivot_table(index="k", columns="h", values="matches", aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Количество совпадений CUSUM с реальными сменами λ")
    plt.xlabel("Порог h")
    plt.ylabel("Параметр чувствительности k")
    plt.tight_layout()
    plt.show()

    # Вывод лучших параметров
    print("📌 Лучшие параметры:", study.best_params)
    print("✅ Совпадений:", -study.best_value)
    #Лучшие параметры: {'k': 5021.904234146086, 'h': 152.29066096058415}


    # Используем лучшие параметры
    best_k = study.best_params['k']
    best_h = study.best_params['h']

    #best_k = 5021.904234146086
    #best_h = 152.29066096058415

    # Сокращение ложных срабатываний с min_gap=30
    alerts = detect_cusum_changes(df_gen, lambda_0, best_k, best_h, min_gap=15)

    print(f"Обнаружено сигналов: {len(alerts)}")
    print(alerts)

    # === Подготовка таблиц данных ===
    true_table_data = []
    for i, (switch_idx, t) in enumerate(zip(cumulative_lengths, switch_times)):
        t_pd = pd.to_datetime(t)
        lam_val = df_gen.loc[switch_idx, 'lambda_est']
        true_table_data.append([
            f"{i + 1}",
            f"{switch_idx}",
            t_pd.strftime("%Y-%m-%d %H:%M"),
            f"{lam_val:.6f}"
        ])

    cusum_table_data = []
    for i, (_, row) in enumerate(alerts.iterrows()):
        cusum_table_data.append([
            f"{i + 1}",
            f"{int(row['event_index'])}",
            pd.to_datetime(row['START_TIME']).strftime("%Y-%m-%d %H:%M"),
            f"{row['TIME_DIFF']:.2f}"
        ])

    # === Создание фигуры ===
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 1)

    # === График накопленного числа событий ===
    ax = fig.add_subplot(gs[0])
    ax.plot(df_gen["START_TIME"], df_gen["INDEX"], label="Накопленное число событий")

    # Флаги для отображения подписи в легенде только один раз
    red_labeled = False
    green_labeled = False
    blue_labeled = False

    # Красные и зелёные линии (смены λ)
    for switch_idx, t in zip(cumulative_lengths, switch_times):
        lam_val = df_gen.loc[switch_idx, 'lambda_est']
        t_pd = pd.to_datetime(t)

        if lam_val == min_lambda:
            ax.axvline(
                x=t_pd,
                color='green',
                linestyle='--',
                linewidth=1.8,
                label='Переход к min λ' if not green_labeled else None
            )
            green_labeled = True
        else:
            ax.axvline(
                x=t_pd,
                color='red',
                linestyle='--',
                linewidth=1,
                label='Смена λ' if not red_labeled else None
            )
            red_labeled = True

    # Синие линии (CUSUM сигналы)
    for _, row in alerts.iterrows():
        ax.axvline(
            x=row['START_TIME'],
            color='blue',
            linestyle='--',
            linewidth=1,
            label='CUSUM' if not blue_labeled else None
        )
        blue_labeled = True

    # Оформление
    ax.set_title("График накопленного числа событий")
    ax.set_xlabel("Время события")
    ax.set_ylabel("Нормированное значение")
    ax.set_xlim(df_gen['START_TIME'].iloc[0], df_gen['START_TIME'].iloc[-1])
    ax.set_ylim(0, 1)
    ax.grid()
    ax.legend()
    # # === Таблица 1: Истинные смены λ ===
    # ax2 = fig.add_subplot(gs[1])
    # ax2.axis('off')
    # col_labels_1 = ["#", "Индекс", "Время", "λ"]
    # ax2.table(cellText=true_table_data, colLabels=col_labels_1, loc='center', cellLoc='center')
    #
    # # === Таблица 2: Сигналы CUSUM ===
    # ax3 = fig.add_subplot(gs[2])
    # ax3.axis('off')
    # col_labels_2 = ["#", "Индекс", "Время", "TIME_DIFF"]
    # ax3.table(cellText=cusum_table_data, colLabels=col_labels_2, loc='center', cellLoc='center')

    # === Сохранение и вывод ===
    plt.tight_layout()
    plt.savefig(diff_plot_file_path, dpi=300, bbox_inches='tight')
    plt.show()





