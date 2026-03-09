from pathlib import Path
import pandas as pd
from modules.converter import convert_excels, convert_xlsx_to_csv_keep_all_fields, compare_csv_structures, merge_csv_files, remove_nbsp_from_csv
from modules.cusum_exp_seasonal import run_cusum_exp_from_csv
from modules.filter_manager import create_filters, load_filtered_dataframe
from modules.report_counter import generate_count_reports, generate_time_distribution_report
from modules.data_loader import get_df_full_filter, get_df_multi_year
from modules.preprocess import preprocess_dataframe, save_histogram, plot_cumulative_events, plot_cumulative_events_with_lambda, plot_cumulative_events_with_cusum_alarms
# from modules.seasonal_lambda import estimate_lambda_for_season
from modules.synthetic_year_generator import generate_synthetic_year, generate_synthetic_year_with_spikes_smooth, generate_spike_report
from modules.cusum_threshold import design_cusum_threshold_analytic, estimate_metrics_mc, compare_analytic_vs_mc_extended,save_comparison_to_csv,save_comparison_to_excel, find_h_from_delta_arl1, compute_arl0_from_delta_arl1, build_tables_h_delta_arl1, fit_new_approximations, run_arl0_delta_experiment, run_arl1_delta_experiment, run_arl1_target_delta_experiment, run_arl0_arl1_delta_experiment
import numpy as np

def main():

    # # Конвертировать xlsx в csv c сохранением всех полей (даже если они не в стандарте)
    # input_dir = Path("KASANT/original/xlsx")
    # output_dir = Path("KASANT/original/csv")
    # all_events_file = Path("KASANT/original/union/all_events.csv")
    # convert_xlsx_to_csv_keep_all_fields(input_dir, output_dir)
    #
    # # Сравнение структур CSV-файлов (проверка на одинаковые поля)
    # compare_csv_structures(output_dir, strict_order=True)
    #
    # # Слияние всех CSV в один (для удобства анализа и создания фильтров)
    # merge_csv_files(output_dir, all_events_file)

    # Удаление неразрывного пробела (NBSP) из объединённого CSV, если он там есть
    all_events_file = Path("KASANT/original/union/all_events.csv")
    remove_nbsp_from_csv(all_events_file)

    # Создание фильтров
    filters_dir = Path("KASANT/filters")
    all_events_file = Path("KASANT/original/union/all_events.csv")

    # Создать фильтры
    create_filters(all_events_file, filters_dir)

    # # Отчет с подсчетом событий
    # reports_dir = Path("reports/count")
    # generate_count_reports(csv_dir, reports_dir)

#=======================================================================================================================

    # # Сборка DataFrame
    # # Пример: дорога, категория, год
    # # Октябрьская (категории: 1,2,3)
    # # Восточно-Сибирская (категории: 1,2,3)
    # # Куйбышевская (категории: 2,3) - мало событий
    # # Приволжская (категории: 2,3)
    # # Северная (категории: 3)

    # save_dir = Path("KASANT/data")
    # # 1) Один год
    # df_one_year = get_df_full_filter(
    #     csv_dir,
    #     #department="CSH",
    #     year="2023",
    #     category="3",
    #     road="Приволжская",
    #     save_dir=save_dir
    # )
    # print(df_one_year.head())
    # print(df_one_year.count())
    #
    # # Сборка DataFrame
    # save_dir = Path("KASANT/data")
    # # 1) Один год
    # df_one_year = get_df_full_filter(
    #     csv_dir,
    #     #department="CSH",
    #     year="2024",
    #     category="3",
    #     road="Приволжская",
    #     save_dir=save_dir
    # )
    # print(df_one_year.head())
    # print(df_one_year.count())


    # # 2) Все годы вместе
    # df_multi = get_df_multi_year(
    #     csv_dir,
    #     #department="CSH",
    #     category="3",
    #     road="Октябрьская",
    #     save_dir = save_dir
    # )
    # print(df_multi.head())
    # print(df_multi.count())

#=======================================================================================================================

    # Создание полей для гистограммы и НЧС
    # file = Path("KASANT/data/filtered_year-2024_road-Приволжская_category-3.csv")
    # graphs_dir = Path("KASANT/graphs")
    # processed_dir = Path("KASANT/processed")
    #
    # # 1. Обработка CSV
    # df = preprocess_dataframe(file, save_dir=processed_dir)
    #
    # # 2. Гистограмма
    # save_histogram(df, graphs_dir, file.stem)
    #
    # # 3. График накопленного числа событий
    # plot_cumulative_events(df, graphs_dir, file.stem)

#=======================================================================================================================

    # csv_dir = Path("KASANT/csv")
    # reports_dir = Path("reports/count")
    #
    # # Пример: дорога, категория, год
    # # Октябрьская (категории: 1,2,3)
    # # Восточно-Сибирская (категории: 1,2,3)
    # # Куйбышевская (категории: 2,3) - мало событий
    # # Приволжская (категории: 2,3)
    # # Северная (категории: 3)
    #
    # generate_time_distribution_report(
    #     csv_dir,
    #     road="Северная",
    #     category="3",
    #     year="2023",
    #     reports_dir=reports_dir
    # )

#=======================================================================================================================

    # csv_dir = Path("KASANT/csv")
    # out_dir = Path("KASANT/calculation/lambda_0")
    # out_dir.mkdir(exist_ok=True)
    #
    # seasons = ["Весна", "Лето", "Осень", "Зима"]
    #
    # tasks = [
    #     ("Октябрьская", [2, 3], 2024),
    #     ("Восточно-Сибирская", [2, 3], 2024),
    #     ("Приволжская", [2, 3], 2024),
    #     ("Северная", [3], 2024),
    # ]
    #
    # for road, categories, year in tasks:
    #
    #     rows = []  # ← сюда будем собирать строки для CSV
    #
    #     print("\n" + "=" * 80)
    #     print(f"Дорога: {road}, год: {year}")
    #
    #     for category in categories:
    #         print(f"\n  Категория {category}")
    #
    #         for season in seasons:
    #             print(f"    → сезон: {season}")
    #
    #             try:
    #                 result = estimate_lambda_for_season(
    #                     csv_dir=csv_dir,
    #                     road=road,
    #                     category=str(category),
    #                     season=season,
    #                     target_year=year,
    #                     min_points=30
    #                 )
    #
    #                 print(
    #                     f"      λ_MLE={result['lambda_mle']:.5f}, "
    #                     f"λ_fitter={result['lambda_fitter']:.5f}, "
    #                     f"n={result['n']}, "
    #                     f"KS_p={result['ks_pvalue']:.3f}, "
    #                     f"KS_OK={result['ks_ok']}"
    #                 )
    #
    #                 # Добавляем строку в будущий CSV
    #                 rows.append({
    #                     "ROAD": road,
    #                     "CATEGORY": category,
    #                     "SEASON": season,
    #                     "YEAR": year,
    #                     "N": result["n"],
    #                     "LAMBDA_MLE": result["lambda_mle"],
    #                     "LAMBDA_FITTER": result["lambda_fitter"],
    #                     "LAMBDA_MONTH": result["lambda_month"],
    #                     "KS_PVALUE": result["ks_pvalue"],
    #                     "KS_OK": result["ks_ok"],
    #                 })
    #
    #             except Exception as e:
    #                 print(f"      ⚠ Нет данных ({e})")
    #
    #     # Сохраняем результаты для дороги
    #     if rows:
    #         df_out = pd.DataFrame(rows)
    #         filename = f"lambda_0_{road}.csv"
    #         df_out.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")
    #         print(f"\n📁 Файл сохранён: {out_dir / filename}")
    #     else:
    #         print(f"\n⚠ Нет данных для {road}, CSV не создан.")

#=======================================================================================================================

    # lambda_dir = Path("KASANT/calculation/lambda_0")
    # synthetic_dir = Path("KASANT/calculation/synthetic")
    # graphs_dir = Path("KASANT/calculation/graphs")
    #
    # # === Генерируем синтетику для 2025 года ===
    # tasks = [
    #     ("Октябрьская",       [2, 3], 2025),
    #     ("Восточно-Сибирская", [2, 3], 2025),
    #     ("Приволжская",       [2, 3], 2025),
    #     ("Северная",          [3],    2025),
    # ]
    #
    # for road, categories, year in tasks:
    #
    #     print("\n" + "=" * 90)
    #     print(f"ГЕНЕРАЦИЯ СИНТЕТИКИ — дорога: {road}, год: {year}")
    #
    #     for category in categories:
    #
    #         print(f"\n  Категория: {category}")
    #
    #         # 1. Генерируем синтетический год
    #         df_syn = generate_synthetic_year(
    #             lambda_dir=lambda_dir,
    #             road=road,
    #             category=str(category),
    #             year=year,
    #             out_dir=synthetic_dir
    #         )
    #
    #         file_path = synthetic_dir / f"synthetic_{road}_{category}_{year}.csv"
    #         print(f"    → Файл синтетики: {file_path}")
    #
    #         # 2. Читаем синтетический CSV
    #         df = pd.read_csv(file_path, parse_dates=["START_TIME"])
    #
    #         # 3. Строим гистограмму TIME_DIFF
    #         print(f"    → Построение гистограммы…")
    #         save_histogram(df, graphs_dir, file_path.stem)
    #
    #         # 4. График накопленного числа событий
    #         print(f"    → Построение графика НЧС…")
    #         plot_cumulative_events(df, graphs_dir, file_path.stem)
    #
    #         print(f"    ✔ Готово для {road}, категория {category}, год {year}")

#=======================================================================================================================

    # lambda_dir = Path("KASANT/calculation/lambda_0")
    # synthetic_dir = Path("KASANT/calculation/synthetic_spike")
    # graphs_dir = Path("KASANT/calculation/graphs_spike")
    #
    # tasks = [
    #     ("Октябрьская",        [2, 3], 2025),
    #     ("Восточно-Сибирская", [2, 3], 2025),
    #     ("Приволжская",        [2, 3], 2025),
    #     ("Северная",           [3],    2025),
    # ]
    #
    # # ==== параметры всплеска ====
    # delta = 3               # λ1 = δ·λ0 → удвоение интенсивности
    # spike_days = 30         # длительность всплеска
    # transition_days = 10     # длительность экспоненциального перехода
    # k = 2                   # крутизна экспоненты
    #
    # for road, categories, year in tasks:
    #
    #     print("\n" + "=" * 90)
    #     print(f"ГЕНЕРАЦИЯ СИНТЕТИКИ СО ВСПЛЕСКАМИ — дорога: {road}, год: {year}")
    #
    #     for category in categories:
    #
    #         print(f"\n  Категория: {category}")
    #
    #         # === 1. Генерируем синтетический год со всплесками ===
    #         df_syn = generate_synthetic_year_with_spikes_smooth(
    #             lambda_dir=lambda_dir,
    #             road=road,
    #             category=str(category),
    #             year=year,
    #             out_dir=synthetic_dir,
    #             delta=delta,
    #             spike_days=spike_days,
    #             transition_days=transition_days,
    #             k=k
    #         )
    #
    #         file_path = synthetic_dir / f"synthetic_smooth_{road}_{category}_{year}.csv"
    #         print(f"    → Файл синтетики: {file_path}")
    #
    #         # === 2. Читаем синтетический CSV ===
    #         df = pd.read_csv(file_path, parse_dates=["START_TIME"])
    #
    #         # === 3. Гистограмма интервалов TIME_DIFF ===
    #         print(f"    → Построение гистограммы…")
    #         save_histogram(df, graphs_dir, file_path.stem)
    #
    #         # === 4. График накопленного числа событий ===
    #         print(f"    → Построение графика НЧС…")
    #         plot_cumulative_events(df, graphs_dir, file_path.stem)
    #
    #         print(f"    ✔ Готово для {road}, категория {category}, год {year}")
    #
    #         print(f"    → Построение графика НЧС + λ(t)…")
    #         # 5. НЧС + λ(t)
    #         plot_cumulative_events_with_lambda(df, graphs_dir, file_path.stem)
    #
    #         print(f"    ✔ Готово для {road}, категория {category}, год {year}")
    #
    #         # сохраняем мини-отчёт
    #         report_dir = Path("KASANT/calculation/reports/synthetic_spike_reports")
    #         generate_spike_report(df_syn, report_dir, road, category, year)

#=======================================================================================================================

    # rows = compare_analytic_vs_mc_extended(
    #     deltas=[1.25, 1.5, 2.0, 2.5, 3.0],
    #     arl0_targets=[100, 250, 500, 1000],
    #     n_runs_arl0=500,
    #     n_runs_arl1=500
    # )
    #
    # save_comparison_to_csv(rows, Path("KASANT/cusum_optimization/cusum_threshold_report_500.csv"))
    # save_comparison_to_excel(rows, Path("KASANT/cusum_optimization/cusum_threshold_report_500.xlsx"))

#=======================================================================================================================

    # ===== Пути для сохранения результатов =====
    csv_path = Path(
        "KASANT/cusum_optimization/static/h_from_delta_arl0.csv"
    )
    json_path = Path(
        "KASANT/cusum_optimization/dynamic/h_from_delta_arl0.json"
    )

    # ===== Параметры эксперимента =====
    deltas = [1.5, 2.0, 2.5, 3.0]
    arl0_targets = [100, 150, 200, 250]

    # Количество прогонов Монте-Карло
    n_runs_mc = 3000

    # Если хочешь задать вручную — можно,
    # но теперь автоподбор h_grid работает корректно
    h_grid = None

    # ===== Запуск экспериментов =====
    # for delta in deltas:
    #
    #     print("\n" + "=" * 80)
    #     print(f"DELTA = {delta}")
    #     print("=" * 80)
    #
    #     for arl0 in arl0_targets:
    #         print(f"\n--- ARL0_target = {arl0} ---")

            # -------------------------------------------------
            # 1) Аналитический режим (повторение статьи)
            # -------------------------------------------------
            # run_arl0_delta_experiment(
            #     arl0_target=arl0,
            #     delta_target=delta,
            #     n_runs_mc=n_runs_mc,
            #     csv_path=csv_path,
            #     json_path=json_path,
            #     mode="analytic",
            #     max_steps=100_000,
            #     n_workers=6,
            # )

            # -------------------------------------------------
            # 2) Практический режим (MC + ограничения)
            # -------------------------------------------------
            # run_arl0_delta_experiment(
            #     arl0_target=arl0,
            #     delta_target=delta,
            #     n_runs_mc=n_runs_mc,
            #     csv_path=csv_path,  # в этом режиме CSV не используется
            #     json_path=json_path,
            #     mode="mc_optimal_E",
            #     h_grid=h_grid,  # None → автоподбор
            #     arl0_tolerance=0.10,  # ±10% по ARL0
            #     arl1_min_factor=3.0,  # ARL1 ≥ ARL0 / 3
            #     arl1_min_abs=5.0,  # минимум ARL1
            #     arl1_max_factor=1.2,  # ARL1 ≤ 1.2 × ARL0
            #     max_steps=100_000,
            #     n_workers=6,
            # )

    # json_path = Path("KASANT/cusum_optimization/dynamic/h_from_delta_arl1.json")
    #
    #
    # # Значения, которые ты указал:
    # deltas = [1.5, 2.0, 2.5, 3.0]
    # arl1_targets = [10, 20, 30, 40]
    # arl0_targets = [100, 150, 200, 250]
    #
    # # Количество прогонов Монте-Карло
    # n_runs_mc = 3000
    #
    # # Если нужен явный диапазон — можно задать, но теперь это необязательно:
    # # h_grid = np.arange(0.5, 8.5, 0.1)
    #
    # for delta in deltas:
    #     print(f"\n============ Δ = {delta} ============")
    #
    #     for arl1 in arl1_targets:
    #
    #         print(f"\n--- ARL1_target = {arl1} ---")
    #
    #         # Запуск улучшенной оптимизации
    #         run_arl1_target_delta_experiment(
    #             arl1_target=arl1,
    #             delta_target=delta,
    #             n_runs_mc=n_runs_mc,
    #             json_path=json_path,
    #             # h_grid=h_grid,           # авто-подбор h_grid
    #             arl1_tolerance=0.10,   # ±20% окно для ARL1
    #             arl0_max=200.0,        # ограничиваем слишком большие ARL0
    #             arl0_min_factor=1.5,   # ARL0_min = ARL1_target * 1.5
    #             max_steps=100000,
    #             n_workers=6            # параллелизм
    #         )

    # ===== Пути =====
    # json_path = Path(
    #     "KASANT/cusum_optimization/fast/h_from_delta_arl0_arl1_fast.json"
    # )
    #
    # # ===== Параметры =====
    # deltas = [1.5, 2.0, 2.5, 3.0]
    # arl0_targets = [100, 150, 200, 250]
    # arl1_targets = [10, 20, 30, 40]
    #
    # # ===== Быстрые параметры MC =====
    # n_runs_mc = 1000
    # h_grid = np.arange(0.8, 6.6, 0.2)
    #
    # arl0_tolerance = 0.25
    # arl1_tolerance = 0.25
    # arl0_max = 250          # можно заменить на 300
    #
    # max_steps = 80_000
    # n_workers = 6
    #
    # # ===== Запуск =====
    # for delta in deltas:
    #     print("\n" + "=" * 90)
    #     print(f"DELTA = {delta}")
    #     print("=" * 90)
    #
    #     for arl0 in arl0_targets:
    #         for arl1 in arl1_targets:
    #             print(f"\n--- δ={delta}, ARL0≈{arl0}, ARL1≈{arl1} ---")
    #
    #             run_arl0_arl1_delta_experiment(
    #                 delta_target=delta,
    #                 arl0_target=arl0,
    #                 arl1_target=arl1,
    #                 n_runs_mc=n_runs_mc,
    #                 json_path=json_path,
    #                 h_grid=h_grid,
    #                 arl0_tolerance=arl0_tolerance,
    #                 arl1_tolerance=arl1_tolerance,
    #                 arl0_max=arl0_max,
    #                 max_steps=max_steps,
    #                 n_workers=n_workers,
    #             )

    # run_cusum_exp_from_csv(
    #     csv_path=Path("KASANT/calculation/synthetic_spike/synthetic_smooth_Октябрьская_3_2025.csv"),
    #     delta_target=3.0,
    #     arl0_target=200,
    #     window_size=30,
    #     cooldown_after_alarm=30,
    #     h_json_dir=Path("KASANT/cusum_optimization/dynamic"),
    #     out_dir=Path("KASANT/cusum_results/"),
    # )
    #
    # df_full = pd.read_csv("KASANT/cusum_results/synthetic_smooth_Октябрьская_3_2025_cusum_full.csv")
    #
    # plot_cumulative_events_with_cusum_alarms(
    #     df=df_full,
    #     save_dir=Path("KASANT/cusum_results/graphs"),
    #     filename_stem="Октябрьская_3_2025"
    # )


if __name__ == "__main__":
    main()
