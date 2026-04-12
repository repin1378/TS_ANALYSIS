from pathlib import Path
import pandas as pd
from modules.converter import convert_excels, convert_xlsx_to_csv_keep_all_fields, compare_csv_structures, merge_csv_files, remove_nbsp_from_csv
from modules.cusum_exp_seasonal import run_cusum_exp_from_csv, run_cusum_batch
from modules.filter_manager import create_filters, load_filtered_dataframe
from modules.report_counter import generate_count_reports, generate_time_distribution_report
from modules.data_loader import get_df_full_filter, get_df_multi_year
from modules.preprocess import preprocess_dataframe, save_histogram, plot_cumulative_events, plot_cumulative_events_with_spike, plot_cumulative_events_with_cusum_alarms, plot_cusum_batch, estimate_ar1_for_directories
from modules.seasonal_lambda import estimate_lambda_for_season, apply_lambda_from_reference_year
from modules.synthetic_year_generator import generate_synthetic_year, generate_synthetic_year_with_spikes_smooth, generate_spike_report
from modules.synthetic_enricher import enrich_synthetic_department_csvs, enrich_synthetic_road_csvs
from modules.cusum_threshold import design_cusum_threshold_analytic, estimate_metrics_mc, compare_analytic_vs_mc_extended,save_comparison_to_csv,save_comparison_to_excel, find_h_from_delta_arl1, compute_arl0_from_delta_arl1, build_tables_h_delta_arl1, fit_new_approximations, run_arl0_delta_experiment, run_arl1_delta_experiment, run_arl1_target_delta_experiment, run_arl0_arl1_delta_experiment
from modules.synthetic_for_department import create_department_reason_top_json, create_department_road_top_json, create_department_responsibility_top_json
from modules.synthetic_for_road import create_road_reason_top_json, create_road_department_top_json, create_road_responsibility_top_json
from modules.synthetic_railway_timeout import create_railway_timeout_reference_json, create_railway_timeout_reference_by_field_json
from modules.compare import compare_road_by_year, compare_department_by_year
import numpy as np

def main():

    # # Конвертировать xlsx в csv c сохранением всех полей (даже если они не в стандарте)
    # input_dir = Path("KASANT/original/xlsx")
    # output_dir = Path("KASANT/original/csv")
    # all_events_file = Path("KASANT/original/union/all_events.csv")
    # convert_xlsx_to_csv_keep_all_fields(input_dir, output_dir)
    # #
    # # # Сравнение структур CSV-файлов (проверка на одинаковые поля)
    # # compare_csv_structures(output_dir, strict_order=True)
    # #
    # # Слияние всех CSV в один (для удобства анализа и создания фильтров)
    # merge_csv_files(output_dir, all_events_file)

    # # Удаление неразрывного пробела (NBSP) из объединённого CSV, если он там есть
    # all_events_file = Path("KASANT/original/union/all_events.csv")
    # remove_nbsp_from_csv(all_events_file)
    #
    # # Создание фильтров
    # filters_dir = Path("KASANT/filters")
    # all_events_file = Path("KASANT/original/union/all_events.csv")
    #
    # # Создать фильтры
    # create_filters(all_events_file, filters_dir)

    # # Отчет с подсчетом событий
    # generate_time_distribution_report(
    #     all_csv_path=Path("KASANT/original/union/all_events.csv"),
    #     departments_json_path=Path("KASANT/filters/departments.json"),
    #     roads_json_path=Path("KASANT/filters/roads.json"),
    #     years_json_path=Path("KASANT/filters/years.json"),
    #     output_dir=Path("KASANT/reports/count"),
    # )

    # # Получение отфильтрованных DataFrame для разных комбинаций DEPARTMENT, YEAR, ROAD, CATEGORY
    # get_df_full_filter(
    #     all_csv_path=Path("KASANT/original/union/all_events.csv"),
    #     departments_json_path=Path("KASANT/filters/departments.json"),
    #     roads_json_path=Path("KASANT/filters/roads.json"),
    #     years_json_path=Path("KASANT/filters/years.json"),
    #     output_dir=Path("KASANT/data/filtered_by_department_year"),
    # )

    # # Создание полей для гистограммы и НЧС
    # # 1. Обработка CSV
    # preprocess_dataframe(
    #     source_dirs=[
    #         Path("KASANT/data/filtered_by_department_year/by_department_year"),
    #         Path("KASANT/data/filtered_by_department_year/by_road_year")
    #     ],
    #     save_dirs=[
    #         Path("KASANT/processed/by_department_year"),
    #         Path("KASANT/processed/by_road_year")
    #     ]
    # )

    # # 2. Гистограмма
    # save_histogram(
    #     source_dirs=[
    #         Path("KASANT/processed/by_department_year"),
    #         Path("KASANT/processed/by_road_year"),
    #     ],
    #     graph_dirs=[
    #         Path("KASANT/graphs/by_department_year"),
    #         Path("KASANT/graphs/by_road_year"),
    #     ],
    # )
    #
    # # 3. График накопленного числа событий
    # plot_cumulative_events(
    #     source_dirs=[
    #         Path("KASANT/processed/by_department_year"),
    #         Path("KASANT/processed/by_road_year"),
    #     ],
    #     graph_dirs=[
    #         Path("KASANT/graphs/by_department_year/cusum"),
    #         Path("KASANT/graphs/by_road_year/cusum"),
    #     ],
    # )

    # # Расчет AR(1) для всех DataFrame в указанных директориях и сохранение отчёта в CSV
    # estimate_ar1_for_directories(
    #     source_dirs=[
    #         Path("KASANT/processed/by_department_year"),
    #         Path("KASANT/processed/by_road_year"),
    #     ],
    #     output_csv=Path("KASANT/calculation/correlation/ar1_report.csv"),
    #     ljung_box_lags=10
    # )

#===========================Генерация синтетических данных============================================================================================

    # #======= Создание JSON для Департаментов =======
    #
    # create_department_reason_top_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_path=Path("KASANT/calculation/synthetic_for_department_json/department_reason_top.json"),
    #     top_n=20,
    # )
    #
    # create_department_road_top_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_path=Path("KASANT/calculation/synthetic_for_department_json/deppartment_road_top.json"),
    #     top_n=10,
    # )
    #
    # create_department_responsibility_top_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_path=Path("KASANT/calculation/synthetic_for_department_json/department_responsibility_top.json"),
    #     top_n=15,
    # )
    #
    # #====== Создание JSON для Дорог =======
    #
    # create_road_reason_top_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_path=Path("KASANT/calculation/synthetic_for_road_json/road_reason_top.json"),
    #     top_n=5,
    # )
    #
    # create_road_department_top_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_path=Path("KASANT/calculation/synthetic_for_road_json/road_department_top.json"),
    #     top_n=5,
    # )
    #
    # create_road_responsibility_top_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_path=Path("KASANT/calculation/synthetic_for_road_json/road_responsibility_top.json"),
    #     top_n=5,
    # )
    #
    # #====== Создание JSON для железнодорожных простоев =======
    #
    # create_railway_timeout_reference_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_path=Path("KASANT/calculation/synthetic_for_timeout_json/railway_timeout_reference.json"),
    # )
    #
    # create_railway_timeout_reference_by_field_json(
    #     csv_path=Path("KASANT/original/union/all_events.csv"),
    #     output_json_dir=Path("KASANT/calculation/synthetic_for_timeout_json/railway_timeout"),
    # )

    # #====== Оценка λ₀ для сезонов по департаментам и дорогам =======
    #
    # estimate_lambda_for_season(
    #     by_department_year_dir=Path("KASANT/processed/by_department_year"),
    #     by_road_year_dir=Path("KASANT/processed/by_road_year"),
    #     output_department_dir=Path("KASANT/calculation/lambda_0/by_department_year"),
    #     output_road_dir=Path("KASANT/calculation/lambda_0/by_road_year"),
    #     min_points=30,
    #     ks_alpha=0.05,
    # )

    # #====== Генерация синтетических данных для 2025 года на основе оценок λ₀ =======
    #
    # generate_synthetic_year(
    #     source_year=2024,
    #     synthetic_year=2025,
    #     input_department_dir=Path("KASANT/calculation/lambda_0/by_department_year"),
    #     input_road_dir=Path("KASANT/calculation/lambda_0/by_road_year"),
    #     output_department_dir=Path("KASANT/calculation/synthetic/by_department_year"),
    #     output_road_dir=Path("KASANT/calculation/synthetic/by_road_year"),
    #     random_seed=42,
    # )
    #
    # #==== Гистограмма интервалов TIME_DIFF для синтетических данных ======
    #
    # save_histogram(
    #     source_dirs=[
    #         Path("KASANT/calculation/synthetic/by_department_year"),
    #         Path("KASANT/calculation/synthetic/by_road_year"),
    #     ],
    #     graph_dirs=[
    #         Path("KASANT/calculation/graphs/by_department_year"),
    #         Path("KASANT/calculation/graphs/by_road_year"),
    #     ],
    # )
    #
    # #==== График накопленного числа событий для синтетических данных ======
    #
    # plot_cumulative_events(
    #     source_dirs=[
    #         Path("KASANT/calculation/synthetic/by_department_year"),
    #         Path("KASANT/calculation/synthetic/by_road_year"),
    #     ],
    #     graph_dirs=[
    #         Path("KASANT/calculation/graphs/cusum/by_department_year"),
    #         Path("KASANT/calculation/graphs/cusum/by_road_year/cusum"),
    #     ],
    # )
    #

    # #==== Генерация синтетических данных для 2025 года со всплесками на основе оценок λ₀ ======
    #
    # generate_synthetic_year_with_spikes_smooth(
    #     source_year=2024,
    #     synthetic_year=2025,
    #     input_department_dir=Path("KASANT/calculation/lambda_0/by_department_year"),
    #     input_road_dir=Path("KASANT/calculation/lambda_0/by_road_year"),
    #     output_department_dir=Path("KASANT/calculation/synthetic_spike/by_department_year"),
    #     output_road_dir=Path("KASANT/calculation/synthetic_spike/by_road_year"),
    #     delta=3.0,
    #     spike_days=20,
    #     transition_days=5,
    #     k=2.0,
    #     random_seed=42,
    # )
    #
    # #===== Создание отчета по синтетическим данным со всплесками ======
    #
    # generate_spike_report(
    #     input_department_dir=Path("KASANT/calculation/synthetic_spike/by_department_year"),
    #     input_road_dir=Path("KASANT/calculation/synthetic_spike/by_road_year"),
    #     output_department_dir=Path("KASANT/calculation/synthetic_spike_report/by_department_year"),
    #     output_road_dir=Path("KASANT/calculation/synthetic_spike_report/by_road_year"),
    # )
    #
    # #==== Гистограмма интервалов TIME_DIFF для синтетических данных с всплесками======
    #
    # save_histogram(
    #     source_dirs=[
    #         Path("KASANT/calculation/synthetic_spike/by_department_year"),
    #         Path("KASANT/calculation/synthetic_spike/by_road_year"),
    #     ],
    #     graph_dirs=[
    #         Path("KASANT/calculation/graphs_spike/by_department_year"),
    #         Path("KASANT/calculation/graphs_spike/by_road_year"),
    #     ],
    # )
    #
    # #==== График накопленного числа событий для синтетических данных со всплесками ======
    # plot_cumulative_events_with_spike(
    #     source_dirs=[
    #         Path("KASANT/calculation/synthetic_spike/by_department_year"),
    #         Path("KASANT/calculation/synthetic_spike/by_road_year"),
    #     ],
    #     graph_dirs=[
    #         Path("KASANT/calculation/graphs_spike/cusum/by_department_year"),
    #         Path("KASANT/calculation/graphs_spike/cusum/by_road_year"),
    #     ],
    # )

    # #==== Обогащение синтетических данных о дорогах для 2025 года на основе топов из реальных данных и правил заполнения полей простоев ======
    #
    # enrich_synthetic_road_csvs(
    #     input_dir=Path("KASANT/calculation/synthetic_spike/by_road_year"),
    #     output_dir=Path("KASANT/calculation/synthetic_spike_enriched/by_road_year"),
    #     road_department_top_json=Path("KASANT/calculation/synthetic_for_road_json/road_department_top.json"),
    #     road_reason_top_json=Path("KASANT/calculation/synthetic_for_road_json/road_reason_top.json"),
    #     road_responsibility_top_json=Path("KASANT/calculation/synthetic_for_road_json/road_responsibility_top.json"),
    #     categories_json=Path("KASANT/filters/categories.json"),
    #     railway_timeout_reference_json=Path("KASANT/calculation/synthetic_for_timeout_json/railway_timeout_reference.json"),
    #     random_seed=42,
    # )
    #
    # #==== Обогащение синтетических данных о дирекциях для 2025 года на основе топов из реальных данных и правил заполнения полей простоев ======
    #
    # enrich_synthetic_department_csvs(
    #     input_dir=Path("KASANT/calculation/synthetic_spike/by_department_year"),
    #     output_dir=Path("KASANT/calculation/synthetic_spike_enriched/by_department_year"),
    #     department_road_top_json=Path("KASANT/calculation/synthetic_for_department_json/deppartment_road_top.json"),
    #     department_reason_top_json=Path("KASANT/calculation/synthetic_for_department_json/department_reason_top.json"),
    #     department_responsibility_top_json=Path("KASANT/calculation/synthetic_for_department_json/department_responsibility_top.json"),
    #     categories_json=Path("KASANT/filters/categories.json"),
    #     railway_timeout_reference_json=Path("KASANT/calculation/synthetic_for_timeout_json/railway_timeout_reference.json"),
    #     random_seed=42,
    # )

    # # ==== Поиск оптимальных параметров для CUSUM ======

    # # Вычисление h для заданных ARL1 и delta
    # # ===== Пути для сохранения результатов =====
    # csv_path = Path(
    #     "KASANT/cusum_optimization/static_new/h_from_delta_arl0.csv"
    # )
    # json_path = Path(
    #     "KASANT/cusum_optimization/dynamic_new/h_from_delta_arl0.json"
    # )
    #
    # # ===== Параметры эксперимента =====
    # deltas = [1.25, 1.5, 2.0, 2.5, 3.0]
    # arl0_targets = [100, 250, 500, 1000]
    #
    # # Количество прогонов Монте-Карло
    # n_runs_mc = 3000
    #
    # # Если хочешь задать вручную — можно,
    # # но теперь автоподбор h_grid работает корректно
    # h_grid = None
    #
    # # ===== Запуск экспериментов =====
    # for delta in deltas:
    #
    #     print("\n" + "=" * 80)
    #     print(f"DELTA = {delta}")
    #     print("=" * 80)
    #
    #     for arl0 in arl0_targets:
    #         print(f"\n--- ARL0_target = {arl0} ---")
    #
    #         # -------------------------------------------------
    #         # 2) Практический режим (MC + ограничения)
    #         # -------------------------------------------------
    #         run_arl0_delta_experiment(
    #             arl0_target=arl0,
    #             delta_target=delta,
    #             n_runs_mc=n_runs_mc,
    #             csv_path=csv_path,  # в этом режиме CSV не используется
    #             json_path=json_path,
    #             mode="mc_optimal_E",
    #             h_grid=h_grid,  # None → автоподбор
    #             arl0_tolerance=0.10,  # ±10% по ARL0
    #             arl1_min_factor=3.0,  # ARL1 ≥ ARL0 / 3
    #             arl1_min_abs=5.0,  # минимум ARL1
    #             arl1_max_factor=0.3,  # ARL1 ≤ 1.2 × ARL0
    #             max_steps=100_000,
    #             n_workers=12,
    #         )

    # # Вычисление h для заданных ARL1 и delta (стандартный режим)
    # # ===== Пути для сохранения результатов =====
    # json_path = Path("KASANT/cusum_optimization/dynamic_new/h_from_delta_arl1.json")
    #
    # deltas = [1.25, 1.5, 2.0, 2.5, 3.0]
    # arl1_targets = [5, 10, 20, 30]
    # n_runs_mc = 3000
    #
    # for delta in deltas:
    #     print(f"\n============ Δ = {delta} ============")
    #     for arl1 in arl1_targets:
    #         print(f"\n--- ARL1_target = {arl1} ---")
    #         run_arl1_target_delta_experiment(
    #             arl1_target=arl1,
    #             delta_target=delta,
    #             n_runs_mc=n_runs_mc,
    #             json_path=json_path,
    #             arl1_tolerance=0.15,
    #             arl0_min_factor=1.5,
    #             max_steps=50_000,
    #             n_workers=12,
    #         )

    # #==== Вычисление h для большого ARL1 (например CSH) ====
    # json_path = Path("KASANT/cusum_optimization/dynamic_new/h_from_delta_arl1.json")
    # deltas       = [2.5, 3.0]
    # arl1_targets = [500, 750, 1000, 1500]   # ~10, ~15, ~20, ~30 дней для CSH
    #
    # for delta in deltas:
    #     print(f"\n{'='*60}")
    #     print(f"  LARGE ARL1 | δ = {delta}")
    #     print(f"{'='*60}")
    #
    #     for arl1 in arl1_targets:
    #         print(f"\n--- ARL1_target = {arl1} ---")
    #
    #         run_arl1_target_delta_experiment(
    #             arl1_target   = arl1,
    #             delta_target  = delta,
    #             n_runs_mc     = 2000,       # прогонов для H1 (точность ARL1)
    #             n_runs_arl0   = 200,        # прогонов для H0 (только для записи)
    #             json_path     = json_path,
    #             arl1_tolerance= 0.20,       # ±20% — шире, т.к. h сетка крупная
    #             arl0_max      = None,       # ← снимаем ограничение сверху по ARL0
    #             arl0_min_factor = None,     # ← снимаем ограничение снизу по ARL0
    #             max_steps     = 200_000,    # для H0; H1 автоматически max(200_000, 20×ARL1)
    #             n_workers     = 12,
    #         )

    # # Запуск оптимизации для заданных ARL0, ARL1 и delta
    # json_path = Path(
    #     "KASANT/cusum_optimization/dynamic_new/h_from_delta_arl0_arl1.json"
    # )
    #
    # # ===== Параметры =====
    # deltas = [1.5, 2.0, 2.5, 3.0]
    # arl0_targets = [100, 150, 200]
    # arl1_targets = [5, 10, 15]
    #
    # # ===== Быстрые параметры MC =====
    # n_runs_mc = 3000
    # # h_grid = np.arange(0.8, 6.6, 0.2)
    #
    # arl0_tolerance = 0.15
    # arl1_tolerance = 0.15
    # # arl0_max = 250          # можно заменить на 300
    #
    # max_steps = 50_000
    # n_workers = 12
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
    #                 # h_grid=h_grid,
    #                 arl0_tolerance=arl0_tolerance,
    #                 arl1_tolerance=arl1_tolerance,
    #                 # arl0_max=arl0_max,
    #                 max_steps=max_steps,
    #                 n_workers=n_workers,
    #             )


    # # ==== Запуск CUSUM (одиночный файл) ======

    # run_cusum_exp_from_csv(
    #     csv_path=Path("KASANT/calculation/synthetic_spike/synthetic_smooth_Октябрьская_3_2025.csv"),
    #     delta_target=1.5,
    #     arl1_target=5.0,
    #     window_size=30,
    #     cooldown_after_alarm=5,
    #     h_json_dir=Path("KASANT/cusum_optimization/dynamic_new"),
    #     out_dir=Path("KASANT/cusum_results/"),
    # )

    # df_full = pd.read_csv("KASANT/cusum_results/synthetic_smooth_Октябрьская_3_2025_cusum_full.csv")

    # plot_cumulative_events_with_cusum_alarms(
    #     df=df_full,
    #     save_dir=Path("KASANT/cusum_results/graphs"),
    #     filename_stem="Октябрьская_3_2025"
    # )

    # # ==== Запуск CUSUM (батч: дороги + департаменты) ======
    # run_cusum_batch(
    #     # ── параметры CUSUM ───────────────────────────────────────────────
    #     delta_target=3.0,           # δ = λ₁/λ₀  (целевое отклонение)
    #     window_size=15,             # окно для оценки λ̂ (число событий)
    #     arl1_target=5.0,           # целевая задержка обнаружения (шаги)
    #     arl0_target=100.0,          # целевой ARL0 (ложные тревоги)
    #     cooldown_after_alarm=30,    # шагов «молчания» после тревоги
    #     # ── таблицы порогов h ─────────────────────────────────────────────
    #     h_json_dir=Path("KASANT/cusum_optimization/dynamic_new"),
    #     # ── входные папки с CSV ───────────────────────────────────────────
    #     roads_csv_dir=Path("KASANT/calculation/synthetic_spike_enriched/by_road_year"),
    #     departments_csv_dir=Path("KASANT/calculation/synthetic_spike_enriched/by_department_year"),
    #     # ── папки для результатов ─────────────────────────────────────────
    #     roads_out_dir=Path("KASANT/cusum_results/by_road_year"),
    #     departments_out_dir=Path("KASANT/cusum_results/by_department_year"),
    #     # ── сводный CSV ───────────────────────────────────────────────────
    #     summary_out_dir=Path("KASANT/cusum_results"),
    #     # ── вывод ─────────────────────────────────────────────────────────
    #     verbose=True,
    #     progress_every=0,           # 0 — не выводить прогресс внутри файлов
    # )

    # # ==== Запуск CUSUM (кастомизация под CSH) ======
    # run_cusum_batch(
    #     # ── параметры CUSUM ───────────────────────────────────────────────
    #     delta_target=3.0,           # δ = λ₁/λ₀  (целевое отклонение)
    #     window_size=15,             # окно для оценки λ̂ (число событий)
    #     arl1_target=500.0,           # целевая задержка обнаружения (шаги)
    #     # arl0_target=15.0,          # целевой ARL0 (ложные тревоги)
    #     cooldown_after_alarm=30,    # шагов «молчания» после тревоги
    #     # ── таблицы порогов h ─────────────────────────────────────────────
    #     h_json_dir=Path("KASANT/cusum_optimization/dynamic_new"),
    #     # ── входные папки с CSV ───────────────────────────────────────────
    #     #roads_csv_dir=Path("KASANT/calculation/synthetic_spike_enriched/by_road_year"),
    #     departments_csv_dir=Path("KASANT/calculation/synthetic_spike_enriched/by_department_year/CSH"),
    #     # ── папки для результатов ─────────────────────────────────────────
    #     # roads_out_dir=Path("KASANT/cusum_results/by_road_year"),
    #     departments_out_dir=Path("KASANT/cusum_results/by_department_year/CSH"),
    #     # ── сводный CSV ───────────────────────────────────────────────────
    #     summary_out_dir=Path("KASANT/cusum_results/CSH"),
    #     # ── вывод ─────────────────────────────────────────────────────────
    #     verbose=True,
    #     progress_every=0,           # 0 — не выводить прогресс внутри файлов
    # )

    # # ==== Построение CUSUM-графиков (батч: дороги + департаменты) ======
    # plot_cusum_batch(
    #     # ── папки с CUSUM-результатами (входные) ─────────────────────────
    #     roads_cusum_dir=Path("KASANT/cusum_results/by_road_year"),
    #     departments_cusum_dir=Path("KASANT/cusum_results/by_department_year"),
    #     # ── папки для сохранения графиков (выходные) ─────────────────────
    #     roads_graph_dir=Path("KASANT/cusum_results/graphs/by_road_year"),
    #     departments_graph_dir=Path("KASANT/cusum_results/graphs/by_department_year"),
    #     # ── колонки (по умолчанию, можно изменить) ───────────────────────
    #     alarm_col="CUSUM_ALARM",    # флаг тревоги CUSUM
    #     spike_col="SPIKE_FLAG",     # флаг периода всплеска (опционально)
    # )

    # # ==== Построение CUSUM-графиков (кастомизация под CSH) ======
    # plot_cusum_batch(
    #     # ── папки с CUSUM-результатами (входные) ─────────────────────────
    #     #roads_cusum_dir=Path("KASANT/cusum_results/by_road_year"),
    #     departments_cusum_dir=Path("KASANT/cusum_results/by_department_year/CSH"),
    #     # ── папки для сохранения графиков (выходные) ─────────────────────
    #     # roads_graph_dir=Path("KASANT/cusum_results/graphs/by_road_year"),
    #     departments_graph_dir=Path("KASANT/cusum_results/graphs/by_department_year/CSH"),
    #     # ── колонки (по умолчанию, можно изменить) ───────────────────────
    #     alarm_col="CUSUM_ALARM",    # флаг тревоги CUSUM
    #     spike_col="SPIKE_FLAG",     # флаг периода всплеска (опционально)
    # )

    # # ==== Сравнение исторических данных с результатами CUSUM — ДОРОГИ ======
    # # Для каждой дороги и каждого CUSUM-аларма:
    # #   - берёт период [начало_сезона → дата_аларма] в CUSUM-году
    # #   - сравнивает с аналогичным периодом в историческом году
    # #   - генерирует HTML-отчёт: out_dir/{Дорога}/alarm_NN_YYYY-MM-DD.html
    #
    # compare_road_by_year(
    #     historical_road_dir=Path("KASANT/processed/by_road_year"),
    #     historical_year=2024,
    #     cusum_road_dir=Path("KASANT/cusum_results/by_road_year"),
    #     out_dir=Path("KASANT/cusum_results/compare/roads"),
    #     top_n_reasons=10,
    #     verbose=True,
    # )

    # # ==== Сравнение исторических данных с результатами CUSUM — ДЕПАРТАМЕНТЫ ======
    # # Отличие от дорог: в таблице «Изменение по ...» показывается ROAD, а не DEPARTMENT
    #
    # compare_department_by_year(
    #     historical_dept_dir=Path("KASANT/processed/by_department_year"),
    #     historical_year=2024,
    #     cusum_dept_dir=Path("KASANT/cusum_results/by_department_year"),
    #     out_dir=Path("KASANT/cusum_results/compare/departments"),
    #     top_n_reasons=10,
    #     verbose=True,
    # )
    #
    # compare_department_by_year(
    #     historical_dept_dir=Path("KASANT/processed/by_department_year/CSH"),
    #     historical_year=2024,
    #     cusum_dept_dir=Path("KASANT/cusum_results/by_department_year/CSH"),
    #     out_dir=Path("KASANT/cusum_results/compare/departments/"),
    #     top_n_reasons=10,
    #     verbose=True,
    # )

    # # ==== Запуск CUSUM за 2024 год с λ₀ из 2023 ======
    # #
    # # Шаг 1. Оценить λ₀ по сезонам для 2023 (если ещё не сделано).
    # #         Результат: KASANT/calculation/lambda_0/by_*_year/lambda_0_{entity}_2023.csv
    # #
    # estimate_lambda_for_season(
    #     by_department_year_dir=Path("KASANT/processed/by_department_year"),
    #     by_road_year_dir=Path("KASANT/processed/by_road_year"),
    #     output_department_dir=Path("KASANT/calculation/lambda_0/by_department_year"),
    #     output_road_dir=Path("KASANT/calculation/lambda_0/by_road_year"),
    #     min_points=30,
    #     ks_alpha=0.05,
    # )

    # # Шаг 2. Перенести λ₀ из 2023 на события 2024:
    # #         - читает lambda_0/*_2023.csv из lambda_0/...
    # #         - читает обработанные события из processed/.../2024/
    # #         - добавляет колонки SEASON и LAMBDA0
    # #         - сохраняет в calculation/lambda_0_2024/...
    # #
    # apply_lambda_from_reference_year(
    #     reference_lambda_dept_dir=Path("KASANT/calculation/lambda_0/by_department_year"),
    #     reference_lambda_road_dir=Path("KASANT/calculation/lambda_0/by_road_year"),
    #     target_dept_dir=Path("KASANT/processed/by_department_year/2024"),
    #     target_road_dir=Path("KASANT/processed/by_road_year/2024"),
    #     output_dept_dir=Path("KASANT/calculation/lambda_0_2024/by_department_year"),
    #     output_road_dir=Path("KASANT/calculation/lambda_0_2024/by_road_year"),
    #     verbose=True,
    # )

    # # Шаг 3. Запуск CUSUM на событиях 2024 с λ₀ из 2023
    # run_cusum_batch(
    #     # ── параметры CUSUM ───────────────────────────────────────────────
    #     delta_target=3.0,           # δ = λ₁/λ₀  (целевое отклонение)
    #     window_size=30,             # окно для оценки λ̂ (число событий)
    #     arl1_target=10,            # целевая задержка обнаружения (шаги)
    #     # arl0_target=100.0,        # целевой ARL0 (ложные тревоги)
    #     cooldown_after_alarm=30,    # шагов «молчания» после тревоги
    #     # ── таблицы порогов h ─────────────────────────────────────────────
    #     h_json_dir=Path("KASANT/cusum_optimization/dynamic_new"),
    #     # ── входные папки с CSV (обогащённые SEASON + LAMBDA0) ───────────
    #     roads_csv_dir=Path("KASANT/calculation/lambda_0_2024/by_road_year"),
    #     departments_csv_dir=Path("KASANT/calculation/lambda_0_2024/by_department_year"),
    #     # ── папки для результатов ─────────────────────────────────────────
    #     roads_out_dir=Path("KASANT/cusum_results/2024/by_road_year"),
    #     departments_out_dir=Path("KASANT/cusum_results/2024/by_department_year"),
    #     # ── сводный CSV ───────────────────────────────────────────────────
    #     summary_out_dir=Path("KASANT/cusum_results/2024"),
    #     # ── вывод ─────────────────────────────────────────────────────────
    #     verbose=True,
    #     progress_every=0,           # 0 — не выводить прогресс внутри файлов
    # )

    # # ==== Построение CUSUM-графиков (батч: дороги + департаменты) ======
    # plot_cusum_batch(
    #     # ── папки с CUSUM-результатами (входные) ─────────────────────────
    #     roads_cusum_dir=Path("KASANT/cusum_results/2024/by_road_year"),
    #     departments_cusum_dir=Path("KASANT/cusum_results/2024/by_department_year"),
    #     # ── папки для сохранения графиков (выходные) ─────────────────────
    #     roads_graph_dir=Path("KASANT/cusum_results/2024/graphs/by_road_year"),
    #     departments_graph_dir=Path("KASANT/cusum_results/2024/graphs/by_department_year"),
    #     # ── колонки (по умолчанию, можно изменить) ───────────────────────
    #     alarm_col="CUSUM_ALARM",    # флаг тревоги CUSUM
    #     spike_col="SPIKE_FLAG",     # флаг периода всплеска (опционально)
    # )

    # ==== Сравнение исторических данных с результатами CUSUM — ДОРОГИ ======
    # Для каждой дороги и каждого CUSUM-аларма:
    #   - берёт период [начало_сезона → дата_аларма] в CUSUM-году
    #   - сравнивает с аналогичным периодом в историческом году
    #   - генерирует HTML-отчёт: out_dir/{Дорога}/alarm_NN_YYYY-MM-DD.html

    compare_road_by_year(
        historical_road_dir=Path("KASANT/processed/by_road_year"),
        historical_year=2023,
        cusum_road_dir=Path("KASANT/cusum_results/2024/by_road_year"),
        out_dir=Path("KASANT/cusum_results/2024/compare/roads"),
        top_n_reasons=10,
        verbose=True,
    )

    # # ==== Сравнение исторических данных с результатами CUSUM — ДЕПАРТАМЕНТЫ ======
    # # Отличие от дорог: в таблице «Изменение по ...» показывается ROAD, а не DEPARTMENT
    # compare_department_by_year(
    #     historical_dept_dir=Path("KASANT/processed/by_department_year"),
    #     historical_year=2023,
    #     cusum_dept_dir=Path("KASANT/cusum_results/2024/by_department_year"),
    #     out_dir=Path("KASANT/cusum_results/2024/compare/departments"),
    #     top_n_reasons=10,
    #     verbose=True,
    # )
    #
    # compare_department_by_year(
    #     historical_dept_dir=Path("KASANT/processed/by_department_year/CSH"),
    #     historical_year=2024,
    #     cusum_dept_dir=Path("KASANT/cusum_results/by_department_year/CSH"),
    #     out_dir=Path("KASANT/cusum_results/compare/departments/"),
    #     top_n_reasons=10,
    #     verbose=True,
    # )

if __name__ == "__main__":
    main()
