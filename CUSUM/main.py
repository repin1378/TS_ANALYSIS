from pathlib import Path
import pandas as pd
from modules.converter import convert_excels
from modules.filter_manager import create_filters, load_filtered_dataframe
from modules.report_counter import generate_count_reports, generate_time_distribution_report
from modules.data_loader import get_df_full_filter, get_df_multi_year
from modules.preprocess import preprocess_dataframe, save_histogram, plot_cumulative_events, plot_cumulative_events_with_lambda
from modules.seasonal_lambda import estimate_lambda_for_season
from modules.synthetic_year_generator import generate_synthetic_year, generate_synthetic_year_with_spikes_smooth

def main():

    # # Конвертировать xlsx в csv
    # input_dir = Path("KASANT/modify")
    # output_dir = Path("KASANT/csv")
    # convert_excels(input_dir, output_dir)

    # # Создание фильтров
    csv_dir = Path("KASANT/csv")
    # filters_dir = Path("KASANT/filters")
    #
    # # Создать фильтры
    # create_filters(csv_dir, filters_dir)
    #
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

    lambda_dir = Path("KASANT/calculation/lambda_0")
    synthetic_dir = Path("KASANT/calculation/synthetic_spike")
    graphs_dir = Path("KASANT/calculation/graphs_spike")

    tasks = [
        ("Октябрьская",        [2, 3], 2025),
        ("Восточно-Сибирская", [2, 3], 2025),
        ("Приволжская",        [2, 3], 2025),
        ("Северная",           [3],    2025),
    ]

    # ==== параметры всплеска ====
    delta = 5               # λ1 = δ·λ0 → удвоение интенсивности
    spike_days = 40         # длительность всплеска
    transition_days = 15     # длительность экспоненциального перехода
    k = 2                   # крутизна экспоненты

    for road, categories, year in tasks:

        print("\n" + "=" * 90)
        print(f"ГЕНЕРАЦИЯ СИНТЕТИКИ СО ВСПЛЕСКАМИ — дорога: {road}, год: {year}")

        for category in categories:

            print(f"\n  Категория: {category}")

            # === 1. Генерируем синтетический год со всплесками ===
            df_syn = generate_synthetic_year_with_spikes_smooth(
                lambda_dir=lambda_dir,
                road=road,
                category=str(category),
                year=year,
                out_dir=synthetic_dir,
                delta=delta,
                spike_days=spike_days,
                transition_days=transition_days,
                k=k
            )

            file_path = synthetic_dir / f"synthetic_smooth_{road}_{category}_{year}.csv"
            print(f"    → Файл синтетики: {file_path}")

            # === 2. Читаем синтетический CSV ===
            df = pd.read_csv(file_path, parse_dates=["START_TIME"])

            # === 3. Гистограмма интервалов TIME_DIFF ===
            print(f"    → Построение гистограммы…")
            save_histogram(df, graphs_dir, file_path.stem)

            # === 4. График накопленного числа событий ===
            print(f"    → Построение графика НЧС…")
            plot_cumulative_events(df, graphs_dir, file_path.stem)

            print(f"    ✔ Готово для {road}, категория {category}, год {year}")

            print(f"    → Построение графика НЧС + λ(t)…")
            # 5. НЧС + λ(t)
            plot_cumulative_events_with_lambda(df, graphs_dir, file_path.stem)

            print(f"    ✔ Готово для {road}, категория {category}, год {year}")



if __name__ == "__main__":
    main()
