from pathlib import Path
from modules.converter import convert_excels
from modules.filter_manager import create_filters, load_filtered_dataframe
from modules.report_counter import generate_count_reports
from modules.data_loader import get_df_full_filter, get_df_multi_year
from modules.preprocess import preprocess_dataframe, save_histogram, plot_cumulative_events

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

    # # Сборка DataFrame
    # save_dir = Path("KASANT/data")
    # # 1) Один год
    # df_one_year = get_df_full_filter(
    #     csv_dir,
    #     department="CSH",
    #     year="2024",
    #     road="Октябрьская",
    #     save_dir=save_dir
    # )
    # print(df_one_year.head())
    # print(df_one_year.count())
    #
    # # 2) Все годы вместе
    # df_multi = get_df_multi_year(
    #     csv_dir,
    #     department="CSH",
    #     road="Октябрьская",
    #     save_dir = save_dir
    # )
    # print(df_multi.head())
    # print(df_multi.count())

    # Создание полей для гистограммы и НЧС
    file = Path("KASANT/data/filtered_department-CSH_year-2024_road-Октябрьская.csv")
    graphs_dir = Path("KASANT/graphs")
    processed_dir = Path("KASANT/processed")

    # 1. Обработка CSV
    df = preprocess_dataframe(file, save_dir=processed_dir)

    # 2. Гистограмма
    save_histogram(df, graphs_dir, file.stem)

    # 3. График накопленного числа событий
    plot_cumulative_events(df, graphs_dir, file.stem)





if __name__ == "__main__":
    main()
