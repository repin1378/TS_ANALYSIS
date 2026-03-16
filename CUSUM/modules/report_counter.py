import pandas as pd
from pathlib import Path
import json

def generate_count_reports(csv_dir: Path, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = list(csv_dir.glob("*.csv"))
    if not files:
        print("❌ Нет CSV-файлов в папке:", csv_dir)
        return

    all_dfs = []

    for f in files:
        # Имя вида CT_2023.csv → CT, 2023
        parts = f.stem.split("_")
        if len(parts) < 2:
            print(f"⚠️ Пропуск: имя файла не соответствует формату DEPT_YEAR → {f.name}")
            continue

        dept, year = parts[0], parts[1]

        df = pd.read_csv(f)
        df["DEPARTMENT"] = dept
        df["YEAR"] = year

        all_dfs.append(df)

    if not all_dfs:
        print("❌ Нет подходящих данных.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # ======================================================
    #  1) Отчёт: DEPARTMENT, YEAR, ROAD
    # ======================================================
    by_road = (
        df_all.groupby(["DEPARTMENT", "YEAR", "ROAD"])
              .size()
              .reset_index(name="COUNT")
              .sort_values(["COUNT", "DEPARTMENT", "YEAR"], ascending=[False, True, True])
    )

    by_road.to_csv(reports_dir / "by_road.csv", index=False, encoding="utf-8-sig")

    # ======================================================
    #  2) Отчёт: DEPARTMENT, YEAR, ROAD, CATEGORY
    # ======================================================
    by_road_cat = (
        df_all.groupby(["DEPARTMENT", "YEAR", "ROAD", "CATEGORY"])
              .size()
              .reset_index(name="COUNT")
              .sort_values(["COUNT", "DEPARTMENT", "YEAR"], ascending=[False, True, True])
    )

    by_road_cat.to_csv(reports_dir / "by_road_category.csv", index=False, encoding="utf-8-sig")

    print("🎉 Отчёты созданы:")
    print("  -", reports_dir / "by_road.csv")
    print("  -", reports_dir / "by_road_category.csv")

def generate_time_distribution_report(
    all_csv_path: Path,
    departments_json_path: Path,
    roads_json_path: Path,
    years_json_path: Path,
    output_dir: Path,
) -> None:
    """
    Формирует отчёты по распределению количества событий по сезонам.

    Параметры:
        all_csv_path: путь до общего CSV-файла
        departments_json_path: путь до departments.json
        roads_json_path: путь до roads.json
        years_json_path: путь до years.json
        output_dir: путь до папки для результирующих CSV

    Результат:
        output_dir/
            by_department/
                seasonal_distribution_by_department_2023.csv
                seasonal_distribution_by_department_2024.csv
                ...
            by_road/
                seasonal_distribution_by_road_2023.csv
                seasonal_distribution_by_road_2024.csv
                ...
            by_year/
                seasonal_distribution_by_year.csv
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    by_department_dir = output_dir / "by_department"
    by_road_dir = output_dir / "by_road"
    by_year_dir = output_dir / "by_year"

    by_department_dir.mkdir(parents=True, exist_ok=True)
    by_road_dir.mkdir(parents=True, exist_ok=True)
    by_year_dir.mkdir(parents=True, exist_ok=True)

    if not all_csv_path.exists():
        print(f"❌ Общий CSV не найден: {all_csv_path}")
        return

    for p in (departments_json_path, roads_json_path, years_json_path):
        if not p.exists():
            print(f"❌ Файл фильтра не найден: {p}")
            return

    with open(departments_json_path, "r", encoding="utf-8") as f:
        departments = json.load(f)

    with open(roads_json_path, "r", encoding="utf-8") as f:
        roads = json.load(f)

    with open(years_json_path, "r", encoding="utf-8") as f:
        years = json.load(f)

    print(f"→ Чтение общего файла: {all_csv_path}")
    df = pd.read_csv(all_csv_path, encoding="utf-8-sig")

    required_columns = {"START_TIME", "DEPARTMENT", "ROAD"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        print(f"❌ В CSV отсутствуют обязательные колонки: {sorted(missing_cols)}")
        return

    # Приводим START_TIME к datetime
    df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
    df = df[df["START_TIME"].notna()].copy()

    if df.empty:
        print("⚠️ После преобразования START_TIME не осталось валидных строк")
        return

    # Если YEAR уже есть — используем, иначе извлекаем из START_TIME
    if "YEAR" not in df.columns:
        df["YEAR"] = df["START_TIME"].dt.year
    else:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").fillna(
            df["START_TIME"].dt.year
        ).astype("Int64")

    # Нормализация строковых полей
    for col in ("DEPARTMENT", "ROAD"):
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Берём только значения из фильтров
    df = df[
        df["DEPARTMENT"].isin(departments)
        & df["ROAD"].isin(roads)
        & df["YEAR"].isin(years)
    ].copy()

    if df.empty:
        print("⚠️ Нет данных после применения фильтров JSON")
        return

    # Определение сезона
    def month_to_season(month: int) -> str:
        if month in (12, 1, 2):
            return "winter"
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        return "autumn"

    df["SEASON"] = df["START_TIME"].dt.month.map(month_to_season)

    season_order = ["winter", "spring", "summer", "autumn"]

    # -------- Отчёт по годам --------
    by_year = (
        df.groupby(["YEAR", "SEASON"])
        .size()
        .reset_index(name="EVENT_COUNT")
    )

    by_year["SEASON"] = pd.Categorical(
        by_year["SEASON"], categories=season_order, ordered=True
    )
    by_year = by_year.sort_values(["YEAR", "SEASON"]).reset_index(drop=True)

    by_year_out = by_year_dir / "seasonal_distribution_by_year.csv"
    by_year.to_csv(by_year_out, index=False, encoding="utf-8-sig")
    print(f"  ✅ Сохранён: {by_year_out}")

    # -------- Отчёты по департаментам --------
    for year in years:
        df_year = df[df["YEAR"] == year].copy()
        if df_year.empty:
            print(f"  ⚠️ Нет данных за {year} для отчёта по департаментам")
            continue

        report_dep = (
            df_year.groupby(["DEPARTMENT", "SEASON"])
            .size()
            .reset_index(name="EVENT_COUNT")
        )

        report_dep["SEASON"] = pd.Categorical(
            report_dep["SEASON"], categories=season_order, ordered=True
        )
        report_dep = report_dep.sort_values(["DEPARTMENT", "SEASON"]).reset_index(drop=True)

        out_path = by_department_dir / f"seasonal_distribution_by_department_{year}.csv"
        report_dep.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Сохранён: {out_path}")

    # -------- Отчёты по дорогам --------
    for year in years:
        df_year = df[df["YEAR"] == year].copy()
        if df_year.empty:
            print(f"  ⚠️ Нет данных за {year} для отчёта по дорогам")
            continue

        report_road = (
            df_year.groupby(["ROAD", "SEASON"])
            .size()
            .reset_index(name="EVENT_COUNT")
        )

        report_road["SEASON"] = pd.Categorical(
            report_road["SEASON"], categories=season_order, ordered=True
        )
        report_road = report_road.sort_values(["ROAD", "SEASON"]).reset_index(drop=True)

        out_path = by_road_dir / f"seasonal_distribution_by_road_{year}.csv"
        report_road.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Сохранён: {out_path}")

    print("\n🎉 Отчёты по сезонному распределению успешно сформированы.")