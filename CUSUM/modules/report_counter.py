import pandas as pd
from pathlib import Path

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
            csv_dir: Path,
            road: str = None,
            category: str = None,
            year: str = None,
            reports_dir: Path = None
    ):
        """
        Генерация отчётов:
          1) Кол-во событий по сезонам
          2) Кол-во событий по месяцам

        Фильтры:
          - road: строка (обязательно)
          - category: строка или None
          - year: строка или None (если None → все годы)

        Сохраняет два CSV в reports/count
        """

        if reports_dir is None:
            reports_dir = Path("reports/count")
        reports_dir.mkdir(parents=True, exist_ok=True)

        files = list(csv_dir.glob("*.csv"))
        dfs = []

        for f in files:
            # имя вида CT_2023.csv
            parts = f.stem.split("_")
            if len(parts) < 2:
                continue

            dept, yr = parts[0], parts[1]

            # фильтр по году
            if year and yr != str(year):
                continue

            df = pd.read_csv(f)
            df["DEPARTMENT"] = dept
            df["YEAR"] = yr
            df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")

            # фильтр по дороге
            if road is not None:
                df = df[df["ROAD"] == road]

            # фильтр по категории
            if category is not None:
                df = df[df["CATEGORY"].astype(str) == str(category)]

            if len(df) > 0:
                dfs.append(df)

        # Если нет данных
        if not dfs:
            print("⚠️ Нет событий по указанным фильтрам")
            return None

        df_all = pd.concat(dfs, ignore_index=True)

        # ============================================================
        # 1) Группировка по сезонам
        # ============================================================

        # Определяем сезон по месяцу
        def month_to_season(m):
            if m in (3, 4, 5):
                return "Весна"
            if m in (6, 7, 8):
                return "Лето"
            if m in (9, 10, 11):
                return "Осень"
            return "Зима"  # декабрь, январь, февраль

        df_all["SEASON"] = df_all["START_TIME"].dt.month.apply(month_to_season)

        by_season = (
            df_all.groupby("SEASON")
                .size()
                .reset_index(name="COUNT")
                .sort_values("COUNT", ascending=False)
        )

        # сохранение отчёта
        season_path = reports_dir / f"season_{road}_{category}_{year}.csv"
        by_season.to_csv(season_path, index=False, encoding="utf-8-sig")

        # ============================================================
        # 2) Группировка по месяцам
        # ============================================================

        df_all["MONTH"] = df_all["START_TIME"].dt.month

        by_month = (
            df_all.groupby("MONTH")
                .size()
                .reset_index(name="COUNT")
                .sort_values("MONTH")
        )

        month_path = reports_dir / f"month_{road}_{category}_{year}.csv"
        by_month.to_csv(month_path, index=False, encoding="utf-8-sig")

        print("📊 Отчёты созданы:")
        print("  -", season_path)
        print("  -", month_path)

        return {
            "season": by_season,
            "month": by_month
        }