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