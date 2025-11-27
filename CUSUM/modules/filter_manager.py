import pandas as pd
from pathlib import Path
import json

def create_filters(csv_dir: Path, filters_dir: Path):
    filters_dir.mkdir(parents=True, exist_ok=True)

    # Находим все CSV
    files = list(csv_dir.glob("*.csv"))
    if not files:
        print("❌ Нет CSV-файлов в каталоге", csv_dir)
        return

    all_df = []

    for f in files:
        # Департамент определяем по имени файла: CT_2023.csv → CT
        dept = f.stem.split("_")[0]
        year = f.stem.split("_")[1]

        df = pd.read_csv(f)

        df["DEPARTMENT"] = dept
        df["YEAR"] = year

        all_df.append(df)

    # Объединяем всё
    df_all = pd.concat(all_df, ignore_index=True)

    # Генерация фильтров
    filters = {
        "departments.json": sorted(df_all["DEPARTMENT"].unique().tolist()),
        "years.json": sorted(df_all["YEAR"].unique().tolist()),
        "categories.json": sorted(df_all["CATEGORY"].dropna().unique().astype(str).tolist()),
        "roads.json": sorted(df_all["ROAD"].dropna().unique().tolist()),
    }

    # Сохраняем в JSON
    for fname, values in filters.items():
        with open(filters_dir / fname, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)

    print("🎉 Фильтры успешно созданы в:", filters_dir)


def load_filtered_dataframe(csv_dir: Path,
                            department=None,
                            year=None,
                            category=None,
                            road=None):

    """
    Формирует dataframe на основе фильтров.
    Любой фильтр может быть None → игнорируется.
    """

    files = list(csv_dir.glob("*.csv"))
    dfs = []

    for f in files:
        dept = f.stem.split("_")[0]
        yr = f.stem.split("_")[1]

        # Фильтруем по имени файла
        if department and dept != department:
            continue
        if year and yr != year:
            continue

        df = pd.read_csv(f)

        df["DEPARTMENT"] = dept
        df["YEAR"] = yr

        # Фильтрация внутри файла
        if category:
            df = df[df["CATEGORY"].astype(str) == str(category)]
        if road:
            df = df[df["ROAD"] == road]

        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        print("⚠️ Нет данных по заданным фильтрам")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
