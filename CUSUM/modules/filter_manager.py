import pandas as pd
from pathlib import Path
import json

def create_filters(csv_file: Path, filters_dir: Path):
    """
    Создаёт JSON-файлы фильтров на основе объединённого CSV.

    Создаёт:
        departments.json
        years.json
        categories.json
        roads.json
        reasons.json
    """

    filters_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        print(f"❌ CSV файл не найден: {csv_file}")
        return

    print(f"→ Чтение данных: {csv_file}")

    df = pd.read_csv(csv_file, encoding="utf-8-sig")

    filters = {
        "departments.json": sorted(df["DEPARTMENT"].dropna().unique().tolist()),
        "years.json": sorted(df["YEAR"].dropna().unique().tolist()),
        "categories.json": sorted(df["CATEGORY"].dropna().astype(str).unique().tolist()),
        "roads.json": sorted(df["ROAD"].dropna().unique().tolist()),
        "reasons.json": sorted(df["REASON"].dropna().astype(str).unique().tolist()),
    }

    for fname, values in filters.items():

        path = filters_dir / fname

        with open(path, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)

        print(f"  ✔ создан {fname} ({len(values)} значений)")

    print("\n🎉 JSON фильтры успешно созданы")


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
