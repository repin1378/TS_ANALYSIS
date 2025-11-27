import pandas as pd
from pathlib import Path


# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: сохранить DF с автоимёнем
# ============================================================
def _save_result_df(df: pd.DataFrame, out_dir: Path, prefix: str, **params):
    """
    Генерирует имя файла на основе параметров фильтрации и сохраняет CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Формируем имя файла
    parts = [prefix]
    for key, value in params.items():
        if value is not None:
            parts.append(f"{key}-{value}")

    filename = "_".join(parts) + ".csv"
    out_path = out_dir / filename

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"💾 Сохранён файл: {out_path}")

    return out_path


# ============================================================
# 1) Получить DF — DEPT + YEAR + ROAD + CATEGORY
# ============================================================
def get_df_full_filter(
    csv_dir: Path,
    department: str = None,
    year: str = None,
    road: str = None,
    category: str = None,
    save_dir: Path = None
):
    """
    Формирует DataFrame для одного года и сохраняет CSV.
    """
    files = list(csv_dir.glob("*.csv"))
    dfs = []

    for f in files:
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue

        dept, yr = parts[0], parts[1]

        if department and dept != department:
            continue
        if year and yr != year:
            continue

        df = pd.read_csv(f)
        df["DEPARTMENT"] = dept
        df["YEAR"] = yr

        if road is not None:
            df = df[df["ROAD"] == road]

        if category is not None:
            df = df[df["CATEGORY"].astype(str) == str(category)]

        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        print("⚠️ Нет данных по заданным фильтрам")
        return pd.DataFrame()

    df_result = pd.concat(dfs, ignore_index=True)

    # ----------- Сохранение в CSV -----------
    if save_dir:
        _save_result_df(
            df_result,
            save_dir,
            prefix="filtered",
            department=department,
            year=year,
            road=road,
            category=category
        )

    return df_result


# ============================================================
# 2) Получить DF — DEPT + ROAD + CATEGORY (все годы)
# ============================================================
def get_df_multi_year(
    csv_dir: Path,
    department: str = None,
    road: str = None,
    category: str = None,
    save_dir: Path = None
):
    """
    Формирует DataFrame, объединяя все годы.
    Сохраняет CSV-файл.
    """
    files = list(csv_dir.glob("*.csv"))
    dfs = []

    for f in files:
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue

        dept, yr = parts[0], parts[1]

        if department and dept != department:
            continue

        df = pd.read_csv(f)
        df["DEPARTMENT"] = dept
        df["YEAR"] = yr

        if road is not None:
            df = df[df["ROAD"] == road]

        if category is not None:
            df = df[df["CATEGORY"].astype(str) == str(category)]

        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        print("⚠️ Нет данных по заданным фильтрам")
        return pd.DataFrame()

    df_result = pd.concat(dfs, ignore_index=True)

    # ----------- Сохранение в CSV -----------
    if save_dir:
        _save_result_df(
            df_result,
            save_dir,
            prefix="filtered_multi_year",
            department=department,
            road=road,
            category=category
        )

    return df_result