import pandas as pd
from pathlib import Path
import json


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
    all_csv_path: Path,
    departments_json_path: Path,
    roads_json_path: Path,
    years_json_path: Path,
    output_dir: Path,
) -> None:
    """
    На основе общего файла all_events.csv и JSON-фильтров формирует отдельные CSV
    со списком событий по комбинациям:

        1. DEPARTMENT + YEAR
        2. ROAD + YEAR

    Параметры:
        all_csv_path: путь до общего файла all_events.csv
        departments_json_path: путь до departments.json
        roads_json_path: путь до roads.json
        years_json_path: путь до years.json
        output_dir: путь до папки для результирующих CSV

    Результат:
        output_dir/
            by_department_year/
                CSH_2023.csv
                CSH_2024.csv
                CT_2023.csv
                ...
            by_road_year/
                Октябрьская_жд_2023.csv
                Октябрьская_жд_2024.csv
                ...
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    by_department_dir = output_dir / "by_department_year"
    by_road_dir = output_dir / "by_road_year"

    by_department_dir.mkdir(parents=True, exist_ok=True)
    by_road_dir.mkdir(parents=True, exist_ok=True)

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

    required_columns = {"DEPARTMENT", "ROAD"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        print(f"❌ В CSV отсутствуют обязательные колонки: {sorted(missing_cols)}")
        return

    # Нормализация текстовых полей
    for col in ("DEPARTMENT", "ROAD"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\u00A0", " ", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # YEAR: используем готовую колонку, либо извлекаем из START_TIME
    if "YEAR" in df.columns:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    elif "START_TIME" in df.columns:
        dt = pd.to_datetime(df["START_TIME"], errors="coerce")
        df["YEAR"] = dt.dt.year.astype("Int64")
    else:
        print("❌ В CSV отсутствует колонка YEAR и нет START_TIME для её вычисления")
        return

    # Нормализуем years из json к int
    years = [int(y) for y in years]

    # Берём только строки, попадающие в фильтры
    df = df[
        df["DEPARTMENT"].isin(departments)
        & df["ROAD"].isin(roads)
        & df["YEAR"].isin(years)
    ].copy()

    if df.empty:
        print("⚠️ Нет данных после применения фильтров JSON")
        return

    def _safe_name(value: str) -> str:
        """Безопасное имя файла."""
        return (
            str(value)
            .replace("\u00A0", " ")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
            .replace(",", "")
            .replace(".", "")
            .strip()
        )

    # -------- CSV по комбинациям DEPARTMENT + YEAR --------
    dep_count = 0
    for department in departments:
        for year in years:
            df_part = df[
                (df["DEPARTMENT"] == department)
                & (df["YEAR"] == year)
            ].copy()

            if df_part.empty:
                continue

            out_path = by_department_dir / f"{_safe_name(department)}_{year}.csv"
            df_part.to_csv(out_path, index=False, encoding="utf-8-sig")
            dep_count += 1
            print(f"  ✅ Сохранён: {out_path}")

    # -------- CSV по комбинациям ROAD + YEAR --------
    road_count = 0
    for road in roads:
        for year in years:
            df_part = df[
                (df["ROAD"] == road)
                & (df["YEAR"] == year)
            ].copy()

            if df_part.empty:
                continue

            out_path = by_road_dir / f"{_safe_name(road)}_{year}.csv"
            df_part.to_csv(out_path, index=False, encoding="utf-8-sig")
            road_count += 1
            print(f"  ✅ Сохранён: {out_path}")

    print("\n🎉 Формирование CSV завершено.")
    print(f"📁 Файлов по DEPARTMENT + YEAR: {dep_count}")
    print(f"📁 Файлов по ROAD + YEAR: {road_count}")


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