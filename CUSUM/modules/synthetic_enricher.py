from pathlib import Path
import json
import numpy as np
import pandas as pd


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _weighted_choice(items: list[dict], value_key: str, rng: np.random.Generator) -> str:
    """
    Выбор значения из top-json с учётом поля share.
    """
    if not items:
        return ""

    values = [item[value_key] for item in items]
    weights = np.array([float(item.get("share", 0)) for item in items], dtype=float)

    if weights.sum() <= 0:
        weights = np.ones(len(values), dtype=float) / len(values)
    else:
        weights = weights / weights.sum()

    return str(rng.choice(values, p=weights))


def _uniform_choice(values: list, rng: np.random.Generator):
    if not values:
        return ""
    return rng.choice(values)


def _prepare_text_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Гарантирует, что колонка существует и имеет object-тип,
    пригодный для записи строк и пустых значений.
    """
    if col not in df.columns:
        df[col] = ""
    else:
        df[col] = df[col].astype("object")
        df[col] = df[col].where(df[col].notna(), "")
    return df


def _prepare_object_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        df = _prepare_text_column(df, col)
    return df


def _fill_timeout_fields(
    row: pd.Series,
    timeout_ref: dict,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Заполняет поля простоев.

    Правило:
        если CATEGORY == "3", поля простоев не заполняются
        если CATEGORY == "1" или "2", поля заполняются
    """
    category = str(row.get("CATEGORY", "")).strip()

    timeout_cols = [
        "FREIGHT_COUNT", "FREIGHT_TIMEOUT",
        "PASSENGER_COUNT", "PASSENGER_TIMEOUT",
        "COMMUTER_COUNT", "COMMUTER_TIMEOUT",
    ]

    if category == "3":
        for col in timeout_cols:
            row[col] = ""
        return row

    if category not in {"1", "2"}:
        for col in timeout_cols:
            row[col] = ""
        return row

    def _fill_pair(count_key_json, timeout_key_json, count_col_df, timeout_col_df):
        counts = timeout_ref.get(count_key_json, [])
        timeouts = timeout_ref.get(timeout_key_json, [])

        count_value = _uniform_choice(counts, rng)
        if count_value == "":
            row[count_col_df] = ""
            row[timeout_col_df] = ""
            return

        try:
            count_value_int = int(count_value)
        except Exception:
            count_value_int = count_value

        row[count_col_df] = count_value_int

        if isinstance(count_value_int, int) and count_value_int == 0:
            row[timeout_col_df] = ""
        else:
            row[timeout_col_df] = _uniform_choice(timeouts, rng)

    _fill_pair("freight_count", "freight_timeout", "FREIGHT_COUNT", "FREIGHT_TIMEOUT")
    _fill_pair("passenger_count", "passenger_timeout", "PASSENGER_COUNT", "PASSENGER_TIMEOUT")
    _fill_pair("commuter_count", "commuter_timeout", "COMMUTER_COUNT", "COMMUTER_TIMEOUT")

    return row


def _postprocess_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит count-поля к nullable integer Int64.
    Пустые значения остаются <NA>.
    """
    count_cols = ["FREIGHT_COUNT", "PASSENGER_COUNT", "COMMUTER_COUNT"]
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def enrich_synthetic_road_csvs(
    input_dir: Path,
    output_dir: Path,
    road_department_top_json: Path,
    road_reason_top_json: Path,
    road_responsibility_top_json: Path,
    categories_json: Path,
    railway_timeout_reference_json: Path,
    random_seed: int | None = None,
) -> None:
    """
    Обогащает synthetic CSV по дорогам.

    Заполняет:
        CATEGORY
        DEPARTMENT
        REASON
        RESPONSIBILITY
        FREIGHT_COUNT, FREIGHT_TIMEOUT
        PASSENGER_COUNT, PASSENGER_TIMEOUT
        COMMUTER_COUNT, COMMUTER_TIMEOUT

    ROAD берётся из самого synthetic CSV.
    """
    rng = np.random.default_rng(random_seed)

    road_to_department = _load_json(road_department_top_json)
    road_to_reason = _load_json(road_reason_top_json)
    road_to_responsibility = _load_json(road_responsibility_top_json)
    categories = [str(x) for x in _load_json(categories_json)]
    timeout_ref = _load_json(railway_timeout_reference_json)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"⚠️ Нет CSV-файлов в папке: {input_dir}")
        return

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")

            df = _prepare_object_columns(df, [
                "CATEGORY",
                "DEPARTMENT",
                "REASON",
                "RESPONSIBILITY",
                "FREIGHT_COUNT",
                "FREIGHT_TIMEOUT",
                "PASSENGER_COUNT",
                "PASSENGER_TIMEOUT",
                "COMMUTER_COUNT",
                "COMMUTER_TIMEOUT",
            ])

            if "ROAD" not in df.columns:
                print(f"⚠️ Пропуск {csv_file.name}: нет ROAD")
                continue

            road_name = str(df["ROAD"].iloc[0]).strip()
            if not road_name:
                print(f"⚠️ Пропуск {csv_file.name}: пустой ROAD")
                continue

            dep_items = road_to_department.get(road_name, [])
            reason_items = road_to_reason.get(road_name, [])
            resp_items = road_to_responsibility.get(road_name, [])

            for idx in df.index:
                df.at[idx, "CATEGORY"] = str(_uniform_choice(categories, rng))
                df.at[idx, "DEPARTMENT"] = _weighted_choice(dep_items, "department", rng)
                df.at[idx, "REASON"] = _weighted_choice(reason_items, "reason", rng)
                df.at[idx, "RESPONSIBILITY"] = _weighted_choice(resp_items, "responsibility", rng)

                row = df.loc[idx].copy()
                row = _fill_timeout_fields(row, timeout_ref, rng)

                for col in [
                    "FREIGHT_COUNT", "FREIGHT_TIMEOUT",
                    "PASSENGER_COUNT", "PASSENGER_TIMEOUT",
                    "COMMUTER_COUNT", "COMMUTER_TIMEOUT",
                ]:
                    df.at[idx, col] = row[col]

            df = _postprocess_count_columns(df)

            out_path = output_dir / csv_file.name
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✅ Обогащён road synthetic CSV: {out_path}")

        except Exception as e:
            print(f"❌ Ошибка обработки {csv_file.name}: {e}")


def enrich_synthetic_department_csvs(
    input_dir: Path,
    output_dir: Path,
    department_road_top_json: Path,
    department_reason_top_json: Path,
    department_responsibility_top_json: Path,
    categories_json: Path,
    railway_timeout_reference_json: Path,
    random_seed: int | None = None,
) -> None:
    """
    Обогащает synthetic CSV по департаментам.

    Заполняет:
        CATEGORY
        ROAD
        REASON
        RESPONSIBILITY
        FREIGHT_COUNT, FREIGHT_TIMEOUT
        PASSENGER_COUNT, PASSENGER_TIMEOUT
        COMMUTER_COUNT, COMMUTER_TIMEOUT

    DEPARTMENT берётся из самого synthetic CSV.
    """
    rng = np.random.default_rng(random_seed)

    department_to_road = _load_json(department_road_top_json)
    department_to_reason = _load_json(department_reason_top_json)
    department_to_responsibility = _load_json(department_responsibility_top_json)
    categories = [str(x) for x in _load_json(categories_json)]
    timeout_ref = _load_json(railway_timeout_reference_json)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"⚠️ Нет CSV-файлов в папке: {input_dir}")
        return

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")

            df = _prepare_object_columns(df, [
                "CATEGORY",
                "ROAD",
                "REASON",
                "RESPONSIBILITY",
                "FREIGHT_COUNT",
                "FREIGHT_TIMEOUT",
                "PASSENGER_COUNT",
                "PASSENGER_TIMEOUT",
                "COMMUTER_COUNT",
                "COMMUTER_TIMEOUT",
            ])

            if "DEPARTMENT" not in df.columns:
                print(f"⚠️ Пропуск {csv_file.name}: нет DEPARTMENT")
                continue

            department_name = str(df["DEPARTMENT"].iloc[0]).strip()
            if not department_name:
                print(f"⚠️ Пропуск {csv_file.name}: пустой DEPARTMENT")
                continue

            road_items = department_to_road.get(department_name, [])
            reason_items = department_to_reason.get(department_name, [])
            resp_items = department_to_responsibility.get(department_name, [])

            for idx in df.index:
                df.at[idx, "CATEGORY"] = str(_uniform_choice(categories, rng))
                df.at[idx, "ROAD"] = _weighted_choice(road_items, "road", rng)
                df.at[idx, "REASON"] = _weighted_choice(reason_items, "reason", rng)
                df.at[idx, "RESPONSIBILITY"] = _weighted_choice(resp_items, "responsibility", rng)

                row = df.loc[idx].copy()
                row = _fill_timeout_fields(row, timeout_ref, rng)

                for col in [
                    "FREIGHT_COUNT", "FREIGHT_TIMEOUT",
                    "PASSENGER_COUNT", "PASSENGER_TIMEOUT",
                    "COMMUTER_COUNT", "COMMUTER_TIMEOUT",
                ]:
                    df.at[idx, col] = row[col]

            df = _postprocess_count_columns(df)

            out_path = output_dir / csv_file.name
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✅ Обогащён department synthetic CSV: {out_path}")

        except Exception as e:
            print(f"❌ Ошибка обработки {csv_file.name}: {e}")