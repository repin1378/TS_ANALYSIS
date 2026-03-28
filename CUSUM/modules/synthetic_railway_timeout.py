from pathlib import Path
import json
import pandas as pd


def _clean_timeout_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Очищает текстовые поля с простоями:
    - заменяет NBSP на обычный пробел
    - удаляет лишние пробелы
    """
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\u00A0", " ", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    return df


def _prepare_numeric_count(series: pd.Series) -> pd.Series:
    """
    Приводит COUNT-колонку к числовому виду и отбрасывает невалидные значения.
    """
    values = pd.to_numeric(series, errors="coerce")
    return values[values.notna()]


def _prepare_timeout_text(series: pd.Series) -> pd.Series:
    """
    Подготавливает TIMEOUT-колонку:
    - убирает пустые/мусорные значения
    - сохраняет строковое представление (например, '0,67ч')
    """
    s = (
        series.astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    mask = (
        s.notna()
        & (s != "")
        & (s.str.lower() != "nan")
        & (s.str.lower() != "none")
    )
    return s[mask]


def _build_timeout_reference(
    df: pd.DataFrame,
    count_col: str,
    timeout_col: str,
    count_key: str,
    timeout_key: str,
) -> dict:
    """
    Формирует справочник значений COUNT/TIMEOUT для одного типа поездов.
    """
    result = {
        count_key: [],
        timeout_key: [],
    }

    if count_col in df.columns:
        counts = _prepare_numeric_count(df[count_col])
        result[count_key] = sorted(counts.astype(int).unique().tolist())

    if timeout_col in df.columns:
        timeouts = _prepare_timeout_text(df[timeout_col])
        result[timeout_key] = sorted(timeouts.unique().tolist())

    return result


def create_railway_timeout_reference_json(
    csv_path: Path,
    output_json_path: Path,
) -> None:
    """
    Формирует единый JSON-справочник простоев для генерации синтетического временного ряда.

    Входные поля CSV:
        FREIGHT_COUNT, FREIGHT_TIMEOUT,
        PASSENGER_COUNT, PASSENGER_TIMEOUT,
        COMMUTER_COUNT, COMMUTER_TIMEOUT

    Выходной JSON:
    {
      "freight_count": [...],
      "freight_timeout": [...],
      "passenger_count": [...],
      "passenger_timeout": [...],
      "commuter_count": [...],
      "commuter_timeout": [...]
    }
    """

    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {
        "FREIGHT_COUNT", "FREIGHT_TIMEOUT",
        "PASSENGER_COUNT", "PASSENGER_TIMEOUT",
        "COMMUTER_COUNT", "COMMUTER_TIMEOUT",
    }

    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        print(f"⚠️ В CSV отсутствуют колонки: {missing_cols}")

    df = _clean_timeout_columns(
        df,
        [
            "FREIGHT_TIMEOUT",
            "PASSENGER_TIMEOUT",
            "COMMUTER_TIMEOUT",
        ],
    )

    result = {}
    result.update(
        _build_timeout_reference(
            df,
            count_col="FREIGHT_COUNT",
            timeout_col="FREIGHT_TIMEOUT",
            count_key="freight_count",
            timeout_key="freight_timeout",
        )
    )
    result.update(
        _build_timeout_reference(
            df,
            count_col="PASSENGER_COUNT",
            timeout_col="PASSENGER_TIMEOUT",
            count_key="passenger_count",
            timeout_key="passenger_timeout",
        )
    )
    result.update(
        _build_timeout_reference(
            df,
            count_col="COMMUTER_COUNT",
            timeout_col="COMMUTER_TIMEOUT",
            count_key="commuter_count",
            timeout_key="commuter_timeout",
        )
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON сохранён: {output_json_path}")


def create_railway_timeout_reference_by_field_json(
    csv_path: Path,
    output_json_dir: Path,
) -> None:
    """
    Формирует отдельные JSON-справочники по каждому полю:
        freight_count.json
        freight_timeout.json
        passenger_count.json
        passenger_timeout.json
        commuter_count.json
        commuter_timeout.json
    """

    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df = _clean_timeout_columns(
        df,
        [
            "FREIGHT_TIMEOUT",
            "PASSENGER_TIMEOUT",
            "COMMUTER_TIMEOUT",
        ],
    )

    output_json_dir.mkdir(parents=True, exist_ok=True)

    field_map = {
        "freight_count.json": (
            "FREIGHT_COUNT",
            lambda s: sorted(_prepare_numeric_count(s).astype(int).unique().tolist()),
        ),
        "freight_timeout.json": (
            "FREIGHT_TIMEOUT",
            lambda s: sorted(_prepare_timeout_text(s).unique().tolist()),
        ),
        "passenger_count.json": (
            "PASSENGER_COUNT",
            lambda s: sorted(_prepare_numeric_count(s).astype(int).unique().tolist()),
        ),
        "passenger_timeout.json": (
            "PASSENGER_TIMEOUT",
            lambda s: sorted(_prepare_timeout_text(s).unique().tolist()),
        ),
        "commuter_count.json": (
            "COMMUTER_COUNT",
            lambda s: sorted(_prepare_numeric_count(s).astype(int).unique().tolist()),
        ),
        "commuter_timeout.json": (
            "COMMUTER_TIMEOUT",
            lambda s: sorted(_prepare_timeout_text(s).unique().tolist()),
        ),
    }

    for file_name, (col_name, extractor) in field_map.items():
        values = []
        if col_name in df.columns:
            values = extractor(df[col_name])

        out_path = output_json_dir / file_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)

        print(f"✅ JSON сохранён: {out_path}")
