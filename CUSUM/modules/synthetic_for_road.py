from pathlib import Path
import json
import pandas as pd


def _clean_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Очищает текстовые колонки от NBSP, лишних пробелов и пустых значений.
    """
    for col in columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    valid_mask = pd.Series(True, index=df.index)
    for col in columns:
        valid_mask &= (
            df[col].notna()
            & (df[col] != "")
            & (df[col].str.lower() != "nan")
            & (df[col].str.lower() != "none")
        )

    return df[valid_mask].copy()


def _create_top_json(
    csv_path: Path,
    output_json_path: Path,
    group_col: str,
    value_col: str,
    value_key: str,
    top_n: int = 5,
) -> None:
    """
    Универсальная функция формирования top-N JSON по зависимости:
        group_col -> value_col
    """

    if top_n <= 0:
        raise ValueError("top_n должен быть больше 0")

    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {group_col, value_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ В CSV отсутствуют обязательные колонки: {sorted(missing_cols)}")
        return

    df = _clean_text_columns(df, [group_col, value_col])

    if df.empty:
        print("⚠️ После очистки не осталось данных")
        return

    result = {}

    for group_value in sorted(df[group_col].unique()):
        part_df = df[df[group_col] == group_value].copy()

        counts = part_df[value_col].value_counts().head(top_n)
        total = len(part_df)

        top_items = []
        for item_value, count in counts.items():
            top_items.append({
                value_key: item_value,
                "count": int(count),
                "share": round(count / total, 4),
            })

        result[group_value] = top_items

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON сохранён: {output_json_path}")


def create_road_reason_top_json(
    csv_path: Path,
    output_json_path: Path,
    top_n: int = 5,
) -> None:
    """
    Формирует JSON с топом причин сбоев (REASON) по каждой дороге (ROAD).
    """
    _create_top_json(
        csv_path=csv_path,
        output_json_path=output_json_path,
        group_col="ROAD",
        value_col="REASON",
        value_key="reason",
        top_n=top_n,
    )


def create_road_department_top_json(
    csv_path: Path,
    output_json_path: Path,
    top_n: int = 5,
) -> None:
    """
    Формирует JSON с топом департаментов (DEPARTMENT) по каждой дороге (ROAD).
    """
    _create_top_json(
        csv_path=csv_path,
        output_json_path=output_json_path,
        group_col="ROAD",
        value_col="DEPARTMENT",
        value_key="department",
        top_n=top_n,
    )


def create_road_responsibility_top_json(
    csv_path: Path,
    output_json_path: Path,
    top_n: int = 5,
) -> None:
    """
    Формирует JSON с топом зон ответственности (RESPONSIBILITY) по каждой дороге (ROAD).
    """
    _create_top_json(
        csv_path=csv_path,
        output_json_path=output_json_path,
        group_col="ROAD",
        value_col="RESPONSIBILITY",
        value_key="responsibility",
        top_n=top_n,
    )
