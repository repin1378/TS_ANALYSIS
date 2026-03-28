from pathlib import Path
import pandas as pd
import json


def create_department_reason_top_json(
    csv_path: Path,
    output_json_path: Path,
    top_n: int = 5,
) -> None:
    """
    Формирует JSON с топом причин сбоев по каждому департаменту.

    Параметры:
        csv_path: путь до общего CSV-файла
        output_json_path: путь до результирующего JSON
        top_n: длина топа причин для каждого департамента
    """

    if top_n <= 0:
        raise ValueError("top_n должен быть больше 0")

    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {"DEPARTMENT", "REASON"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ В CSV отсутствуют обязательные колонки: {sorted(missing_cols)}")
        return

    # Очистка текстовых полей
    for col in ["DEPARTMENT", "REASON"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Убираем пустые и мусорные значения
    df = df[
        df["DEPARTMENT"].notna()
        & df["REASON"].notna()
        & (df["DEPARTMENT"] != "")
        & (df["REASON"] != "")
        & (df["DEPARTMENT"].str.lower() != "nan")
        & (df["REASON"].str.lower() != "nan")
    ].copy()

    if df.empty:
        print("⚠️ После очистки не осталось данных")
        return

    result = {}

    for department in sorted(df["DEPARTMENT"].unique()):
        dep_df = df[df["DEPARTMENT"] == department].copy()

        counts = (
            dep_df["REASON"]
            .value_counts()
            .head(top_n)
        )

        total = len(dep_df)

        top_reasons = []
        for reason, count in counts.items():
            top_reasons.append({
                "reason": reason,
                "count": int(count),
                "share": round(count / total, 4),
            })

        result[department] = top_reasons

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON сохранён: {output_json_path}")

def create_department_road_top_json(
    csv_path: Path,
    output_json_path: Path,
    top_n: int = 5,
) -> None:
    """
    Формирует JSON с топом дорог (ROAD) по каждому департаменту (DEPARTMENT).

    Параметры:
        csv_path: путь до общего CSV-файла
        output_json_path: путь до результирующего JSON
        top_n: длина топа дорог для каждого департамента
    """

    if top_n <= 0:
        raise ValueError("top_n должен быть больше 0")

    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {"DEPARTMENT", "ROAD"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ В CSV отсутствуют обязательные колонки: {sorted(missing_cols)}")
        return

    # Очистка текстовых полей
    for col in ["DEPARTMENT", "ROAD"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Удаление мусора
    df = df[
        df["DEPARTMENT"].notna()
        & df["ROAD"].notna()
        & (df["DEPARTMENT"] != "")
        & (df["ROAD"] != "")
        & (df["DEPARTMENT"].str.lower() != "nan")
        & (df["ROAD"].str.lower() != "nan")
    ].copy()

    if df.empty:
        print("⚠️ После очистки не осталось данных")
        return

    result = {}

    for department in sorted(df["DEPARTMENT"].unique()):
        dep_df = df[df["DEPARTMENT"] == department].copy()

        counts = dep_df["ROAD"].value_counts().head(top_n)

        total = len(dep_df)

        top_roads = []
        for road, count in counts.items():
            top_roads.append({
                "road": road,
                "count": int(count),
                "share": round(count / total, 4),
            })

        result[department] = top_roads

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON сохранён: {output_json_path}")

def create_department_responsibility_top_json(
    csv_path: Path,
    output_json_path: Path,
    top_n: int = 5,
) -> None:
    """
    Формирует JSON с топом RESPONSIBILITY по каждому DEPARTMENT.

    Параметры:
        csv_path: путь до общего CSV-файла
        output_json_path: путь до результирующего JSON
        top_n: длина топа значений RESPONSIBILITY
    """

    if top_n <= 0:
        raise ValueError("top_n должен быть больше 0")

    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {"DEPARTMENT", "RESPONSIBILITY"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ В CSV отсутствуют колонки: {sorted(missing_cols)}")
        return

    # Очистка текста
    for col in ["DEPARTMENT", "RESPONSIBILITY"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Удаление мусорных значений
    df = df[
        df["DEPARTMENT"].notna()
        & df["RESPONSIBILITY"].notna()
        & (df["DEPARTMENT"] != "")
        & (df["RESPONSIBILITY"] != "")
        & (df["DEPARTMENT"].str.lower() != "nan")
        & (df["RESPONSIBILITY"].str.lower() != "nan")
    ].copy()

    if df.empty:
        print("⚠️ После очистки не осталось данных")
        return

    result = {}

    for department in sorted(df["DEPARTMENT"].unique()):
        dep_df = df[df["DEPARTMENT"] == department]

        counts = dep_df["RESPONSIBILITY"].value_counts().head(top_n)
        total = len(dep_df)

        top_items = []
        for resp, count in counts.items():
            top_items.append({
                "responsibility": resp,
                "count": int(count),
                "share": round(count / total, 4),
            })

        result[department] = top_items

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON сохранён: {output_json_path}")