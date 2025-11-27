from pathlib import Path
import pandas as pd

def convert_excels(input_dir: Path, output_dir: Path):

    """
    Конвертирует Excel (.xlsx) файлы в CSV.
    Переименовывает поля:
        Категория → CATEGORY
        Начало → START_TIME
        Место → ROAD
    Преобразует START_TIME к виду YYYY-MM-DD HH:MM:SS
    Обрезает ROAD до первой запятой
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ищем только .xlsx
    files = list(input_dir.glob("*.xlsx"))

    if not files:
        print(f"❌ Нет файлов .xlsx в {input_dir}")
        return

    for fpath in files:
        print(f"→ Обрабатывается: {fpath.name}")

        # Загружаем Excel (openpyxl по умолчанию)
        df = pd.read_excel(fpath, engine="openpyxl")

        # Переименования полей
        rename_map = {}
        for col in df.columns:
            if "Категор" in col:
                rename_map[col] = "CATEGORY"
            elif "Начало" in col:
                rename_map[col] = "START_TIME"
            elif "Место" in col:
                rename_map[col] = "ROAD"

        df = df.rename(columns=rename_map)

        # Формат времени
        if "START_TIME" in df.columns:
            df["START_TIME"] = (
                pd.to_datetime(df["START_TIME"], errors="coerce")
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

        # Обрезка ROAD до первой запятой
        if "ROAD" in df.columns:
            df["ROAD"] = (
                df["ROAD"].astype(str)
                .str.split(",").str[0]
                .str.strip()
            )

        # Сохранение CSV
        out_path = output_dir / f"{fpath.stem}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"  ✅ Сохранён: {out_path}")

    print("\n🎉 Готово! Все файлы преобразованы в CSV.")