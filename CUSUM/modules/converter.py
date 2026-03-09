import warnings
from pathlib import Path
import pandas as pd
from datetime import datetime, date
import re

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

def _parse_start_time(series: pd.Series) -> pd.Series:
    """Преобразует значения времени из Excel к строковому виду '%Y-%m-%d %H:%M:%S'.

    Поддерживает:
    - datetime/date объекты
    - числовые Excel serial date
    - строки разных форматов (с/без секунд, ISO и т.п.)

    Внутри использует несколько попыток парсинга и безопасный fallback.
    """

    s = series.copy()
    dt = pd.Series(pd.NaT, index=s.index)

    # 0) Уже datetime/date
    mask_py_dt = s.apply(lambda x: isinstance(x, (datetime, date)))
    if mask_py_dt.any():
        dt.loc[mask_py_dt] = pd.to_datetime(s.loc[mask_py_dt], errors="coerce")

    # 1) Excel serial date (числа)
    num = pd.to_numeric(s, errors="coerce")
    mask_num = dt.isna() & num.notna()
    if mask_num.any():
        dt.loc[mask_num] = pd.to_datetime(
            num.loc[mask_num],
            unit="d",
            origin="1899-12-30",
            errors="coerce",
        )

    # 2) Строки: пробуем несколько явных форматов
    mask_str = dt.isna() & s.notna()
    if mask_str.any():
        ss = s.loc[mask_str].astype(str).str.strip()

        formats = [
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d.%m.%Y",
            "%Y-%m-%d",
        ]

        remain = ss.index
        for fmt in formats:
            if len(remain) == 0:
                break
            parsed = pd.to_datetime(ss.loc[remain], format=fmt, errors="coerce")
            ok = parsed.notna()
            if ok.any():
                dt.loc[parsed.index[ok]] = parsed.loc[ok]
                remain = parsed.index[~ok]

        # 3) Последний шанс: dayfirst=True (глушим warning только внутри fallback)
        if len(remain) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(ss.loc[remain], errors="coerce", dayfirst=True)
            dt.loc[parsed.index] = parsed

    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


_FREIGHT_COUNT_RE = re.compile(r"к\s*уч[её]ту\s*(\d+)", re.IGNORECASE)
_FREIGHT_FIRST_COUNT_RE = re.compile(r"\b(\d+)\s*шт\b", re.IGNORECASE)
_FREIGHT_TIME_RE = re.compile(r"(\d+(?:[\.,]\d+)?)\s*ч\b", re.IGNORECASE)


def _split_freight(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Разбивает FREIGHT на два поля:

    - FREIGHT_COUNT: количество поездов
      * если есть '(к учету N)' → берём N
      * иначе берём первое число перед 'шт'

    - FREIGHT_TIMEOUT: время простоя (например '0,67ч')
      * берём последнее значение вида '<число>ч' из строки

    На входе допускаются NaN.
    """

    def _one(val):
        if pd.isna(val):
            return None, None
        text = str(val).strip()
        if not text:
            return None, None

        # COUNT
        m = _FREIGHT_COUNT_RE.search(text)
        if m:
            count = m.group(1)
        else:
            m2 = _FREIGHT_FIRST_COUNT_RE.search(text)
            count = m2.group(1) if m2 else None

        # TIMEOUT (last match)
        times = _FREIGHT_TIME_RE.findall(text)
        if times:
            t = times[-1].replace(".", ",")  # приводим к запятой как в примере
            timeout = f"{t}ч"
        else:
            timeout = None

        return count, timeout

    out = series.apply(_one)
    count_series = out.apply(lambda x: x[0])
    timeout_series = out.apply(lambda x: x[1])
    return count_series, timeout_series

def _split_train_metrics(series: pd.Series, prefix: str, df: pd.DataFrame) -> None:
    """
    Разделяет колонку вида:
    '2шт (к учету 1) 0,67ч'
    '3шт 1,18ч'

    на:
        PREFIX_COUNT
        PREFIX_TIMEOUT
    """

    count_col = f"{prefix}_COUNT"
    time_col = f"{prefix}_TIMEOUT"

    counts = []
    times = []

    for val in series:

        if pd.isna(val):
            counts.append(None)
            times.append(None)
            continue

        text = str(val)

        # количество к учету
        m = re.search(r"к уч[её]ту\s*(\d+)", text, re.IGNORECASE)

        if m:
            count = int(m.group(1))
        else:
            m = re.search(r"(\d+)\s*шт", text)
            count = int(m.group(1)) if m else None

        # время простоя
        m = re.search(r"(\d+[.,]\d+)\s*ч", text)
        timeout = f"{m.group(1)}ч" if m else None

        counts.append(count)
        times.append(timeout)

    df[count_col] = counts
    df[time_col] = times


def convert_xlsx_to_csv_keep_all_fields(input_dir: Path, output_dir: Path) -> None:
    """Конвертирует Excel (.xlsx) файлы в CSV и выполняет преобразование полей.

    Дополнительно:
        - добавляет поле DEPARTMENT по префиксу имени файла
        - добавляет поле YEAR из START_TIME
        - разделяет FREIGHT/PASSENGER/COMMUTER на *_COUNT и *_TIMEOUT
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.xlsx"))
    if not files:
        print(f"❌ Нет файлов .xlsx в {input_dir}")
        return

    for fpath in files:
        print(f"→ Обрабатывается: {fpath.name}")

        department = fpath.stem.split("_")[0]

        df = pd.read_excel(fpath, engine="openpyxl", dtype=object)

        rename_map: dict = {}
        for col in df.columns:
            col_str = str(col).strip()

            if col_str == "#":
                rename_map[col] = "№"
            elif "Категор" in col_str:
                rename_map[col] = "CATEGORY"
            elif "Начало" in col_str:
                rename_map[col] = "START_TIME"
            elif "Место" in col_str:
                rename_map[col] = "ROAD"
            elif "Ответствен" in col_str:
                rename_map[col] = "RESPONSIBILITY"
            elif "Грузов" in col_str:
                rename_map[col] = "FREIGHT"
            elif "Пассажир" in col_str:
                rename_map[col] = "PASSENGER"
            elif "Пригород" in col_str:
                rename_map[col] = "COMMUTER"
            elif "Причин" in col_str:
                rename_map[col] = "REASON"

        df = df.rename(columns=rename_map)

        # DEPARTMENT из имени файла
        df["DEPARTMENT"] = department

        # Формат времени
        if "START_TIME" in df.columns:
            df["START_TIME"] = _parse_start_time(df["START_TIME"])

            # YEAR из START_TIME
            df["YEAR"] = (
                pd.to_datetime(df["START_TIME"], errors="coerce")
                .dt.year
            )

        # ROAD до первой запятой
        if "ROAD" in df.columns:
            df["ROAD"] = df["ROAD"].astype(str).str.split(",").str[0].str.strip()

        # Убираем переносы строк
        for col in ("FREIGHT", "PASSENGER", "COMMUTER"):
            if col in df.columns:
                ser = df[col]
                df[col] = ser.where(
                    ser.isna(),
                    ser.astype(str)
                    .str.replace(r"[\r\n]+", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip(),
                )

        # Разделение показателей поездов
        if "FREIGHT" in df.columns:
            _split_train_metrics(df["FREIGHT"], "FREIGHT", df)

        if "PASSENGER" in df.columns:
            _split_train_metrics(df["PASSENGER"], "PASSENGER", df)

        if "COMMUTER" in df.columns:
            _split_train_metrics(df["COMMUTER"], "COMMUTER", df)

        out_path = output_dir / f"{fpath.stem}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"  ✅ Сохранён: {out_path}")

    print("\n🎉 Готово! Все файлы преобразованы в CSV.")

def compare_csv_structures(
    input_dir: Path,
    pattern: str = "*.csv",
    strict_order: bool = False,
) -> None:
    """
    Сравнивает структуру (набор колонок) всех CSV файлов в директории.

    Args:
        input_dir: директория с CSV
        pattern: шаблон файлов (по умолчанию "*.csv")
        strict_order: если True — сравнивает ещё и порядок колонок,
                      если False — только состав колонок

    Выводит в консоль:
      - эталонный (reference) набор колонок (по объединению всех файлов)
      - для каждого файла: missing/extra, и (опционально) несовпадение порядка
    """

    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"❌ Нет файлов по шаблону {pattern} в {input_dir}")
        return

    # Считываем только заголовки
    cols_by_file: dict[str, list[str]] = {}
    for f in files:
        try:
            df_head = pd.read_csv(f, nrows=0, encoding="utf-8-sig")
        except UnicodeDecodeError:
            # запасной вариант, если вдруг файл не в utf-8-sig
            df_head = pd.read_csv(f, nrows=0, encoding="utf-8")
        cols_by_file[f.name] = list(df_head.columns)

    # Эталон: объединение всех колонок в порядке первого появления
    reference_cols: list[str] = []
    seen = set()
    for fname in cols_by_file:
        for c in cols_by_file[fname]:
            if c not in seen:
                seen.add(c)
                reference_cols.append(c)

    ref_set = set(reference_cols)

    print(f"📌 Найдено файлов: {len(files)}")
    print("📌 Reference (объединение всех колонок):")
    print(reference_cols)

    any_diff = False

    for fname, cols in cols_by_file.items():
        col_set = set(cols)
        missing = [c for c in reference_cols if c not in col_set]
        extra = [c for c in cols if c not in ref_set]

        order_diff = False
        if strict_order:
            # сравниваем только общие колонки в reference-порядке
            cols_in_ref_order = [c for c in cols if c in ref_set]
            ref_in_file_order = [c for c in reference_cols if c in col_set]
            order_diff = cols_in_ref_order != ref_in_file_order

        if missing or extra or (strict_order and order_diff):
            any_diff = True
            print(f"\n❗ Несовпадение: {fname}")
            if missing:
                print(f"  - Missing ({len(missing)}): {missing}")
            if extra:
                print(f"  - Extra   ({len(extra)}): {extra}")
            if strict_order and order_diff:
                print("  - Порядок колонок отличается")

    if not any_diff:
        print("\n✅ Все CSV файлы имеют одинаковую структуру.")

def merge_csv_files(input_dir: Path, output_file: Path) -> None:
    """
    Объединяет все CSV файлы из директории в один CSV.

    Args:
        input_dir: папка с CSV файлами
        output_file: путь к итоговому CSV файлу
    """

    files = sorted(input_dir.glob("*.csv"))

    if not files:
        print(f"❌ Нет CSV файлов в {input_dir}")
        return

    dfs = []

    for f in files:
        print(f"→ Читается: {f.name}")

        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка чтения {f.name}: {e}")

    if not dfs:
        print("❌ Нет данных для объединения")
        return

    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n✅ Объединено файлов: {len(dfs)}")
    print(f"📄 Итоговый файл: {output_file}")
    print(f"📊 Всего строк: {len(merged_df)}")

def remove_nbsp_from_csv(csv_path: Path) -> None:
    """
    Удаляет неразрывные пробелы (NBSP, \u00A0) из CSV файла.

    Заменяет NBSP на обычный пробел и удаляет лишние пробелы
    во всех строковых колонках.
    """

    if not csv_path.exists():
        print(f"❌ Файл не найден: {csv_path}")
        return

    print(f"→ Очистка NBSP: {csv_path.name}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("✅ NBSP успешно удалены")