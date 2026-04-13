import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# 1) Обработка одного файла + сохранение обновлённого CSV
# ============================================================
def preprocess_dataframe(source_dirs: list[Path], save_dirs: list[Path]) -> None:
    """
    Обрабатывает CSV-файлы из списка папок и сохраняет результаты
    в соответствующие папки.

    Добавляемые поля:
        DELTA_TIME
        DELTA_MINUTES
        TIME_DIFF
        EVENT_NUMBER
        INDEX

    Параметры:
        source_dirs : список папок с исходными CSV
        save_dirs   : список папок для сохранения обработанных CSV
    """

    if len(source_dirs) != len(save_dirs):
        raise ValueError("source_dirs и save_dirs должны быть одинаковой длины")

    for source_dir, save_dir in zip(source_dirs, save_dirs):

        source_dir = Path(source_dir)
        save_dir = Path(save_dir)

        if not source_dir.exists():
            print(f"⚠️ Папка не найдена: {source_dir}")
            continue

        save_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(source_dir.glob("*.csv"))

        if not csv_files:
            print(f"⚠️ Нет CSV файлов в папке: {source_dir}")
            continue

        print(f"\n📂 Обработка папки: {source_dir}")

        for csv_file in csv_files:

            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")

                if "START_TIME" not in df.columns:
                    print(f"⚠️ В файле нет START_TIME: {csv_file.name}")
                    continue

                # преобразуем время
                df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")

                # удаляем строки без времени
                df = df[df["START_TIME"].notna()].copy()

                if df.empty:
                    print(f"⚠️ Пустой файл после очистки дат: {csv_file.name}")
                    continue

                # сортировка по времени
                df = df.sort_values("START_TIME").reset_index(drop=True)

                first_time = df["START_TIME"].iloc[0]

                # время от первого события
                df["DELTA_TIME"] = df["START_TIME"] - first_time

                # минуты от первого события
                df["DELTA_MINUTES"] = df["DELTA_TIME"].dt.total_seconds() / 60

                # разница между событиями
                df["TIME_DIFF"] = df["DELTA_MINUTES"].diff().fillna(0)

                # номер события
                df["EVENT_NUMBER"] = range(1, len(df) + 1)

                # нормированный индекс
                if len(df) > 1:
                    df["INDEX"] = df.index / (len(df) - 1)
                else:
                    df["INDEX"] = 1.0

                out_path = save_dir / csv_file.name
                df.to_csv(out_path, index=False, encoding="utf-8-sig")

                print(f"✅ Сохранён: {out_path}")

            except Exception as e:
                print(f"❌ Ошибка обработки {csv_file.name}: {e}")


# ============================================================
# 2) Гистограмма TIME_DIFF
# ============================================================
def save_histogram(source_dirs: list[Path], graph_dirs: list[Path]) -> None:
    """
    Строит гистограммы TIME_DIFF для всех CSV-файлов из списка папок
    и сохраняет их в соответствующие папки.

    Параметры:
        source_dirs : список папок с исходными CSV
        graph_dirs  : список папок для сохранения PDF-гистограмм

    Важно:
        source_dirs и graph_dirs должны быть одинаковой длины.
    """

    if len(source_dirs) != len(graph_dirs):
        raise ValueError("source_dirs и graph_dirs должны быть одинаковой длины")

    for source_dir, graph_dir in zip(source_dirs, graph_dirs):
        source_dir = Path(source_dir)
        graph_dir = Path(graph_dir)

        if not source_dir.exists():
            print(f"⚠️ Папка не найдена: {source_dir}")
            continue

        graph_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(source_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️ Нет CSV файлов в папке: {source_dir}")
            continue

        print(f"\n📂 Построение гистограмм для папки: {source_dir}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")

                if "TIME_DIFF" not in df.columns:
                    print(f"⚠️ В файле нет TIME_DIFF: {csv_file.name}")
                    continue

                time_diff = pd.to_numeric(df["TIME_DIFF"], errors="coerce").dropna()

                if time_diff.empty:
                    print(f"⚠️ Нет валидных значений TIME_DIFF: {csv_file.name}")
                    continue

                file_name = csv_file.stem
                out_path = graph_dir / f"{file_name}.pdf"

                n = len(time_diff)
                tmax = time_diff.max()

                # 1) Автоматический выбор DPI
                if n < 500:
                    dpi = 150
                elif n < 5000:
                    dpi = 200
                elif n < 50000:
                    dpi = 300
                else:
                    dpi = 400

                # 2) Автоматический выбор hist_step
                if n < 500:
                    hist_step = max(2, tmax / 20)
                elif n < 5000:
                    hist_step = max(1, tmax / 40)
                elif n < 50000:
                    hist_step = max(0.5, tmax / 60)
                else:
                    hist_step = max(0.25, tmax / 80)

                # округляем шаг до "красивого" числа
                if hist_step > 10:
                    hist_step = round(hist_step, -1)
                elif hist_step > 1:
                    hist_step = round(hist_step, 1)
                else:
                    hist_step = round(hist_step, 2)

                # защита от нулевого шага / пустого диапазона
                if hist_step <= 0:
                    hist_step = 1

                xmin = 0
                xmax = max(hist_step, ((tmax // hist_step) + 1) * hist_step)
                bin_edges = np.arange(xmin, xmax + hist_step, hist_step)

                print(
                    f"📌 {csv_file.name}: "
                    f"n={n}, max={tmax:.2f}, hist_step={hist_step}, dpi={dpi}"
                )

                plt.figure(figsize=(10, 5))
                plt.hist(
                    time_diff,
                    bins=bin_edges,
                    edgecolor="black",
                    alpha=0.7,
                )

                plt.xlabel("Интервалы между событиями (мин)")
                plt.ylabel("Частота")
                plt.title(f"Гистограмма TIME_DIFF — {file_name}")
                plt.grid(axis="y", linestyle="--", alpha=0.6)
                plt.xlim(xmin, xmax)

                plt.tight_layout()
                plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
                plt.close()

                print(f"📊 Гистограмма сохранена: {out_path}")

            except Exception as e:
                print(f"❌ Ошибка построения гистограммы для {csv_file.name}: {e}")

# ============================================================
# 3) График НЧС
# ============================================================
def plot_cumulative_events(source_dirs: list[Path], graph_dirs: list[Path]) -> None:
    """
    Строит графики накопленного числа событий для всех CSV-файлов
    из списка папок и сохраняет их в соответствующие папки.

    Для каждого файла строится график:
      - INDEX по START_TIME
      - квартальные линии
      - сезонные линии
      - горизонтальная линия y=1
      - обрезка графика по последнему событию

    Параметры:
        source_dirs : список папок с исходными CSV
        graph_dirs  : список папок для сохранения PDF-графиков

    Важно:
        source_dirs и graph_dirs должны быть одинаковой длины.
    """

    if len(source_dirs) != len(graph_dirs):
        raise ValueError("source_dirs и graph_dirs должны быть одинаковой длины")

    for source_dir, graph_dir in zip(source_dirs, graph_dirs):
        source_dir = Path(source_dir)
        graph_dir = Path(graph_dir)

        if not source_dir.exists():
            print(f"⚠️ Папка не найдена: {source_dir}")
            continue

        graph_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(source_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️ Нет CSV файлов в папке: {source_dir}")
            continue

        print(f"\n📂 Построение cumulative-графиков для папки: {source_dir}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")

                required_cols = {"START_TIME", "INDEX"}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    print(f"⚠️ В файле {csv_file.name} нет колонок: {sorted(missing_cols)}")
                    continue

                df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
                df["INDEX"] = pd.to_numeric(df["INDEX"], errors="coerce")

                df = df[df["START_TIME"].notna() & df["INDEX"].notna()].copy()

                if df.empty:
                    print(f"⚠️ Нет валидных данных для графика: {csv_file.name}")
                    continue

                df = df.sort_values("START_TIME").reset_index(drop=True)

                file_name = csv_file.stem
                out_path = graph_dir / f"{file_name}_cumulative.pdf"

                plt.figure(figsize=(12, 6))

                # 1. График INDEX
                plt.plot(
                    df["START_TIME"],
                    df["INDEX"],
                    linewidth=2,
                    color="black",
                    label="Накопленное число событий"
                )

                # Диапазон времени
                start = df["START_TIME"].min().normalize()
                end = df["START_TIME"].max().normalize()

                # 2. Квартальные линии
                quarter_starts = pd.date_range(start=start, end=end, freq="QS")

                for i, q in enumerate(quarter_starts):
                    if start <= q <= end:
                        plt.axvline(
                            q,
                            linestyle="--",
                            color="gray",
                            linewidth=1.2,
                            alpha=0.7,
                            label="Квартальная граница" if i == 0 else None
                        )

                # 3. Сезонные линии
                season_offsets = [(3, 1), (6, 1), (9, 1), (12, 1)]
                years = range(start.year, end.year + 1)
                season_lines = []

                for year in years:
                    for month, day in season_offsets:
                        season_date = pd.Timestamp(year, month, day)
                        if start <= season_date <= end:
                            season_lines.append(season_date)

                for i, d in enumerate(season_lines):
                    plt.axvline(
                        d,
                        linestyle=":",
                        color="tab:blue",
                        linewidth=1.4,
                        alpha=0.8,
                        label="Сезон" if i == 0 else None
                    )

                # 4. Горизонтальная линия y = 1
                plt.axhline(1, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

                # 5. Границы графика
                first_time = df["START_TIME"].min()
                last_time = df["START_TIME"].max()

                plt.xlim(first_time, last_time)
                plt.ylim(0, 1)

                # 6. Подписи и стиль
                plt.title("График накопленного числа событий", fontsize=14)
                plt.xlabel("Время", fontsize=12)
                plt.ylabel("Нормированный индекс событий", fontsize=12)
                plt.grid(alpha=0.4)

                # 7. Убираем дубликаты в легенде
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc="upper left")

                # 8. Сохранение
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"📈 График сохранён: {out_path}")

            except Exception as e:
                print(f"❌ Ошибка построения графика для {csv_file.name}: {e}")

def plot_cumulative_events_with_spike(
    source_dirs: list[Path],
    graph_dirs: list[Path],
) -> None:
    """
    Строит графики нормированного накопленного числа событий
    с выделением периодов всплесков (SPIKE_FLAG) для всех CSV-файлов
    из списка папок.

    Параметры:
        source_dirs : список папок с исходными CSV
        graph_dirs  : список папок для сохранения PDF-графиков

    Важно:
        source_dirs и graph_dirs должны быть одинаковой длины.
    """

    if len(source_dirs) != len(graph_dirs):
        raise ValueError("source_dirs и graph_dirs должны быть одинаковой длины")

    for source_dir, graph_dir in zip(source_dirs, graph_dirs):
        source_dir = Path(source_dir)
        graph_dir = Path(graph_dir)

        if not source_dir.exists():
            print(f"⚠️ Папка не найдена: {source_dir}")
            continue

        graph_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(source_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️ Нет CSV файлов в папке: {source_dir}")
            continue

        print(f"\n📂 Построение spike-графиков для папки: {source_dir}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")

                required_cols = {"START_TIME", "SPIKE_FLAG"}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    print(f"⚠️ В файле {csv_file.name} нет колонок: {sorted(missing_cols)}")
                    continue

                df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
                df["SPIKE_FLAG"] = pd.to_numeric(df["SPIKE_FLAG"], errors="coerce").fillna(0).astype(int)

                df = df[df["START_TIME"].notna()].copy()

                if df.empty:
                    print(f"⚠️ Нет валидных данных для графика: {csv_file.name}")
                    continue

                # --- сортировка и расчёт НЧС ---
                df_sorted = df.sort_values("START_TIME").reset_index(drop=True)
                df_sorted["CUM_EVENTS"] = np.arange(1, len(df_sorted) + 1)
                df_sorted["CUM_NORM"] = df_sorted["CUM_EVENTS"] / df_sorted["CUM_EVENTS"].max()

                fig, ax = plt.subplots(figsize=(16, 8))

                # --- нормированный НЧС ---
                ax.plot(
                    df_sorted["START_TIME"],
                    df_sorted["CUM_NORM"],
                    color="black",
                    linewidth=2,
                    label="НЧС (нормированный)"
                )

                # --- сегментация всплесков ---
                spike = df_sorted["SPIKE_FLAG"].values
                times = df_sorted["START_TIME"].values

                segments = []
                in_seg = False
                start_t = None

                for i in range(len(spike)):
                    if spike[i] == 1 and not in_seg:
                        in_seg = True
                        start_t = times[i]
                    if spike[i] == 0 and in_seg:
                        segments.append((start_t, times[i - 1]))
                        in_seg = False

                if in_seg:
                    segments.append((start_t, times[-1]))

                # --- рисуем каждый сегмент красным ---
                for idx, (s, e) in enumerate(segments):
                    ax.axvline(s, color="red", linestyle="--", linewidth=1.2)
                    ax.axvline(e, color="red", linestyle="--", linewidth=1.2)

                    ax.axvspan(
                        s, e,
                        color="red",
                        alpha=0.15,
                        label="Период всплеска" if idx == 0 else None
                    )

                # --- ограничиваем диапазоны ---
                ax.set_ylim(0, 1)
                ax.set_xlim(df_sorted["START_TIME"].min(), df_sorted["START_TIME"].max())

                ax.set_title(
                    f"Нормированный НЧС с сезонными всплесками: {csv_file.stem}",
                    fontsize=16
                )
                ax.set_xlabel("Время")
                ax.set_ylabel("Нормированный НЧС")

                ax.grid(linestyle="--", alpha=0.4)

                # --- легенда без дублей ---
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                out_path = graph_dir / f"{csv_file.stem}_cumulative_spike.pdf"
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"📈 График НЧС со всплесками сохранён: {out_path}")

            except Exception as e:
                print(f"❌ Ошибка построения графика для {csv_file.name}: {e}")

# ============================================================
# 4. ГРАФИК НЧС + ВЕРТИКАЛЬНЫЕ ЛИНИИ СБОЕВ CUSUM
# ============================================================

def plot_cumulative_events_with_cusum_alarms(
    df: pd.DataFrame,
    save_dir: Path,
    filename_stem: str,
    alarm_col: str = "CUSUM_ALARM",
    spike_col: str = "SPIKE_FLAG",
):
    """
    График накопленного числа событий (INDEX) с наложением:
        • периодов всплесков (SPIKE_FLAG) — закрашенные красные зоны
        • CUSUM-сигналов (CUSUM_ALARM)   — синие вертикальные линии

    Позволяет наглядно проверить, что CUSUM-сигналы
    попадают внутрь периодов сбоя.

    Параметры
    ----------
    df            : DataFrame с колонками START_TIME, INDEX, CUSUM_ALARM
                    (и опционально SPIKE_FLAG)
    save_dir      : папка для сохранения графика
    filename_stem : базовое имя файла (без расширения)
    alarm_col     : имя колонки с флагом тревоги (по умолчанию CUSUM_ALARM)
    spike_col     : имя колонки с периодом всплеска (по умолчанию SPIKE_FLAG)
                    если колонка отсутствует — зоны всплесков не рисуются
    """

    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{filename_stem}_cumulative_cusum.pdf"

    # --- подготовка данных ---
    df = df.copy()
    df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
    df = df.sort_values("START_TIME").reset_index(drop=True)

    has_spike = spike_col in df.columns

    plt.figure(figsize=(12, 6))

    # =====================================================
    # 1. Та же кривая НЧС (INDEX), как в plot_cumulative_events
    # =====================================================
    plt.plot(
        df["START_TIME"], df["INDEX"],
        linewidth=2, color="black",
        label="Накопленное число событий"
    )

    # =====================================================
    # 2. ПЕРИОДЫ ВСПЛЕСКОВ (SPIKE_FLAG) — если колонка есть
    # =====================================================
    if has_spike:
        spike = df[spike_col].values
        times = df["START_TIME"].values

        segments = []
        in_seg = False
        start_t = None

        for i in range(len(spike)):
            if spike[i] == 1 and not in_seg:
                in_seg = True
                start_t = times[i]
            if spike[i] == 0 and in_seg:
                segments.append((start_t, times[i - 1]))
                in_seg = False

        if in_seg:
            segments.append((start_t, times[-1]))

        for idx, (s, e) in enumerate(segments):
            plt.axvline(s, color="red", linestyle="--", linewidth=1.2)
            plt.axvline(e, color="red", linestyle="--", linewidth=1.2)
            plt.axvspan(
                s, e,
                color="red",
                alpha=0.15,
                label="Период всплеска" if idx == 0 else None
            )

    # =====================================================
    # 3. ВЕРТИКАЛЬНЫЕ ЛИНИИ CUSUM
    # =====================================================
    alarm_times = df.loc[df[alarm_col] == 1, "START_TIME"]

    for i, t in enumerate(alarm_times):
        plt.axvline(
            t,
            linestyle="--",
            color="blue",
            linewidth=1.4,
            alpha=0.9,
            label="CUSUM-сбой" if i == 0 else None
        )

    # =====================================================
    # 4. Горизонтальная линия y = 1
    # =====================================================
    plt.axhline(1, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

    # =====================================================
    # 5. Границы и оформление
    # =====================================================
    plt.xlim(df["START_TIME"].min(), df["START_TIME"].max())
    plt.ylim(0, 1)

    plt.title(
        "График накопленного числа событий:\n"
        "периоды всплесков и CUSUM-сигналы",
        fontsize=14
    )
    plt.xlabel("Время", fontsize=12)
    plt.ylabel("Нормированный индекс событий", fontsize=12)
    plt.grid(alpha=0.4)

    # --- легенда без дублей ---
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 CUSUM-график с периодами всплесков сохранён: {out_path}")

    return out_path


# ============================================================
# 5. БАТЧ-ПОСТРОЕНИЕ ГРАФИКОВ CUSUM (папки дорог и департаментов)
# ============================================================

def _find_cusum_full_csvs(directory: Path) -> list[Path]:
    """
    Возвращает список CSV-файлов с результатами CUSUM из папки directory.

    Признаки «полного» CUSUM-файла (имеет колонку CUSUM_ALARM):
        • заканчивается на _cusum_full.csv  — новое соглашение
        • либо не заканчивается на _cusum_events.csv
          и не является batch_summary.csv   — старое соглашение

    Файлы сортируются по имени.
    """
    all_csvs = sorted(directory.glob("*.csv"))

    result = []
    for p in all_csvs:
        name = p.name
        # явное новое соглашение
        if name.endswith("_cusum_full.csv"):
            result.append(p)
            continue
        # старое соглашение: исключаем events и summary
        if name.endswith("_cusum_events.csv"):
            continue
        if name == "batch_summary.csv":
            continue
        result.append(p)

    return result


def _stem_from_cusum_full(path: Path) -> str:
    """
    Возвращает базовое имя файла без CUSUM-суффикса:
        Дальневосточная-2025_cusum_full  →  Дальневосточная-2025
        Дальневосточная-2025             →  Дальневосточная-2025
    """
    stem = path.stem
    if stem.endswith("_cusum_full"):
        stem = stem[: -len("_cusum_full")]
    return stem


def plot_cusum_batch(
    *,
    roads_cusum_dir: Path | None = None,
    departments_cusum_dir: Path | None = None,
    roads_graph_dir: Path | None = None,
    departments_graph_dir: Path | None = None,
    alarm_col: str = "CUSUM_ALARM",
    spike_col: str = "SPIKE_FLAG",
) -> list[dict]:
    """
    Батч-построение CUSUM-графиков по папкам с результатами.

    Структура входных папок
    -----------------------
        roads_cusum_dir/
            Дальневосточная-2025.csv          ← полный файл с CUSUM_ALARM
            Дальневосточная-2025_cusum_events.csv
            ...
        departments_cusum_dir/
            CSH-2025.csv
            CSH-2025_cusum_events.csv
            ...

    Структура выходных папок
    ------------------------
        roads_graph_dir/
            Дальневосточная-2025_cumulative_cusum.pdf
            Горьковская-2025_cumulative_cusum.pdf
            ...
        departments_graph_dir/
            CSH-2025_cumulative_cusum.pdf
            ...

    Параметры
    ---------
    roads_cusum_dir       : папка с CUSUM-результатами по дорогам
                            (None — пропустить)
    departments_cusum_dir : папка с CUSUM-результатами по департаментам
                            (None — пропустить)
    roads_graph_dir       : папка для графиков дорог
                            (обязательна, если roads_cusum_dir задан)
    departments_graph_dir : папка для графиков департаментов
                            (обязательна, если departments_cusum_dir задан)
    alarm_col             : колонка с флагом тревоги (CUSUM_ALARM)
    spike_col             : колонка с периодом всплеска (SPIKE_FLAG);
                            если отсутствует в файле — зоны не рисуются

    Возвращает
    ----------
    list[dict] — сводная таблица:
        source      : "road" | "department"
        csv_name    : имя входного файла
        stem        : базовое имя (Дальневосточная-2025)
        graph_path  : путь к сохранённому PDF
        alarm_count : количество CUSUM-тревог в файле
        error       : текст ошибки или None
    """
    from typing import Optional

    if roads_cusum_dir is None and departments_cusum_dir is None:
        raise ValueError(
            "Укажите хотя бы одну из папок: "
            "roads_cusum_dir или departments_cusum_dir"
        )

    # ── сборка задач: (csv_path, graph_dir, source_label) ────────────────────
    tasks: list[tuple[Path, Path, str]] = []

    if roads_cusum_dir is not None:
        if roads_graph_dir is None:
            raise ValueError("roads_graph_dir обязателен при заданном roads_cusum_dir")
        files = _find_cusum_full_csvs(Path(roads_cusum_dir))
        if not files:
            print(f"[WARN] В папке дорог не найдено CUSUM-файлов: {roads_cusum_dir}")
        for f in files:
            tasks.append((f, Path(roads_graph_dir), "road"))

    if departments_cusum_dir is not None:
        if departments_graph_dir is None:
            raise ValueError(
                "departments_graph_dir обязателен при заданном departments_cusum_dir"
            )
        files = _find_cusum_full_csvs(Path(departments_cusum_dir))
        if not files:
            print(f"[WARN] В папке департаментов не найдено CUSUM-файлов: {departments_cusum_dir}")
        for f in files:
            tasks.append((f, Path(departments_graph_dir), "department"))

    if not tasks:
        print("[WARN] Нет файлов для построения графиков.")
        return []

    total = len(tasks)
    print(f"\n{'='*52}")
    print(f"  CUSUM PLOT BATCH — файлов: {total}")
    print(f"{'='*52}\n")

    summary: list[dict] = []

    for idx, (csv_path, graph_dir, source) in enumerate(tasks, start=1):
        stem = _stem_from_cusum_full(csv_path)
        label = f"[{idx}/{total}] {source.upper()} | {stem}"
        print(f"  {label}")

        record: dict = {
            "source":      source,
            "csv_name":    csv_path.name,
            "stem":        stem,
            "graph_path":  None,
            "alarm_count": 0,
            "error":       None,
        }

        try:
            df = pd.read_csv(csv_path)

            if alarm_col not in df.columns:
                raise ValueError(
                    f"Колонка '{alarm_col}' не найдена в {csv_path.name}. "
                    f"Доступные: {list(df.columns)}"
                )

            alarm_count = int(df[alarm_col].sum())

            out_path = plot_cumulative_events_with_cusum_alarms(
                df=df,
                save_dir=graph_dir,
                filename_stem=stem,
                alarm_col=alarm_col,
                spike_col=spike_col,
            )

            record["alarm_count"] = alarm_count
            record["graph_path"]  = str(out_path)

        except Exception as exc:
            record["error"] = str(exc)
            print(f"  [ERROR] {label} — {exc}")

    ok_count  = sum(1 for r in summary if r["error"] is None)
    err_count = sum(1 for r in summary if r["error"] is not None)

    print(f"\n{'='*52}")
    print(f"  CUSUM PLOT BATCH ЗАВЕРШЁН")
    print(f"  Успешно:  {ok_count} / {total}")
    print(f"  Ошибок:   {err_count}")
    print(f"{'='*52}\n")

    return summary


def plot_cusum_batch_html(
    *,
    roads_cusum_dir: Path | None = None,
    departments_cusum_dir: Path | None = None,
    roads_graph_dir: Path | None = None,
    departments_graph_dir: Path | None = None,
    alarm_col: str = "CUSUM_ALARM",
    spike_col: str = "SPIKE_FLAG",
) -> list[dict]:
    """
    Батч-построение интерактивных HTML-графиков CUSUM (Plotly).

    Для каждого CUSUM-файла создаёт HTML с возможностью:
        • зума колесом мыши / выделением области
        • пана зажатым ЛКМ
        • тултипов: дата, индекс, сезон, δ̂
        • двойной клик — сброс масштаба

    Состав графика:
        Верхняя панель  — кривая НЧС (INDEX), зоны всплесков, алармы ▼
        Нижняя панель   — статистика CUSUM S + порог H (если есть колонка CUSUM_S)

    Параметры
    ---------
    roads_cusum_dir       : папка с CUSUM-результатами по дорогам (None — пропустить)
    departments_cusum_dir : папка с CUSUM-результатами по департаментам (None — пропустить)
    roads_graph_dir       : папка для HTML-графиков дорог
    departments_graph_dir : папка для HTML-графиков департаментов
    alarm_col             : колонка флага тревоги (CUSUM_ALARM)
    spike_col             : колонка периода всплеска (SPIKE_FLAG)

    Возвращает
    ----------
    list[dict] — сводная таблица: source, csv_name, stem, graph_path, alarm_count, error
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _build_html(df: pd.DataFrame, stem: str, alarm_col: str, spike_col: str) -> go.Figure:
        df = df.copy()
        df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
        df = df[df["START_TIME"].notna()].sort_values("START_TIME").reset_index(drop=True)

        has_spike   = spike_col in df.columns
        has_cusum_s = "CUSUM_S" in df.columns
        has_h       = "H" in df.columns
        has_season  = "SEASON" in df.columns
        has_delta   = "DELTA_HAT" in df.columns

        alarm_mask = pd.to_numeric(df[alarm_col], errors="coerce").fillna(0) > 0
        alarm_df   = df[alarm_mask].copy()
        n_alarms   = int(alarm_mask.sum())

        n_rows  = 2 if has_cusum_s else 1
        row_h   = [0.68, 0.32] if has_cusum_s else [1.0]
        s_titles = [stem, "CUSUM статистика S"] if has_cusum_s else [stem]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            row_heights=row_h,
            subplot_titles=s_titles,
            vertical_spacing=0.06,
        )

        times = df["START_TIME"]
        index = df["INDEX"]

        # ── 1. Кривая НЧС ────────────────────────────────────────────────────
        ht_main = (
            "<b>%{x|%d.%m.%Y %H:%M}</b><br>Индекс: %{y:.4f}<br>"
            + ("Сезон: %{customdata}<extra></extra>" if has_season
               else "<extra></extra>")
        )
        fig.add_trace(go.Scatter(
            x=times, y=index,
            mode="lines",
            name="НЧС (INDEX)",
            line=dict(color="black", width=2),
            hovertemplate=ht_main,
            customdata=df["SEASON"].values if has_season else None,
        ), row=1, col=1)

        # ── 2. Периоды всплесков ──────────────────────────────────────────────
        if has_spike:
            spike_vals = pd.to_numeric(df[spike_col], errors="coerce").fillna(0).values
            in_seg, seg_start, first = False, None, True
            for k in range(len(spike_vals)):
                if spike_vals[k] > 0 and not in_seg:
                    in_seg, seg_start = True, times.iloc[k]
                if spike_vals[k] == 0 and in_seg:
                    fig.add_vrect(
                        x0=seg_start, x1=times.iloc[k - 1],
                        fillcolor="red", opacity=0.12,
                        layer="below", line_width=0,
                        name="Период всплеска" if first else None,
                        showlegend=first,
                    )
                    in_seg, first = False, False
            if in_seg:
                fig.add_vrect(
                    x0=seg_start, x1=times.iloc[-1],
                    fillcolor="red", opacity=0.12,
                    layer="below", line_width=0,
                    name="Период всплеска" if first else None,
                    showlegend=first,
                )

        # ── 3. Алармы ─────────────────────────────────────────────────────────
        if n_alarms > 0:
            alarm_y = np.interp(
                alarm_df["START_TIME"].values.astype(np.int64),
                times.values.astype(np.int64),
                index.values,
            )

            # Тултип для маркеров
            cd_cols = []
            if has_season and "SEASON" in alarm_df.columns:
                cd_cols.append(alarm_df["SEASON"].values)
            if has_delta and "DELTA_HAT" in alarm_df.columns:
                cd_cols.append(alarm_df["DELTA_HAT"].round(3).values)
            alarm_cd = np.column_stack(cd_cols) if cd_cols else None

            ht_alarm = "<b>⚠ CUSUM-аларм</b><br>%{x|%d.%m.%Y %H:%M}<br>"
            ci = 0
            if has_season and "SEASON" in alarm_df.columns:
                ht_alarm += f"Сезон: %{{customdata[{ci}]}}<br>"; ci += 1
            if has_delta and "DELTA_HAT" in alarm_df.columns:
                ht_alarm += f"δ̂ = %{{customdata[{ci}]:.2f}}<br>"
            ht_alarm += "<extra></extra>"

            if alarm_cd is None:
                ht_alarm = "<b>⚠ CUSUM-аларм</b><br>%{x|%d.%m.%Y %H:%M}<extra></extra>"

            fig.add_trace(go.Scatter(
                x=alarm_df["START_TIME"], y=alarm_y,
                mode="markers",
                name=f"CUSUM-аларм ({n_alarms})",
                marker=dict(symbol="triangle-down", size=12,
                            color="red", line=dict(color="darkred", width=1)),
                hovertemplate=ht_alarm,
                customdata=alarm_cd,
            ), row=1, col=1)

            for t_alarm in alarm_df["START_TIME"]:
                fig.add_vline(
                    x=t_alarm,
                    line=dict(color="red", width=1.4, dash="dash"),
                )

        # ── 4. CUSUM-статистика S ─────────────────────────────────────────────
        if has_cusum_s:
            cusum_s = pd.to_numeric(df["CUSUM_S"], errors="coerce").fillna(0)
            fig.add_trace(go.Scatter(
                x=times, y=cusum_s,
                mode="lines",
                name="CUSUM S",
                line=dict(color="darkorange", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,140,0,0.13)",
                hovertemplate="<b>%{x|%d.%m.%Y %H:%M}</b><br>S = %{y:.3f}<extra></extra>",
            ), row=2, col=1)

            if has_h:
                h_med = float(pd.to_numeric(df["H"], errors="coerce").median())
                if np.isfinite(h_med) and h_med > 0:
                    fig.add_hline(
                        y=h_med,
                        line=dict(color="red", width=1.4, dash="dash"),
                        annotation_text=f"h = {h_med:.2f}",
                        annotation_position="top right",
                        row=2, col=1,
                    )

            if n_alarms > 0:
                for t_alarm in alarm_df["START_TIME"]:
                    fig.add_vline(
                        x=t_alarm,
                        line=dict(color="red", width=1.0, dash="dot"),
                    )

        # ── оформление ────────────────────────────────────────────────────────
        alarm_label = (f"  —  алармов: <b>{n_alarms}</b>"
                       if n_alarms > 0 else "  —  алармов не выявлено")
        fig.update_layout(
            title=dict(text=f"{stem}{alarm_label}", font=dict(size=14)),
            hovermode="x unified",
            height=520 if has_cusum_s else 380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=30, t=75, b=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb",
                         tickformat="%d %b %Y")
        fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
        fig.update_yaxes(title_text="Индекс (0–1)", row=1, col=1)
        if has_cusum_s:
            fig.update_yaxes(title_text="CUSUM S", row=2, col=1)
            fig.update_xaxes(title_text="Дата", row=2, col=1)
        else:
            fig.update_xaxes(title_text="Дата", row=1, col=1)

        return fig

    # ── сборка задач ──────────────────────────────────────────────────────────
    if roads_cusum_dir is None and departments_cusum_dir is None:
        raise ValueError(
            "Укажите хотя бы одну из папок: roads_cusum_dir или departments_cusum_dir"
        )

    tasks: list[tuple[Path, Path, str]] = []

    if roads_cusum_dir is not None:
        if roads_graph_dir is None:
            raise ValueError("roads_graph_dir обязателен при заданном roads_cusum_dir")
        for f in _find_cusum_full_csvs(Path(roads_cusum_dir)):
            tasks.append((f, Path(roads_graph_dir), "road"))
        if not tasks:
            print(f"[WARN] В папке дорог не найдено CUSUM-файлов: {roads_cusum_dir}")

    if departments_cusum_dir is not None:
        if departments_graph_dir is None:
            raise ValueError(
                "departments_graph_dir обязателен при заданном departments_cusum_dir"
            )
        dept_tasks = []
        for f in _find_cusum_full_csvs(Path(departments_cusum_dir)):
            dept_tasks.append((f, Path(departments_graph_dir), "department"))
        if not dept_tasks:
            print(f"[WARN] В папке департаментов не найдено CUSUM-файлов: {departments_cusum_dir}")
        tasks.extend(dept_tasks)

    if not tasks:
        print("[WARN] Нет файлов для построения графиков.")
        return []

    total = len(tasks)
    print(f"\n{'='*52}")
    print(f"  CUSUM HTML BATCH — файлов: {total}")
    print(f"{'='*52}\n")

    summary: list[dict] = []

    for idx, (csv_path, graph_dir, source) in enumerate(tasks, start=1):
        stem  = _stem_from_cusum_full(csv_path)
        label = f"[{idx}/{total}] {source.upper()} | {stem}"
        print(f"  {label}")

        record: dict = {
            "source":      source,
            "csv_name":    csv_path.name,
            "stem":        stem,
            "graph_path":  None,
            "alarm_count": 0,
            "error":       None,
        }

        try:
            df = pd.read_csv(csv_path)

            if alarm_col not in df.columns:
                raise ValueError(
                    f"Колонка '{alarm_col}' не найдена в {csv_path.name}. "
                    f"Доступные: {list(df.columns)}"
                )

            alarm_count = int(
                pd.to_numeric(df[alarm_col], errors="coerce").fillna(0).sum()
            )

            fig = _build_html(df, stem, alarm_col, spike_col)

            graph_dir.mkdir(parents=True, exist_ok=True)
            out_path = graph_dir / f"{stem}_cumulative_cusum.html"
            fig.write_html(
                out_path,
                include_plotlyjs="cdn",
                full_html=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )

            record["alarm_count"] = alarm_count
            record["graph_path"]  = str(out_path)
            print(f"    алармов={alarm_count} → {out_path.name}")

        except Exception as exc:
            record["error"] = str(exc)
            print(f"  [ERROR] {label} — {exc}")

        summary.append(record)

    ok_count  = sum(1 for r in summary if r["error"] is None)
    err_count = sum(1 for r in summary if r["error"] is not None)

    print(f"\n{'='*52}")
    print(f"  CUSUM HTML BATCH ЗАВЕРШЁН")
    print(f"  Успешно:  {ok_count} / {total}")
    print(f"  Ошибок:   {err_count}")
    print(f"{'='*52}\n")

    return summary


# Функция для оценки коэффициента авторегрессии первого порядка (rho) по полю TIME_DIFF и теста Ljung–Box для всех CSV-файлов в нескольких папках, с сохранением отчёта в итоговый CSV.
def estimate_ar1_for_directories(
    source_dirs: list[Path],
    output_csv: Path,
    ljung_box_lags: int = 10,
) -> None:
    """
    Рассчитывает коэффициент авторегрессии первого порядка (RHO_AR1)
    по полю TIME_DIFF для всех CSV-файлов в нескольких папках
    и дополнительно выполняет тест Ljung–Box.

    Параметры:
        source_dirs      : список папок с CSV-файлами
        output_csv       : путь к итоговому CSV-отчёту
        ljung_box_lags   : число лагов для теста Ljung–Box

    В итоговый CSV записываются:
        SOURCE_DIR
        FILE_NAME
        ROWS_TOTAL
        ROWS_VALID
        TIME_DIFF_MEAN
        TIME_DIFF_STD
        RHO_AR1
        LJUNG_BOX_LAGS
        LJUNG_BOX_PVALUE
        LJUNG_BOX_AUTOCORR
    """

    records = []

    for source_dir in source_dirs:
        source_dir = Path(source_dir)

        if not source_dir.exists():
            print(f"⚠️ Папка не найдена: {source_dir}")
            continue

        csv_files = sorted(source_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️ Нет CSV файлов в папке: {source_dir}")
            continue

        print(f"\n📂 Анализ папки: {source_dir}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")

                if "TIME_DIFF" not in df.columns:
                    print(f"⚠️ В файле нет TIME_DIFF: {csv_file.name}")
                    continue

                x = pd.to_numeric(df["TIME_DIFF"], errors="coerce").dropna().to_numpy()

                rows_total = len(df)
                rows_valid = len(x)

                if rows_valid < 3:
                    print(f"⚠️ Недостаточно данных для оценки rho: {csv_file.name}")
                    records.append({
                        "SOURCE_DIR": str(source_dir),
                        "FILE_NAME": csv_file.name,
                        "ROWS_TOTAL": rows_total,
                        "ROWS_VALID": rows_valid,
                        "TIME_DIFF_MEAN": np.nan,
                        "TIME_DIFF_STD": np.nan,
                        "RHO_AR1": np.nan,
                        "LJUNG_BOX_LAGS": ljung_box_lags,
                        "LJUNG_BOX_PVALUE": np.nan,
                        "LJUNG_BOX_AUTOCORR": np.nan,
                    })
                    continue

                # --- Оценка RHO_AR1 ---
                x_mean = np.mean(x)
                numerator = np.sum((x[1:] - x_mean) * (x[:-1] - x_mean))
                denominator = np.sum((x - x_mean) ** 2)

                rho = np.nan if denominator == 0 else numerator / denominator

                # --- Тест Ljung–Box ---
                # число лагов не должно быть >= длины ряда
                lb_lags = min(ljung_box_lags, max(1, rows_valid - 1))

                if rows_valid < 5:
                    lb_pvalue = np.nan
                    lb_autocorr = np.nan
                else:
                    lb_result = acorr_ljungbox(x, lags=[lb_lags], return_df=True)
                    lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
                    lb_autocorr = lb_pvalue <= 0.05

                records.append({
                    "SOURCE_DIR": str(source_dir),
                    "FILE_NAME": csv_file.name,
                    "ROWS_TOTAL": rows_total,
                    "ROWS_VALID": rows_valid,
                    "TIME_DIFF_MEAN": float(np.mean(x)),
                    "TIME_DIFF_STD": float(np.std(x, ddof=1)) if rows_valid > 1 else np.nan,
                    "RHO_AR1": float(rho) if pd.notna(rho) else np.nan,
                    "LJUNG_BOX_LAGS": lb_lags,
                    "LJUNG_BOX_PVALUE": lb_pvalue,
                    "LJUNG_BOX_AUTOCORR": lb_autocorr,
                })

                rho_str = f"{rho:.4f}" if pd.notna(rho) else "nan"
                pval_str = f"{lb_pvalue:.4f}" if pd.notna(lb_pvalue) else "nan"

                print(
                    f"✅ {csv_file.name}: "
                    f"rho = {rho_str}, "
                    f"Ljung–Box p-value = {pval_str}"
                )

            except Exception as e:
                print(f"❌ Ошибка обработки {csv_file.name}: {e}")
                records.append({
                    "SOURCE_DIR": str(source_dir),
                    "FILE_NAME": csv_file.name,
                    "ROWS_TOTAL": np.nan,
                    "ROWS_VALID": np.nan,
                    "TIME_DIFF_MEAN": np.nan,
                    "TIME_DIFF_STD": np.nan,
                    "RHO_AR1": np.nan,
                    "LJUNG_BOX_LAGS": ljung_box_lags,
                    "LJUNG_BOX_PVALUE": np.nan,
                    "LJUNG_BOX_AUTOCORR": np.nan,
                })

    if not records:
        print("❌ Нет данных для формирования отчёта")
        return

    report_df = pd.DataFrame(records)

    report_df = report_df.sort_values(
        by=["SOURCE_DIR", "FILE_NAME"],
        kind="stable"
    ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n📄 Отчёт сохранён: {output_csv}")





