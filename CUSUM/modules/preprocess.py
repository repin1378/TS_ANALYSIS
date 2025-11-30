import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# 1) Обработка одного файла + сохранение обновлённого CSV
# ============================================================
def preprocess_dataframe(csv_file: Path, save_dir: Path = None):
    """
    Загружает CSV, рассчитывает DELTA_TIME, DELTA_MINUTES, TIME_DIFF, INDEX
    и сохраняет обновлённый CSV в save_dir.

    Возвращает обработанный DataFrame.
    """

    df = pd.read_csv(csv_file)

    # Преобразование времени
    df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")

    # Сортировка по времени
    df = df.sort_values("START_TIME").reset_index(drop=True)

    # DELTA_TIME — timedelta от первого события
    df["DELTA_TIME"] = df["START_TIME"] - df["START_TIME"].iloc[0]

    # DELTA_MINUTES
    df["DELTA_MINUTES"] = df["DELTA_TIME"].dt.total_seconds() / 60

    # TIME_DIFF — разница между соседними событиями
    df["TIME_DIFF"] = df["DELTA_MINUTES"].diff().fillna(0)

    # INDEX — нормированный индекс
    df["INDEX"] = df.index / len(df)

    # ---------------- Сохранение файла ----------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        out_path = save_dir / csv_file.name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"💾 Обработанный CSV сохранён: {out_path}")

    return df


# ============================================================
# 2) Гистограмма TIME_DIFF
# ============================================================
def save_histogram(df: pd.DataFrame, graph_dir: Path, file_name: str):
    """
    Строит гистограмму TIME_DIFF начиная с 0.
    DPI и hist_step подбираются автоматически под размер выборки.
    """

    graph_dir.mkdir(parents=True, exist_ok=True)
    out_path = graph_dir / f"{file_name}.pdf"

    n = len(df)
    tmax = df["TIME_DIFF"].max()

    # ---------------------------
    # 1) Автоматический выбор DPI
    # ---------------------------
    if n < 500:
        dpi = 150
    elif n < 5000:
        dpi = 200
    elif n < 50000:
        dpi = 300
    else:
        dpi = 400

    # ---------------------------
    # 2) Автоматический выбор hist_step
    # ---------------------------
    if n < 500:
        hist_step = max(2, tmax / 20)     # 20 бинов
    elif n < 5000:
        hist_step = max(1, tmax / 40)     # 40 бинов
    elif n < 50000:
        hist_step = max(0.5, tmax / 60)   # 60 бинов
    else:
        hist_step = max(0.25, tmax / 80)  # 80 бинов

    # округляем шаг до красивого числа
    if hist_step > 10:
        hist_step = round(hist_step, -1)   # десятки
    elif hist_step > 1:
        hist_step = round(hist_step, 1)    # десятые
    else:
        hist_step = round(hist_step, 2)    # сотые

    print(f"📌 Автонастройка: n={n}, max={tmax:.2f}, hist_step={hist_step}, dpi={dpi}")

    # --------- Бины -----------
    xmin = 0
    xmax = ((tmax // hist_step) + 1) * hist_step
    bin_edges = np.arange(xmin, xmax + hist_step, hist_step)

    # --------- Построение ---------
    plt.figure(figsize=(10, 5))
    plt.hist(df["TIME_DIFF"], bins=bin_edges,
             edgecolor='black', alpha=0.7)

    plt.xlabel("Интервалы между событиями (мин)")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма TIME_DIFF — {file_name}")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"📊 Гистограмма сохранена: {out_path}")

    return out_path

# ============================================================
# 3) График НЧС
# ============================================================

def plot_cumulative_events(df: pd.DataFrame, graph_dir: Path, file_name: str):
    """
    Строит график накопленного числа событий:
      - INDEX по START_TIME
      - квартальные линии
      - сезонные линии
      - горизонтальная линия y=1
      - обрезка графика по последнему событию
    """

    graph_dir.mkdir(parents=True, exist_ok=True)
    out_path = graph_dir / f"{file_name}_cumulative.pdf"

    plt.figure(figsize=(12, 6))

    # === 1. График INDEX ===
    plt.plot(
        df["START_TIME"], df["INDEX"],
        linewidth=2, color="black",
        label="Накопленное число событий"
    )

    # Диапазон времени
    start = df["START_TIME"].min().normalize()
    end = df["START_TIME"].max().normalize()

    # === 2. Квартальные линии ===
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

    # === 3. Сезонные линии ===
    season_offsets = [(3, 1), (6, 1), (9, 1), (12, 1)]
    season_names = ["Весна", "Лето", "Осень", "Зима"]

    years = range(start.year, end.year + 1)
    season_lines = []

    for year in years:
        for (month, day), name in zip(season_offsets, season_names):
            season_date = pd.Timestamp(year, month, day)
            if start <= season_date <= end:
                season_lines.append((season_date, name))

    for i, (d, name) in enumerate(season_lines):
        plt.axvline(
            d,
            linestyle=":",
            color="tab:blue",
            linewidth=1.4,
            alpha=0.8,
            label="Сезон" if i == 0 else None
        )

    # === 4. Горизонтальная линия y = 1 ===
    plt.axhline(1, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

    # === 5. Границы графика ===
    first_time = df["START_TIME"].min()
    last_time = df["START_TIME"].max()

    plt.xlim(first_time, last_time)
    plt.ylim(0, 1)

    # === 6. Подписи и стиль ===
    plt.title("График накопленного числа событий", fontsize=14)
    plt.xlabel("Время", fontsize=12)
    plt.ylabel("Нормированный индекс событий", fontsize=12)
    plt.grid(alpha=0.4)

    # === 7. Убираем дубликаты в легенде ===
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left")

    # === 8. Сохранение ===
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 График накопленного числа событий сохранён: {out_path}")
    return out_path

