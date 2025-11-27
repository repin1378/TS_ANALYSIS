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
def save_histogram(df: pd.DataFrame, graph_dir: Path, file_name: str,
                   hist_step: float = 10):
    """
    Строит гистограмму TIME_DIFF начиная с 0.
    """

    graph_dir.mkdir(parents=True, exist_ok=True)
    out_path = graph_dir / f"{file_name}.pdf"

    # Ось X начинает с 0
    xmin = 0
    xmax = df["TIME_DIFF"].max()
    xmax = ((xmax // hist_step) + 1) * hist_step

    bin_edges = np.arange(xmin, xmax + hist_step, hist_step)

    plt.figure(figsize=(8, 5))
    plt.hist(df["TIME_DIFF"], bins=bin_edges,
             edgecolor='black', alpha=0.7)

    plt.xlabel("Интервалы между событиями (мин)")
    plt.ylabel("Частота")
    plt.title("Гистограмма")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(xmin, xmax)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
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
        - вертиковые пунктирные линии по кварталам
    """

    graph_dir.mkdir(parents=True, exist_ok=True)
    out_path = graph_dir / f"{file_name}_cumulative.pdf"

    plt.figure(figsize=(12, 6))

    # === 1. Сам график накопленного числа событий ===
    plt.plot(df["START_TIME"], df["INDEX"],
             linewidth=2, color="black",
             label="Накопленное число событий")

    # === 2. Добавляем линии по кварталам ===
    start = df["START_TIME"].min()
    end = df["START_TIME"].max()

    # Строим квартальные границы между минимальным и максимальным временем
    quarter_starts = pd.date_range(
        start=start.normalize(),
        end=end.normalize(),
        freq="QS"   # Quarter Start
    )

    for q in quarter_starts:
        if start <= q <= end:
            plt.axvline(
                q,
                linestyle="--",
                color="gray",
                linewidth=1.2,
                alpha=0.7,
                label="Квартальная граница" if q == quarter_starts[0] else None
            )

    # === 3. Настройки графика ===
    plt.title("График накопленного числа событий", fontsize=14)
    plt.xlabel("Время событий", fontsize=12)
    plt.ylabel("Нормированное значение INDEX", fontsize=12)

    plt.grid(alpha=0.4)
    plt.ylim(0, 1)

    # Легенда уникализируется автоматически
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # === 4. Сохранение графика ===
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 График накопленного числа событий сохранён: {out_path}")
    return out_path