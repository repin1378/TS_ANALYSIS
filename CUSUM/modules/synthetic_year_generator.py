import numpy as np
import pandas as pd
from pathlib import Path


SEASON_TO_MONTHS = {
    "Зима":  [1, 2],         # только Jan, Feb
    "Весна": [3, 4, 5],
    "Лето":  [6, 7, 8],
    "Осень": [9, 10, 11]
}


def generate_synthetic_year(
    lambda_dir: Path,
    road: str,
    category: str,
    year: int,
    out_dir: Path
) -> pd.DataFrame:

    """
    Генерирует синтетический годовой временной ряд:
    - читает lambda_0_<дорога>.csv
    - берёт λ_fitter и λ_month для каждого сезона
    - генерирует события экспонентой Exp(λ) помесячно
    - создаёт общий CSV за год с корректными полями

    Параметры:
        lambda_dir  — директория, где лежат lambda_0_Дорога.csv
        road         — дорога ("Октябрьская")
        category     — категория ("2", "3")
        year         — год генерации
        out_dir      — куда сохранить generated CSV
    """

    # ===== 1. Читаем lambda_0_<road>.csv =====
    lambda_file = lambda_dir / f"lambda_0_{road}.csv"
    df_lambdas = pd.read_csv(lambda_file)

    # фильтруем нужную категорию
    df_lambdas = df_lambdas[df_lambdas["CATEGORY"].astype(str) == str(category)]
    if df_lambdas.empty:
        raise ValueError(f"Нет λ₀ для дороги {road}, категории {category}")

    all_rows = []  # тут будем копить события всех сезонов

    # ===== 2. Генерация событий по сезонам =====
    for season, months in SEASON_TO_MONTHS.items():

        df_s = df_lambdas[df_lambdas["SEASON"] == season]
        if df_s.empty:
            continue

        lambda_minute = float(df_s["LAMBDA_FITTER"].iloc[0])
        lambda_month = float(df_s["LAMBDA_MONTH"].iloc[0])

        # сколько событий генерим на 1 месяц?
        n_month_events = max(1, int(round(lambda_month)))

        # Генерируем для каждого месяца сезона
        for m in months:

            start_time = pd.Timestamp(year=year, month=m, day=1, hour=0, minute=0)

            # Генерация интервалов с экспонентой Exp(λ)
            scale = 1.0 / lambda_minute
            time_diff = np.random.exponential(scale=scale, size=n_month_events)

            df = pd.DataFrame({"TIME_DIFF": time_diff})
            df["DELTA_MINUTES"] = df["TIME_DIFF"].cumsum()
            df["START_TIME"] = start_time + pd.to_timedelta(df["DELTA_MINUTES"], unit="m")
            df["SEASON"] = season
            df["MONTH"] = m

            # Добавляем параметры
            df["LAMBDA_MONTH"] = lambda_month

            all_rows.append(df)

    # ===== 3. Объединяем всё =====
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = df_all.sort_values("START_TIME").reset_index(drop=True)

    # ===== 4. Финальные поля =====
    df_all["DELTA_TIME"] = df_all["START_TIME"] - df_all["START_TIME"].iloc[0]
    df_all["INDEX"] = df_all.index / len(df_all)

    df_all["CATEGORY"] = str(category)
    df_all["ROAD"] = road
    df_all["DEPARTMENT"] = "SYNTHETIC"
    df_all["YEAR"] = year

    # переставляем колонки
    df_all = df_all[
        [
            "CATEGORY", "START_TIME", "ROAD", "DEPARTMENT", "YEAR",
            "DELTA_TIME", "DELTA_MINUTES", "TIME_DIFF", "INDEX",
            "SEASON", "MONTH", "LAMBDA_MONTH"
        ]
    ]

    # ===== 5. Сохраняем =====
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_{road}_{category}_{year}.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"📁 Сгенерирован синтетический годовой временной ряд: {out_path}")
    return df_all

def generate_synthetic_year_with_spikes_smooth(
    lambda_dir: Path,
    road: str,
    category: str,
    year: int,
    out_dir: Path,
    delta: float,             # λ1 = δ * λ0
    spike_days: int = 20,     # длительность всплеска
    transition_days: int = 5, # длительность плавного перехода
    k: float = 2.0            # скорость экспоненциального перехода
):
    """
    Генерация синтетического годового ряда с 4 всплесками (по одному в каждом сезоне),
    где λ(t) = λ0 → λ1 с плавным экспоненциальным переходом.
    """

    lambda_file = lambda_dir / f"lambda_0_{road}.csv"
    df_lambdas = pd.read_csv(lambda_file)

    # фильтруем по категории
    df_lambdas = df_lambdas[df_lambdas["CATEGORY"].astype(str) == str(category)]
    if df_lambdas.empty:
        raise ValueError(f"Нет λ₀ для {road}, CATEGORY={category}")

    all_rows = []

    for season, months in SEASON_TO_MONTHS.items():

        df_s = df_lambdas[df_lambdas["SEASON"] == season]
        if df_s.empty:
            continue

        lambda0 = float(df_s["LAMBDA_FITTER"].iloc[0])
        lambda_month = float(df_s["LAMBDA_MONTH"].iloc[0])
        lambda1 = lambda0 * delta                      # λ1 = δ λ0

        # ===== Определяем временные рамки сезона =====
        season_start = pd.Timestamp(year=year, month=months[0], day=1)

        # конец сезона = конец последнего месяца
        last_month = months[-1]
        season_end = pd.Timestamp(year=year, month=last_month, day=1) + pd.offsets.MonthEnd(1)

        season_minutes = (season_end - season_start).total_seconds() / 60

        # ===== Кол-во событий на сезон =====
        n_events = max(1, int(round(lambda_month * len(months))))

        # ===== Генерируем базовые интервалы по λ0 =====
        df = pd.DataFrame()
        df["TIME_DIFF"] = np.random.exponential(scale=1/lambda0, size=n_events)
        df["DELTA_MINUTES"] = df["TIME_DIFF"].cumsum()
        df["START_TIME"] = season_start + pd.to_timedelta(df["DELTA_MINUTES"], unit="m")

        # ===== Расчёт середины сезона =====
        season_mid = season_start + pd.Timedelta(minutes=season_minutes / 2)

        spike_start = season_mid - pd.Timedelta(days=spike_days / 2)
        spike_end   = season_mid + pd.Timedelta(days=spike_days / 2)

        transition_start = spike_start - pd.Timedelta(days=transition_days)

        # ===== Флаги =====
        df["SPIKE_FLAG"] = (
            (df["START_TIME"] >= spike_start) &
            (df["START_TIME"] <= spike_end)
        ).astype(int)

        df["TRANSITION_FLAG"] = (
            (df["START_TIME"] >= transition_start) &
            (df["START_TIME"] <  spike_start)
        ).astype(int)

        # ===== Построение λ_dynamic(t) =====
        lambda_dynamic = np.full(len(df), lambda0)

        # переходная зона
        trans_mask = df["TRANSITION_FLAG"] == 1
        if trans_mask.any():
            t_min = (df.loc[trans_mask, "START_TIME"] - transition_start).dt.total_seconds() / 60
            T = transition_days * 1440
            idx = df.index[trans_mask]
            lambda_dynamic[idx] = (
                lambda0 + (lambda1 - lambda0) * (1 - np.exp(-k * (t_min / T)))
            )

        # всплеск
        lambda_dynamic[df["SPIKE_FLAG"] == 1] = lambda1

        df["LAMBDA_DYNAMIC"] = lambda_dynamic
        df["LAMBDA0"] = lambda0
        df["LAMBDA1"] = lambda1
        df["LAMBDA_MONTH"] = lambda_month
        df["SEASON"] = season

        # ===== Перегенерируем интервалы =====
        df["TIME_DIFF"] = [np.random.exponential(scale=1/lam) for lam in lambda_dynamic]

        # заново считаем временные метки
        df["DELTA_MINUTES"] = df["TIME_DIFF"].cumsum()
        df["START_TIME"] = season_start + pd.to_timedelta(df["DELTA_MINUTES"], unit="m")

        all_rows.append(df)

    # ===== Объединяем весь год =====
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = df_all.sort_values("START_TIME").reset_index(drop=True)

    df_all["DELTA_TIME"] = df_all["START_TIME"] - df_all["START_TIME"].iloc[0]
    df_all["INDEX"] = df_all.index / len(df_all)

    df_all["ROAD"] = road
    df_all["CATEGORY"] = category
    df_all["DEPARTMENT"] = "SYNTHETIC"
    df_all["YEAR"] = year

    # ===== Сохранение =====
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_smooth_{road}_{category}_{year}.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"🔥 Сгенерирован synthetic CSV c 4 всплесками строго по серединам сезонов: {out_path}")
    return df_all

def generate_spike_report(
    df_synthetic: pd.DataFrame,
    out_dir: Path,
    road: str,
    category: str,
    year: int
):
    """
    Создаёт мини-отчёт по всплескам:
    - время начала всплеска
    - время окончания всплеска
    - λ0 и λ1 в минуту
    - λ0 и λ1 в пересчёте на месяц
    - сезон
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # сгруппируем по сезонам
    for season, df_season in df_synthetic.groupby("SEASON"):

        # где именно был всплеск?
        spike_mask = df_season["SPIKE_FLAG"] == 1

        if not spike_mask.any():
            continue

        spike_start = df_season.loc[spike_mask, "START_TIME"].min()
        spike_end   = df_season.loc[spike_mask, "START_TIME"].max()

        lambda0 = df_season["LAMBDA0"].iloc[0]
        lambda1 = df_season["LAMBDA1"].iloc[0]

        lambda0_month = df_season["LAMBDA_MONTH"].iloc[0]
        lambda1_month = lambda0_month * (lambda1 / lambda0)

        rows.append({
            "ROAD": road,
            "CATEGORY": category,
            "YEAR": year,
            "SEASON": season,
            "SPIKE_START": spike_start,
            "SPIKE_END": spike_end,
            "LAMBDA0": lambda0,
            "LAMBDA1": lambda1,
            "LAMBDA0_MONTH": lambda0_month,
            "LAMBDA1_MONTH": lambda1_month
        })

    df_report = pd.DataFrame(rows)

    out_path = out_dir / f"report_{road}_{category}_{year}.csv"
    df_report.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"📄 Мини-отчёт по всплескам сохранён: {out_path}")
    return df_report
