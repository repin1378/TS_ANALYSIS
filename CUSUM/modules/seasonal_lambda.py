from pathlib import Path
import pandas as pd
import numpy as np
from fitter import Fitter
from scipy.stats import kstest
from typing import Optional


def _month_to_season(m: int) -> str:
    if m in (3, 4, 5):
        return "Весна"
    if m in (6, 7, 8):
        return "Лето"
    if m in (9, 10, 11):
        return "Осень"
    return "Зима"


def estimate_lambda_for_season(
    csv_dir: Path,
    road: str,
    category: str,
    season: str,
    target_year: Optional[int] = None,
    minutes_in_unit: Optional[float] = None,
    min_points: int = 30,
    ks_alpha: float = 0.05
):
    """
    Оценивает λ₀ для экспоненты по указанному сезону и году.
    target_year=None -> усреднение по всем годам.
    ks_alpha — порог p-value для KS теста.
    """

    files = list(csv_dir.glob("*.csv"))
    dfs = []
    years_used = set()

    for f in files:
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue

        _, yr = parts[0], parts[1]
        yr = int(yr)

        if target_year is not None and yr != target_year:
            continue

        df = pd.read_csv(f)
        df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
        df["MONTH"] = df["START_TIME"].dt.month
        df["SEASON"] = df["MONTH"].apply(_month_to_season)

        df = df[df["ROAD"] == road]
        df = df[df["CATEGORY"].astype(str) == str(category)]
        df = df[df["SEASON"] == season]

        if len(df) > 0:
            years_used.add(yr)
            dfs.append(df)

    if not dfs:
        yinfo = target_year if target_year is not None else "все годы"
        raise ValueError(
            f"Нет данных для '{road}', категории '{category}', сезона '{season}' за {yinfo}."
        )

    df_all = pd.concat(dfs, ignore_index=True)

    if "TIME_DIFF" not in df_all.columns:
        df_all = df_all.sort_values("START_TIME").reset_index(drop=True)
        delta = df_all["START_TIME"] - df_all["START_TIME"].iloc[0]
        dm = delta.dt.total_seconds() / 60
        df_all["TIME_DIFF"] = dm.diff().fillna(0)

    data = df_all["TIME_DIFF"].to_numpy()
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) < min_points:
        raise ValueError(
            f"Недостаточно данных для надёжной оценки λ₀: {len(data)} < {min_points}"
        )

    # λ_MLE
    mu = float(np.mean(data))
    lambda_mle = 1.0 / mu

    # λ_fitter
    f = Fitter(data, distributions=["expon"], timeout=10)
    f.fit()
    loc, scale = f.fitted_param["expon"]
    lambda_fitter = 1.0 / scale

    # KS
    ks_stat, ks_pvalue = kstest(data, "expon", args=(0, 1 / lambda_fitter))

    # ========= Новое: λ в расчёте на месяц =========
    MINUTES_IN_MONTH = 30 * 24 * 60  # 43 200
    lambda_month = lambda_fitter * MINUTES_IN_MONTH

    result = {
        "season": season,
        "road": road,
        "category": str(category),
        "years_used": sorted(years_used),
        "n": len(data),
        "lambda_mle": lambda_mle,
        "lambda_fitter": lambda_fitter,
        "lambda_month": lambda_month,     # ← добавлено
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ks_ok": ks_pvalue >= ks_alpha,
        "ks_alpha": ks_alpha
    }

    if minutes_in_unit is not None:
        result["lambda_unit"] = lambda_fitter * minutes_in_unit

    return result