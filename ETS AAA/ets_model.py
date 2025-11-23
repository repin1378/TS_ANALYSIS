# ets_model.py
"""
Модуль для выполнения ETS(A,A,A) прогнозирования.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from metrics import compute_metrics          # <-- расчёт метрик вынесен
from plotting import plot_ets_forecast       # <-- новый импорт

def run_ets_forecast(data: pd.DataFrame, roads: list, outdir: str = "ets_results"):
    """
    Выполняет ETS(A,A,A) прогнозирование для всех дорожных рядов.
    Возвращает forecasts_df, metrics_df.
    """

    train, test = data.iloc[:36], data.iloc[36:]
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    forecasts = []
    metrics = []

    for r in roads:

        y_train, y_test = train[r], test[r]

        # --- Модель на train ---
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        )
        fit = model.fit(optimized=True)
        pred_test = fit.forecast(12)

        # --- Модель на полной истории ---
        model_full = ExponentialSmoothing(
            data[r],
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        )
        fit_full = model_full.fit(optimized=True)
        pred_full = fit_full.forecast(12)

        future_idx = pd.date_range("2026-01-01", periods=12, freq="MS")

        # --- Интервалы ---
        resid_std = np.std(fit_full.resid)
        lower = pred_full - 1.96 * resid_std
        upper = pred_full + 1.96 * resid_std

        # --- Прогноз на 1 дорогу ---
        df_pred = pd.DataFrame({
            "road": r,
            "date": future_idx,
            "forecast": pred_full.values,
            "lower_95": lower.values,
            "upper_95": upper.values
        })
        forecasts.append(df_pred)

        # --- Метрики ---
        metrics.append({
            "road": r,
            **compute_metrics(y_test, pred_test, fit_full)
        })

        # --- График — теперь вызываем plotting.py ---
        plot_ets_forecast(
            data=data[r],
            future_idx=future_idx,
            pred_full=pred_full,
            lower=lower,
            upper=upper,
            road=r,
            outpath=outdir / f"forecast_{r}.png"
        )

    # --- Сводные таблицы ---
    forecasts_df = pd.concat(forecasts)
    metrics_df = pd.DataFrame(metrics).set_index("road")

    forecasts_df.to_csv(outdir / "forecasts_2026_with_CI.csv", index=False)
    metrics_df.to_csv(outdir / "ets_metrics.csv")

    print("✅ ETS(A,A,A) прогнозирование выполнено.")
    print(f"Файлы сохранены в: {outdir.resolve()}")

    return forecasts_df, metrics_df
