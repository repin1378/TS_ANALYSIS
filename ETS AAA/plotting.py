"""
plotting.py

Классическая визуализация прогнозов:
- без разрыва между историей и прогнозом
- с месячной сеткой
- с защитой от отрицательных значений
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ======================================================
# ВСПОМОГАТЕЛЬНО
# ======================================================

def _concat_last_point(data: pd.Series, future_idx, forecast):
    """
    Соединяет последнюю точку истории с прогнозом,
    чтобы не было визуального разрыва.
    """
    x = [data.index[-1]] + list(future_idx)
    y = [data.iloc[-1]] + list(forecast)
    return x, y


# ======================================================
# ETS
# ======================================================

def plot_ets_forecast(
    data: pd.Series,
    future_idx: pd.DatetimeIndex,
    forecast: pd.Series,
    lower: pd.Series | None,
    upper: pd.Series | None,
    road: str,
    title: str,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(12, 5))

    # --- исторический ряд ---
    ax.plot(
        data.index,
        data.values,
        label="Historical",
        color="black",
        linewidth=1.6,
    )

    # --- защита от отрицательных значений ---
    forecast_plot = forecast.clip(lower=0)
    lower_plot = lower.clip(lower=0) if lower is not None else None
    upper_plot = upper.clip(lower=0) if upper is not None else None

    # --- соединённый прогноз ---
    fx, fy = _concat_last_point(data, future_idx, forecast_plot)

    ax.plot(
        fx,
        fy,
        label="Forecast",
        color="tab:blue",
        linewidth=2.2,
    )

    # --- доверительный интервал ---
    if lower_plot is not None and upper_plot is not None:
        ix = [data.index[-1]] + list(future_idx)
        ax.fill_between(
            ix,
            [data.iloc[-1]] + list(lower_plot),
            [data.iloc[-1]] + list(upper_plot),
            color="tab:blue",
            alpha=0.25,
            label="95% interval",
        )

    # --- ось X: годы + месяцы ---
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    ax.set_xlim(data.index.min(), future_idx.max())
    ax.set_title(f"{title} — {road}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ======================================================
# CatBoost
# ======================================================

def plot_catboost_forecast(
    data: pd.Series,
    future_idx: pd.DatetimeIndex,
    forecast: pd.Series,
    lower: pd.Series | None,
    upper: pd.Series | None,
    road: str,
    title: str,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(12, 5))

    # --- исторический ряд ---
    ax.plot(
        data.index,
        data.values,
        label="Historical",
        color="black",
        linewidth=1.6,
    )

    # --- защита от отрицательных значений ---
    forecast_plot = forecast.clip(lower=0)
    lower_plot = lower.clip(lower=0) if lower is not None else None
    upper_plot = upper.clip(lower=0) if upper is not None else None

    # --- соединённый прогноз ---
    fx, fy = _concat_last_point(data, future_idx, forecast_plot)

    ax.plot(
        fx,
        fy,
        label="Forecast",
        color="tab:orange",
        linewidth=2.2,
    )

    # --- доверительный интервал ---
    if lower_plot is not None and upper_plot is not None:
        ix = [data.index[-1]] + list(future_idx)
        ax.fill_between(
            ix,
            [data.iloc[-1]] + list(lower_plot),
            [data.iloc[-1]] + list(upper_plot),
            color="tab:orange",
            alpha=0.25,
            label="90% interval",
        )

    # --- ось X ---
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    ax.set_xlim(data.index.min(), future_idx.max())
    ax.set_title(f"{title} — {road}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
