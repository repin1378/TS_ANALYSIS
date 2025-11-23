# plotting.py
"""
Модуль построения графиков прогнозов ETS-модели.
Используется в ets_model.py.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_ets_forecast(data: pd.Series,
                      future_idx: pd.DatetimeIndex,
                      pred_full,
                      lower,
                      upper,
                      road: str,
                      outpath):
    """
    Строит и сохраняет график прогноза ETS(A,A,A) для одной дороги.

    Parameters
    ----------
    data : pd.Series
        Исторические значения временного ряда (наблюдения)
    future_idx : DatetimeIndex
        Даты прогноза (12 месяцев будущего)
    pred_full : array-like
        Прогнозные значения
    lower : array-like
        Нижняя доверительная граница (95%)
    upper : array-like
        Верхняя доверительная граница (95%)
    road : str
        Название дороги
    outpath : Path
        Путь к файлу PNG
    """

    plt.figure(figsize=(8, 4))
    plt.plot(data.index, data.values, label="наблюдения")
    plt.plot(future_idx, pred_full, "--", label="прогноз")
    plt.fill_between(future_idx, lower, upper,
                     color="orange", alpha=0.3, label="95% ДИ")
    plt.axvline(data.index[-1], color="gray", linestyle=":")
    plt.title(f"ETS(A,A,A) — {road} (прогноз 2026)")
    plt.xlabel("Дата")
    plt.ylabel("Значение")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()