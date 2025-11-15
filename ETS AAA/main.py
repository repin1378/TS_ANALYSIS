"""
ETS(A,A,A) — прогнозирование с доверительными интервалами (95%)

Шаги:
1. Генерация синтетических временных рядов (2022–2025)
2. Обучение ETS(A,A,A)
3. Прогноз на 12 месяцев 2026 года
4. Расчёт доверительных интервалов
5. Сохранение прогнозов, метрик и графиков
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from series_generator import generate_series
from generation_config import ROADS, DATES, PARAMS
from generation_config import SEED  # если нужно использовать seed явно

# -----------------------------------------------------------------------------
# 1. Генерация синтетических данных
# -----------------------------------------------------------------------------
# np.random.seed(42)  # "зерно" генератора случайных чисел в библиотеке
#
# roads = [
#     "Окт", "Клнг", "Моск", "Горьк", "Сев", "С-Кав", "Ю-Вост", "Прив", "Кбш",
#     "Сверд", "Ю-Ур", "З-Сиб", "Крас", "В-Сиб", "Заб", "Двост", "Сеть"
# ]
#
# dates = pd.date_range("2022-01-01", periods=48, freq="MS")  # последовательность календарных дат (MS = начало месяца)


# генерация аддитивного временного рядя длины length из трёх систематических составляющих — уровень, линейный тренд, сезонность — плюс шум (гауссов), и редкие выбросы (“шпильки”).
# base — базовый уровень ряда (средний уровень без тренда/сезонности). Единицы такие же, как у целевого показателя (например, поездо-час/1 млн поездо-км).
# trend — приращение за один шаг времени. Если данные месячные, то это изменение в единицах за месяц. Знак «−» даёт нисходящий тренд.
# seasonal_amp — амплитуда сезонных колебаний (аддитивная). При 10 сезонная волна колеблется примерно ±10 вокруг трендовой линии.
# noise_scale — стандартное отклонение белого шума N(0, noise_scale^2).
# length — количество точек (по умолчанию 48 месяцев = 4 года).
# def generate_series(base, trend, seasonal_amp, noise_scale, length=48):
#     t = np.arange(length)     # дискретное время в шагах (месяцах).
#     seasonal = seasonal_amp * np.sin(2 * np.pi * t / 12)    # используется синус с периодом 12 (месяцев) — классическая гладкая сезонность.
#     series = base + trend * t + seasonal + np.random.normal(scale=noise_scale, size=length)     # базовая аддитивная структура + шум
#     spikes = np.random.choice([0, 0, 0, 50, -50], size=length, p=[0.8, 0.1, 0.05, 0.03, 0.02])  # около 5% точек — выбросы
#     return series + spikes

# словарь параметров params, где ключ — название дороги
# "Дорога": (base, trend, seasonal_amp, noise_scale)
# params = {
#     "Окт": (180, -0.05, 10, 20),
#     "Клнг": (20, 0.02, 5, 3),
#     "Моск": (150, -0.1, 20, 30),
#     "Горьк": (120, -0.02, 8, 10),
#     "Сев": (180, -0.05, 12, 15),
#     "С-Кав": (95, 0.01, 15, 8),
#     "Ю-Вост": (70, -0.01, 6, 6),
#     "Прив": (560, 0.5, 30, 40),
#     "Кбш": (185, -0.2, 18, 20),
#     "Сверд": (200, -0.05, 16, 18),
#     "Ю-Ур": (186, -0.08, 14, 12),
#     "З-Сиб": (138, -0.1, 10, 12),
#     "Крас": (172, -0.01, 8, 8),
#     "В-Сиб": (230, 0.02, 12, 20),
#     "Заб": (514, -0.3, 40, 60),
#     "Двост": (501, -0.02, 35, 50),
#     "Сеть": (229, -0.05, 20, 25),
# }

roads = ROADS
dates = DATES
params = PARAMS

# cоздание синтетического набора данных
# cоздаётся пустой DataFrame data с индексом dates (48 месяцев 2022–2025 гг.);
# для каждой дороги r:
# из словаря берутся её параметры (base, trend, amp, noise);
# Функция generate_series() создаёт ряд длиной 48 месяцев;
# ряд добавляется в таблицу data как новый столбец.
data = pd.DataFrame(index=dates)
for r in roads:
    base, trend, amp, noise = params[r]
    data[r] = generate_series(base, trend, amp, noise, len(dates))

# -----------------------------------------------------------------------------
# 2. Подготовка и создание папки результатов
# -----------------------------------------------------------------------------
train, test = data.iloc[:36], data.iloc[36:] # разделяет временной ряд на обучающую и тестовую выборки
outdir = Path("ets_results")
outdir.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Моделирование ETS(A,A,A)
# -----------------------------------------------------------------------------
forecasts = []  # таблицы с прогнозами по каждой дороге;
metrics = []    # метрики точности и статистику модели

for r in roads:

    y_train, y_test = train[r], test[r]     # для каждой дороги выбирается: y_train — обучающая часть временного ряда (2022–2024); y_test — тестовая часть (2025)

    # --- Обучение на train ---
    model = ExponentialSmoothing(                                       # создаётся объект модели ETS(A,A,A)
        y_train, trend="add", seasonal="add",                           # аддитивный тренд, аддитивная сезонность
        seasonal_periods=12, initialization_method="estimated"          # длина сезонного цикла (12 месяцев); начальные значения уровня, тренда и сезонности оцениваются автоматически
    )
    fit = model.fit(optimized=True)     # подгонка модели (fit) на обучающей части
    pred_test = fit.forecast(12)        # cтроится прогноз на 12 месяцев вперёд

    # --- MAPE ---
    mape = (np.abs((y_test - pred_test) / np.where(y_test == 0, np.nan, y_test))).mean() * 100  # Расчёт метрики точности (MAPE) (Mean Absolute Percentage Error)

    # --- Финальная модель и прогноз ---
    model_full = ExponentialSmoothing(          # модель перестраивается на всех доступных данных (2022–2025), чтобы прогнозировать новый, ещё не наблюдавшийся год — 2026.
        data[r], trend="add", seasonal="add",
        seasonal_periods=12, initialization_method="estimated"
    )
    fit_full = model_full.fit(optimized=True)
    pred_full = fit_full.forecast(12)                                   # прогноз на 12 будущих месяцев;
    future_idx = pd.date_range("2026-01-01", periods=12, freq="MS")     # список дат для этих прогнозов: с января по декабрь 2026 года

    # --- Доверительные интервалы ---
    resid_std = np.std(fit_full.resid)          # доверительный интервал (95%) для каждого прогнозного значения
    lower = pred_full - 1.96 * resid_std
    upper = pred_full + 1.96 * resid_std

    df_pred = pd.DataFrame({        # Каждая таблица df_pred содержит прогноз для одной дороги
        "road": r,
        "date": future_idx,
        "forecast": pred_full.values,
        "lower_95": lower.values,
        "upper_95": upper.values
    })
    forecasts.append(df_pred)

    metrics.append({
        "road": r,                                                   # Название дороги
        "MAPE_test_%": float(mape),                                  # Ошибка на тестовых данных
        "AIC": getattr(fit_full, "aic", np.nan),                     # Информационный критерий Акаике — чем меньше, тем лучше модель
        "resid_std": resid_std                                       # Стандартное отклонение остатков (σ), показатель шума
    })

    # --- График ---
    plt.figure(figsize=(8, 4))
    plt.plot(data.index, data[r], label="наблюдения")
    plt.plot(future_idx, pred_full, "--", label="прогноз")
    plt.fill_between(future_idx, lower, upper, color="orange", alpha=0.3, label="95% ДИ")
    plt.axvline(data.index[-1], color="gray", linestyle=":")
    plt.title(f"ETS(A,A,A) — {r} (прогноз на 2026)")
    plt.xlabel("Дата")
    plt.ylabel("Значение")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"forecast_{r}.png")
    plt.close()

# -----------------------------------------------------------------------------s
# 4. Сводные таблицы
# -----------------------------------------------------------------------------
forecasts_df = pd.concat(forecasts)     # Объединяет все прогнозы по дорогам в один общий
forecasts_pivot = forecasts_df.pivot(index="date", columns="road", values="forecast")
metrics_df = pd.DataFrame(metrics).set_index("road")

# -----------------------------------------------------------------------------
# 5. Сохранение
# -----------------------------------------------------------------------------
forecasts_df.to_csv(outdir / "forecasts_2026_with_CI.csv", index=False)
metrics_df.to_csv(outdir / "ets_metrics.csv", index=True)   # анализ точности и стабильности моделей

print("✅ ETS(A,A,A) прогнозирование завершено!")
print(f"Файлы сохранены в: {outdir.resolve()}")
print("- forecasts_2026_with_CI.csv — прогнозы с доверительными интервалами")
print("- ets_metrics.csv — метрики точности (MAPE, AIC, σ остатков)")
print("- forecast_<road>.png — графики с 95% доверительными интервалами")