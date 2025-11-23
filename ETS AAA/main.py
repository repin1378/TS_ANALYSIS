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
from ets_model import run_ets_forecast
from ets_report import generate_ets_pdf_report


# Вызов generation_config.py для генерации временных рядов
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

# --------------------------------
# Вызов вынесенной ETS-модели
# --------------------------------
forecasts_df, metrics_df = run_ets_forecast(data, roads)

# Генерация PDF
generate_ets_pdf_report(
    forecasts_df,
    metrics_df,
    charts_dir="ets_results",
    outfile="ets_results/ets_report.pdf"
)