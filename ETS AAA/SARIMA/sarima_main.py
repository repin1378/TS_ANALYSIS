# SARIMA/sarima_main.py

"""
Главный модуль SARIMA.
Здесь демонстрируется:
- генерация исходного временного ряда
- вызов шага 1: проверка входных данных
"""

import numpy as np
# Импорт шага 1 SARIMA
from SARIMA.check_input import assert_valid_series
# Импорт генерации данных
from series_generator import generate_series
from generation_config import PARAMS, ROADS
# Обработка выбросов
from SARIMA.outliers import process_outliers
# Нормализация масштаба
from SARIMA.scale import scale_series



def sarima_test_run():

    """
    Тестовая функция SARIMA:
    1. Берёт параметры первой дороги
    2. Генерирует временной ряд длины 72
    3. Выполняет шаг 1 (проверку входного ряда)
    4. Обработка выбросов
    5. Нормализация масштаба
    """

    road = ROADS[0]                                                 # Например, "Окт"
    base, trend, amp, noise = PARAMS[road]

    print(f"Генерируем ряд для дороги: {road}")

    # SARIMA требует ровно 72 точки
    x = generate_series( base, trend, amp, noise, length=72,
        num_forced_outliers=5  # гарантировано будут выбросы
    )

    print("Первые 5 значений ряда:", x[:5])

    # Шаг 1 — проверка входных данных
    assert_valid_series(x)

    print("\nПроверка пройдена успешно!")
    print("Можно переходить к шагу 2 (обработка выбросов).")

    clean_series, outlier_flags, num_outliers = process_outliers(x)

    print("Найдено выбросов:", num_outliers)
    print("Флаги:", outlier_flags[:20])
    print("После очистки:", clean_series[:10])

    clean_series, flags, num_outliers = process_outliers(x)
    scaled_series, mean_value = scale_series(clean_series)

    print("Среднее значение u1:", mean_value)
    print("Первые 5 значений после нормализации:", scaled_series[:5])

if __name__ == "__main__":
    sarima_test_run()
