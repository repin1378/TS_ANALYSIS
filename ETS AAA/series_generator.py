import numpy as np

def generate_series(base: float, trend: float, seasonal_amp: float, noise_scale: float, length: int = 48, num_forced_outliers: int = 0) -> np.ndarray:

    # генерация аддитивного временного рядя длины length из трёх систематических составляющих — уровень, линейный тренд, сезонность — плюс шум (гауссов), и редкие выбросы (“шпильки”).
    # base — базовый уровень ряда (средний уровень без тренда/сезонности). Единицы такие же, как у целевого показателя (например, поездо-час/1 млн поездо-км).
    # trend — приращение за один шаг времени. Если данные месячные, то это изменение в единицах за месяц. Знак «−» даёт нисходящий тренд.
    # seasonal_amp — амплитуда сезонных колебаний (аддитивная). При 10 сезонная волна колеблется примерно ±10 вокруг трендовой линии.
    # noise_scale — стандартное отклонение белого шума N(0, noise_scale^2).
    # length — количество точек (по умолчанию 48 месяцев = 4 года).

    t = np.arange(length)   # дискретное время в шагах (месяцах).

    # Сезонная компонента
    seasonal = seasonal_amp * np.sin(2 * np.pi * t / 12)    # используется синус с периодом 12 (месяцев) — классическая гладкая сезонность

    # Трендовая составляющая
    base_trend = base + trend * t

    # Случайный шум
    noise = np.random.normal(scale=noise_scale, size=length)

    # Случайные выбросы по старой логике
    spikes = np.random.choice(
        [0, 0, 0, 50, -50],
        size=length,
        p=[0.8, 0.1, 0.05, 0.03, 0.02]
    )

    series = base_trend + seasonal + noise + spikes

    #  Принудительные выбросы (если нужно)

    if num_forced_outliers > 0:
        # Берём случайные индексы без повторений
        positions = np.random.choice(length, size=num_forced_outliers, replace=False)

        # Амплитуды выбросов: ±(60…120)
        amplitudes = np.random.choice([80, 100, 120, -80, -100, -120],
                                      size=num_forced_outliers)

        for pos, amp in zip(positions, amplitudes):
            series[pos] += amp

    return series