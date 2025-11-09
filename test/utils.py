import numpy as np
import pandas as pd
from datetime import timedelta
import optuna
from collections import deque

def generate_segmented_exponential_dataset(df_result, segments, seed=None):
    """
    Генерирует единый DataFrame с экспоненциальными выборками, составленными из нескольких сегментов.

    Параметры:
        df_result (pd.DataFrame): таблица с колонкой 'lambda_est' и 'dataframe_index'
        segments (list of tuples): список (length, lambda_index), где:
            - length: количество точек для генерации
            - lambda_index: индекс строки в df_result (откуда брать lambda_est)
        seed (int or None): для воспроизводимости

    Возвращает:
        pd.DataFrame с колонками: TIME_DIFF, global_index, source_index, lambda_est
    """
    if seed is not None:
        np.random.seed(seed)

    if df_result.empty or 'lambda_est' not in df_result.columns:
        raise ValueError("df_result должен содержать колонку 'lambda_est'.")

    rows = []
    global_idx = 0

    for length, lambda_idx in segments:
        if lambda_idx >= len(df_result):
            raise IndexError(f"lambda_index {lambda_idx} выходит за границы df_result.")

        lam = df_result.iloc[lambda_idx]['lambda_est']
        source = df_result.iloc[lambda_idx]['dataframe_index']
        scale = 1 / lam

        data = np.random.exponential(scale=scale, size=length)

        for val in data:
            rows.append({
                'TIME_DIFF': val,
                'global_index': global_idx,
                'source_index': source,
                'lambda_est': lam
            })
            global_idx += 1

    return pd.DataFrame(rows)


def add_start_times(df_gen, start_time_0):
    """
    Добавляет колонку START_TIME на основе TIME_DIFF (в минутах) и начального времени.

    Параметры:
        df_gen (pd.DataFrame): таблица с колонкой 'TIME_DIFF'
        start_time_0 (str or pd.Timestamp): начальное время (например, '2023-01-01 00:00:00')

    Возвращает:
        df_gen с новой колонкой 'START_TIME'
    """
    if not isinstance(start_time_0, pd.Timestamp):
        start_time_0 = pd.to_datetime(start_time_0)

    time_deltas = pd.to_timedelta(df_gen['TIME_DIFF'].cumsum(), unit='m')
    df_gen['START_TIME'] = start_time_0 + time_deltas
    return df_gen


def apply_cusum(df, target_lambda, k, h):
    """
    Применяет односторонний CUSUM (только верхняя граница) к TIME_DIFF.

    :param df: DataFrame с колонкой TIME_DIFF
    :param target_lambda: базовая интенсивность (λ)
    :param k: чувствительность
    :param h: порог отклонения
    :return: DataFrame с флагом 'CUSUM_signal'
    """
    mu0 = 1 / target_lambda  # ожидаемое среднее значение TIME_DIFF
    cusum_neg = [0]
    signals = [False]

    for i in range(1, len(df)):
        x = df['TIME_DIFF'].iloc[i]
        s_neg = min(0, cusum_neg[-1] + (x - mu0 + k))  # обратный знак
        signal = abs(s_neg) > h
        cusum_neg.append(s_neg)
        signals.append(signal)

    df_cusum = df.copy()
    df_cusum['CUSUM_signal'] = signals
    return df_cusum


def count_matches(df_cusum, true_points, tolerance=5):
    """Считает совпадения сигналов с истинными точками разрыва."""
    signal_indices = df_cusum.index[df_cusum['CUSUM_signal']].tolist()
    matches = sum(
        any(abs(signal - true_point) <= tolerance for signal in signal_indices)
        for true_point in true_points
    )
    return matches

def make_objective(
    df_gen,
    lambda_0,
    true_change_points,
    k_range=(1000, 4000),
    h_range=(2, 10),
    tolerance=5
):
    def objective(trial):
        k = trial.suggest_float("k", k_range[0], k_range[1])
        h = trial.suggest_float("h", h_range[0], h_range[1])
        df_cusum = apply_cusum(df_gen, lambda_0, k, h)
        matches = count_matches(df_cusum, true_change_points, tolerance=tolerance)
        return -matches
    return objective


def detect_cusum_changes(df_gen, lambda_0, k, h, min_gap=30):
    """
    Применяет CUSUM и возвращает только уникальные расхождения (по группам сигналов).

    :param df_gen: DataFrame с колонками ['TIME_DIFF', 'START_TIME']
    :param lambda_0: базовая интенсивность
    :param k: чувствительность
    :param h: порог
    :param min_gap: минимальное расстояние (в индексах) между сигналами
    :return: DataFrame с найденными точками изменений
    """
    df_cusum = apply_cusum(df_gen, lambda_0, k, h)

    # Сигналы, которые CUSUM отметил
    signal_df = df_cusum[df_cusum['CUSUM_signal']].copy()
    signal_df = signal_df.reset_index().rename(columns={'index': 'event_index'})

    # Группировка сигналов: оставляем только первые в серии
    grouped_alerts = []
    last_index = -min_gap * 2  # инициализация заведомо вне диапазона
    for _, row in signal_df.iterrows():
        if row['event_index'] - last_index >= min_gap:
            grouped_alerts.append(row)
            last_index = row['event_index']

    # Возвращаем сгруппированный DataFrame
    return pd.DataFrame(grouped_alerts)[['event_index', 'START_TIME', 'TIME_DIFF']]

# ======================= GLR-CUSUM для экспоненты + утилиты ============================
def estimate_lambda0_from_slice(df: pd.DataFrame,
                                start: int | None = None,
                                end: int | None = None,
                                time_col: str | None = None,
                                t0=None, t1=None,
                                col: str = "TIME_DIFF") -> float:
    """
    Оценка λ0 (MLE) по указанному «нормальному» промежутку.
    Можно дать индексный срез [start:end) или по времени [t0, t1] и колонке time_col.
    """
    if time_col and (t0 is not None or t1 is not None):
        m = df
        if t0 is not None: m = m[m[time_col] >= pd.to_datetime(t0)]
        if t1 is not None: m = m[m[time_col] <= pd.to_datetime(t1)]
        x = m[col].to_numpy()
    else:
        s = 0 if start is None else start
        e = len(df) if end is None else end
        x = df[col].iloc[s:e].to_numpy()

    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) < 5:
        raise ValueError("Слишком короткий нормальный интервал для оценки λ0.")
    mu0 = float(np.mean(x))
    return 1.0 / mu0


def glr_cusum_exp(df: pd.DataFrame,
                  col: str = "TIME_DIFF",
                  lambda0_init: float = 1.0,
                  W: int = 1000,
                  baseline_window: int = 500,
                  freeze_after: int = 500,
                  h: float = 6.0):
    """
    GLR-CUSUM для X~Exp(λ), λ0 неизвестно: поддерживаем λ0 по скользящему окну baseline_window.
    Статистика G_t = max_{t-W<=k<=t} LLR_{k:t}, где LLR_{k:t} = n log(n/(λ0 * sumX)) - n + λ0*sumX.
    При G_t >= h -> тревога; после тревоги замораживаем обновление λ0 на freeze_after шагов.

    Возвращает df с колонками: GLR_stat, CUSUM_signal, lambda0_used.
    """
    x = df[col].to_numpy()
    n = len(x)

    # префиксные суммы для быстрых LLR
    Sx = np.zeros(n + 1)
    for i in range(1, n + 1):
        Sx[i] = Sx[i - 1] + x[i - 1]

    lam0 = float(lambda0_init)
    buf = deque(maxlen=baseline_window)
    frozen = 0

    G = np.zeros(n)
    alarm = np.zeros(n, dtype=bool)
    lam_used = np.zeros(n)

    for t in range(n):
        xi = x[t]
        # обновление λ0 (если не заморожено)
        if frozen == 0 and xi > 0:
            buf.append(xi)
            if len(buf) == baseline_window:
                lam0 = 1.0 / (sum(buf) / len(buf))

        lam_used[t] = lam0

        t1 = t + 1
        k_min = max(1, t1 - W)
        best = 0.0
        # сканируем возможные точки начала на последнем окне W
        for k in range(k_min, t1 + 1):
            nseg = t1 - (k - 1)
            sumx = Sx[t1] - Sx[k - 1]
            if sumx <= 0:
                continue
            lam_hat = nseg / sumx
            llr = nseg * (np.log(lam_hat) - np.log(lam0)) - (lam_hat - lam0) * sumx
            if llr > best:
                best = llr
        G[t] = max(0.0, best)

        if G[t] >= h:
            alarm[t] = True
            frozen = freeze_after  # «заморозка» λ0
        else:
            if frozen > 0:
                frozen -= 1

    out = df.copy()
    out["GLR_stat"] = G
    out["CUSUM_signal"] = alarm
    out["lambda0_used"] = lam_used
    return out

def group_signals(df_cusum: pd.DataFrame, min_gap: int = 30):
    """
    Оставляет только первые сработки с шагом не чаще min_gap.
    Ожидает булев столбец 'CUSUM_signal' и индексы по времени/событиям.
    """
    sig = df_cusum[df_cusum["CUSUM_signal"]].reset_index().rename(columns={'index': 'event_index'})
    groups = []
    last = -10**9
    for _, r in sig.iterrows():
        if r["event_index"] - last >= min_gap:
            groups.append(r)
            last = r["event_index"]
    cols = ["event_index"]
    if "START_TIME" in df_cusum.columns:
        cols.append("START_TIME")
    if "TIME_DIFF" in df_cusum.columns:
        cols.append("TIME_DIFF")
    return pd.DataFrame(groups)[cols]


def autotune_h_glr(target_arl0: int,
                   lambda0: float,
                   W: int = 1000,
                   baseline_window: int = 500,
                   reps: int = 800,
                   n_max: int = 50_000,
                   tol: float = 0.15,
                   seed: int | None = 42) -> float:
    """
    Подбор порога h для GLR-CUSUM под целевую ARL0 быстрым MC и бинарным поиском.
    Возвращает h, у которого эмпирическая ARL0 в пределах ±tol от цели.

    Примечание: симулируем X~Exp(lambda0), обновляем λ0 как и в glr_cusum_exp (baseline_window),
    но без «заморозки» (fire редко при H0).
    """
    rng = np.random.default_rng(seed)

    def estimate_arl0(h_val: float) -> float:
        arls = []
        for _ in range(reps):
            # генерим поток под H0 (λ = lambda0)
            x = rng.exponential(scale=1.0 / lambda0, size=n_max)
            # быстрая оценка ARL0 на одной трассе
            # реализуем укороченный glr-процесс «на лету»
            Sx = 0.0
            buf = deque(maxlen=baseline_window)
            lam0_used = lambda0
            G = 0.0
            fired_at = n_max
            # для ускорения делаем инкрементальные префиксные суммы на окне W
            # храним последние W сумм для перебора k
            from collections import deque as dq
            last_x = dq(maxlen=W)
            pref = dq(maxlen=W + 1)
            pref.append(0.0)

            for t in range(n_max):
                xi = x[t]
                if xi > 0:
                    buf.append(xi)
                    if len(buf) == baseline_window:
                        lam0_used = 1.0 / (sum(buf) / len(buf))

                # обновляем окно и префиксные суммы
                last_x.append(xi)
                pref.append(pref[-1] + xi)
                # сканируем по отрезку окна
                best = 0.0
                m = len(last_x)
                for nseg in range(1, m + 1):
                    sumx = pref[-1] - pref[-1 - nseg]
                    if sumx <= 0:
                        continue
                    lam_hat = nseg / sumx
                    llr = nseg * (np.log(lam_hat) - np.log(lam0_used)) - (lam_hat - lam0_used) * sumx
                    if llr > best:
                        best = llr
                G = max(0.0, best)
                if G >= h_val:
                    fired_at = t + 1
                    break
            arls.append(fired_at)
        return float(np.mean(arls))

    # грубые границы поиска
    lo, hi = 1.0, 20.0
    # расширим hi, если ARL0 слишком маленькая
    while estimate_arl0(hi) < target_arl0 * (1 - tol):
        hi *= 1.5
        if hi > 200:
            break

    # бинарный поиск
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        arl = estimate_arl0(mid)
        if arl < target_arl0 * (1 - tol):
            lo = mid  # мало — увеличиваем порог
        elif arl > target_arl0 * (1 + tol):
            hi = mid  # много — уменьшаем порог
        else:
            return mid
    return 0.5 * (lo + hi)



