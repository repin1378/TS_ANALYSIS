import numpy as np
import pandas as pd
from datetime import timedelta
import optuna

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




