import numpy as np
import pandas as pd
from datetime import timedelta

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
