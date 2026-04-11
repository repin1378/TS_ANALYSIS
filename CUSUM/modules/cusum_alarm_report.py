# modules/cusum_alarm_report.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. УТИЛИТЫ
# ============================================================

_ROAD_INDICATOR_COLS = {
    "CATEGORY", "REASON", "DEPARTMENT", "RESPONSIBILITY",
    "FREIGHT_COUNT", "FREIGHT_TIMEOUT",
    "PASSENGER_COUNT", "PASSENGER_TIMEOUT",
    "COMMUTER_COUNT", "COMMUTER_TIMEOUT",
}


def _is_road_csv(df: pd.DataFrame) -> bool:
    """True если DataFrame содержит хотя бы одно дорожное поле."""
    return bool(_ROAD_INDICATOR_COLS & set(df.columns))


def find_prev_year_csv(csv_path: Path) -> Optional[Path]:
    """
    Автоопределение CSV за предыдущий год.

    Ищет первое вхождение четырёхзначного года (20xx) в имени файла
    и заменяет на year-1. Если файл не существует — возвращает None.

    Пример:
        synthetic_smooth_Октябрьская_2_2025.csv → ..._2024.csv
        filtered_year-2024_road-Октябрьская_category-1.csv → ...-2023_...
        CT_2024.csv → CT_2023.csv
    """
    name = csv_path.stem
    match = re.search(r"(20\d{2})", name)
    if not match:
        return None
    year = int(match.group(1))
    prev_name = name.replace(str(year), str(year - 1), 1) + csv_path.suffix
    prev_path = csv_path.parent / prev_name
    return prev_path if prev_path.exists() else None


def _to_analogous_date(alarm_date: pd.Timestamp, year: int) -> pd.Timestamp:
    """
    Переносит дату alarm_date на указанный год (та же дата — месяц и день).
    Обрабатывает 29 февраля: если год невисокосный → 28 февраля.
    """
    try:
        return alarm_date.replace(year=year)
    except ValueError:
        return alarm_date.replace(year=year, day=28)


def _find_season_start(
    df: pd.DataFrame,
    season: str,
    up_to: pd.Timestamp,
    date_col: str,
    season_col: str,
) -> pd.Timestamp:
    """
    Начало сезона season в df: первая дата season_col == season, не позже up_to.
    Если строк не найдено — возвращает up_to (fallback).
    """
    mask = (df[season_col] == season) & (df[date_col] <= up_to)
    subset = df.loc[mask, date_col]
    return subset.min() if not subset.empty else up_to


def _slice_season(
    df: pd.DataFrame,
    season: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_col: str,
    season_col: str,
) -> pd.DataFrame:
    """
    Срез df: только строки с season_col == season и date_col ∈ [start, end].
    """
    mask = (
        (df[season_col] == season)
        & (df[date_col] >= start)
        & (df[date_col] <= end)
    )
    return df.loc[mask].copy()


def _pct_change(curr: float, prev: float) -> Optional[float]:
    """(curr − prev) / prev × 100. None если prev == 0."""
    if prev == 0:
        return None
    return round((curr - prev) / prev * 100, 2)


# ============================================================
# 2. СТАТИСТИКА ПО ГРУППАМ
# ============================================================

def _group_comparison(
    curr_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    col: str,
) -> list[dict]:
    """
    Сравнение количества событий по значениям колонки col.

    Возвращает список словарей, отсортированный по абсолютному |pct_change|:
        value       — значение поля
        curr_count  — количество событий в текущем периоде
        prev_count  — количество событий в предыдущем периоде
        pct_change  — процентное изменение (None если prev_count == 0)
    """
    col_in_curr = col in curr_df.columns
    col_in_prev = col in prev_df.columns

    if not col_in_curr and not col_in_prev:
        return []

    curr_counts = curr_df[col].value_counts() if col_in_curr else pd.Series(dtype=int)
    prev_counts = prev_df[col].value_counts() if col_in_prev else pd.Series(dtype=int)

    all_values = set(curr_counts.index) | set(prev_counts.index)

    rows = []
    for v in sorted(all_values, key=str):
        curr_n = int(curr_counts.get(v, 0))
        prev_n = int(prev_counts.get(v, 0))
        rows.append({
            "value": str(v),
            "curr_count": curr_n,
            "prev_count": prev_n,
            "pct_change": _pct_change(curr_n, prev_n),
        })

    return sorted(rows, key=lambda r: abs(r["pct_change"] or 0), reverse=True)


def _downtime_sum(df: pd.DataFrame, count_col: str, timeout_col: str) -> float:
    """
    Суммарное произведение COUNT × TIMEOUT.
    Пропускает строки с NaN или нечисловыми значениями.
    """
    if count_col not in df.columns or timeout_col not in df.columns:
        return 0.0
    cnt = pd.to_numeric(df[count_col], errors="coerce")
    tmt = pd.to_numeric(df[timeout_col], errors="coerce")
    return float((cnt * tmt).sum(skipna=True))


def _downtime_comparison(
    curr_df: pd.DataFrame,
    prev_df: pd.DataFrame,
) -> dict:
    """
    Сравнение суммарных простоев по трём видам движения.

    Возвращает словарь вида:
        {
            "freight":   {"curr": ..., "prev": ..., "pct_change": ...},
            "passenger": {...},
            "commuter":  {...},
        }
    """
    pairs = [
        ("freight",   "FREIGHT_COUNT",   "FREIGHT_TIMEOUT"),
        ("passenger", "PASSENGER_COUNT", "PASSENGER_TIMEOUT"),
        ("commuter",  "COMMUTER_COUNT",  "COMMUTER_TIMEOUT"),
    ]
    result = {}
    for name, count_col, timeout_col in pairs:
        curr_val = _downtime_sum(curr_df, count_col, timeout_col)
        prev_val = _downtime_sum(prev_df, count_col, timeout_col)
        result[name] = {
            "curr": round(curr_val, 2),
            "prev": round(prev_val, 2),
            "pct_change": _pct_change(curr_val, prev_val),
        }
    return result


# ============================================================
# 3. DATACLASS ОТЧЁТА
# ============================================================

@dataclass
class AlarmReport:
    """
    Структура отчёта для одного срабатывания CUSUM.

    Поля сравнения заполняются только если CSV содержит дорожные данные
    (is_road_csv == True).
    """

    # --- Мета ---
    alarm_time: str
    alarm_season: str
    current_year: int
    previous_year: int

    # --- Периоды сравнения ---
    season_start_current: str
    alarm_date_current: str
    season_start_previous: str
    analogous_date_previous: str

    # --- Общий счётчик событий ---
    n_events_current: int
    n_events_previous: int
    total_pct_change: Optional[float]

    # --- Тип данных ---
    is_road_csv: bool

    # --- Сравнение по разрезам (только для дорожных CSV) ---
    by_category: list[dict] = field(default_factory=list)
    by_reason: list[dict] = field(default_factory=list)
    by_department: list[dict] = field(default_factory=list)
    by_responsibility: list[dict] = field(default_factory=list)

    # --- Простои (только для дорожных CSV) ---
    downtime: dict = field(default_factory=dict)


# ============================================================
# 4. ГЕНЕРАЦИЯ ОТЧЁТА ДЛЯ ОДНОГО АЛАРМА
# ============================================================

def generate_alarm_report(
    alarm_time: pd.Timestamp,
    alarm_season: str,
    curr_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    out_dir: Path,
    date_col: str = "START_TIME",
    season_col: str = "SEASON",
    report_name: Optional[str] = None,
) -> Path:
    """
    Формирует JSON-отчёт для одного срабатывания CUSUM.

    Логика:
        1. Определяем начало текущего сезона в curr_df (season_col == alarm_season,
           дата не позже alarm_time).
        2. Срезаем curr_df от начала сезона до alarm_time.
        3. В предыдущем году находим аналогичную дату (тот же месяц/день, year-1).
        4. Находим начало того же сезона в prev_df.
        5. Срезаем prev_df от начала сезона до аналогичной даты.
        6. Считаем % изменения по всем разрезам.

    Параметры:
        alarm_time   — метка времени срабатывания CUSUM
        alarm_season — значение сезона из events CSV (строка)
        curr_df      — полный DataFrame текущего года
        prev_df      — полный DataFrame предыдущего года
        out_dir      — папка для сохранения отчёта
        date_col     — название колонки с датой/временем
        season_col   — название колонки с сезоном
        report_name  — базовое имя JSON-файла (без расширения);
                       если None — генерируется из alarm_time

    Возвращает:
        Path к сохранённому JSON-файлу.
    """

    curr_df = curr_df.copy()
    prev_df = prev_df.copy()
    curr_df[date_col] = pd.to_datetime(curr_df[date_col], errors="coerce")
    prev_df[date_col] = pd.to_datetime(prev_df[date_col], errors="coerce")

    curr_year = int(alarm_time.year)
    prev_year = curr_year - 1

    # --- Срез текущего года: от начала сезона до alarm_time ---
    season_start_curr = _find_season_start(curr_df, alarm_season, alarm_time, date_col, season_col)
    curr_slice = _slice_season(curr_df, alarm_season, season_start_curr, alarm_time, date_col, season_col)

    # --- Аналогичная дата и срез предыдущего года ---
    analogous_date = _to_analogous_date(alarm_time, prev_year)
    season_start_prev = _find_season_start(prev_df, alarm_season, analogous_date, date_col, season_col)
    prev_slice = _slice_season(prev_df, alarm_season, season_start_prev, analogous_date, date_col, season_col)

    n_curr = len(curr_slice)
    n_prev = len(prev_slice)
    is_road = _is_road_csv(curr_df) or _is_road_csv(prev_df)

    report = AlarmReport(
        alarm_time=str(alarm_time),
        alarm_season=str(alarm_season),
        current_year=curr_year,
        previous_year=prev_year,
        season_start_current=str(season_start_curr),
        alarm_date_current=str(alarm_time),
        season_start_previous=str(season_start_prev),
        analogous_date_previous=str(analogous_date),
        n_events_current=n_curr,
        n_events_previous=n_prev,
        total_pct_change=_pct_change(n_curr, n_prev),
        is_road_csv=is_road,
    )

    if is_road:
        report.by_category = _group_comparison(curr_slice, prev_slice, "CATEGORY")
        report.by_reason = _group_comparison(curr_slice, prev_slice, "REASON")
        report.by_department = _group_comparison(curr_slice, prev_slice, "DEPARTMENT")
        report.by_responsibility = _group_comparison(curr_slice, prev_slice, "RESPONSIBILITY")
        report.downtime = _downtime_comparison(curr_slice, prev_slice)

    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str = alarm_time.strftime("%Y%m%d_%H%M%S")
    fname = report_name or f"alarm_report_{ts_str}"
    json_path = out_dir / f"{fname}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=4, ensure_ascii=False)

    return json_path


# ============================================================
# 5. ГЕНЕРАЦИЯ ОТЧЁТОВ ДЛЯ ВСЕХ АЛАРМОВ
# ============================================================

def generate_all_alarm_reports(
    events_csv: Path,
    curr_csv_path: Path,
    out_dir: Path,
    prev_csv_path: Optional[Path] = None,
    date_col: str = "START_TIME",
    season_col: str = "SEASON",
) -> list[Path]:
    """
    Формирует JSON-отчёт для каждого срабатывания в events CSV.

    Параметры:
        events_csv    — путь к *_cusum_events.csv (выход run_cusum_exp_from_csv)
        curr_csv_path — исходный CSV текущего года (полный, с SEASON и т.д.)
        out_dir       — папка для сохранения отчётов
        prev_csv_path — CSV предыдущего года; если None — автоопределение
                        по имени файла (замена года на year-1)
        date_col      — колонка с датой в основном CSV
        season_col    — колонка с сезоном в основном CSV

    Возвращает:
        Список путей к сохранённым JSON-отчётам.
        Дополнительно сохраняет сводный CSV summary.csv в out_dir.
    """

    events_df = pd.read_csv(events_csv, encoding="utf-8-sig")

    if events_df.empty:
        print("[REPORT] Нет срабатываний в events CSV.")
        return []

    curr_df = pd.read_csv(curr_csv_path, encoding="utf-8-sig")

    if prev_csv_path is None:
        prev_csv_path = find_prev_year_csv(curr_csv_path)

    if prev_csv_path is None or not prev_csv_path.exists():
        print(f"[REPORT] CSV предыдущего года не найден. Отчёты не сформированы.")
        return []

    prev_df = pd.read_csv(prev_csv_path, encoding="utf-8-sig")

    report_paths: list[Path] = []
    summary_rows: list[dict] = []
    stem = curr_csv_path.stem

    for _, row in events_df.iterrows():
        alarm_time = pd.to_datetime(row[date_col])
        alarm_season = str(row[season_col])
        ts_str = alarm_time.strftime("%Y%m%d_%H%M%S")
        report_name = f"{stem}_alarm_{ts_str}"

        json_path = generate_alarm_report(
            alarm_time=alarm_time,
            alarm_season=alarm_season,
            curr_df=curr_df,
            prev_df=prev_df,
            out_dir=out_dir,
            date_col=date_col,
            season_col=season_col,
            report_name=report_name,
        )

        print(f"[REPORT] {json_path.name}")
        report_paths.append(json_path)

        # --- Строка для сводного CSV ---
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        summary_row: dict = {
            "alarm_time": data["alarm_time"],
            "alarm_season": data["alarm_season"],
            "current_year": data["current_year"],
            "previous_year": data["previous_year"],
            "season_start_current": data["season_start_current"],
            "analogous_date_previous": data["analogous_date_previous"],
            "n_events_current": data["n_events_current"],
            "n_events_previous": data["n_events_previous"],
            "total_pct_change": data["total_pct_change"],
        }

        # Добавляем итоги простоев если есть
        for transport in ("freight", "passenger", "commuter"):
            if transport in data.get("downtime", {}):
                summary_row[f"{transport}_curr"] = data["downtime"][transport]["curr"]
                summary_row[f"{transport}_prev"] = data["downtime"][transport]["prev"]
                summary_row[f"{transport}_pct_change"] = data["downtime"][transport]["pct_change"]

        summary_rows.append(summary_row)

    if summary_rows:
        summary_path = out_dir / f"{stem}_alarm_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[REPORT] Сводный CSV: {summary_path}")

    print(f"[REPORT] Сформировано отчётов: {len(report_paths)}")
    return report_paths


# ============================================================
# 6. ОБЁРТКА: CUSUM + ОТЧЁТЫ В ОДИН ВЫЗОВ
# ============================================================

def run_cusum_with_reports(
    csv_path: Path,
    *,
    delta_target: float,
    window_size: int,
    h_json_dir: Path,
    out_dir: Path,
    prev_csv_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    arl0_target: Optional[float] = None,
    arl1_target: Optional[float] = None,
    cooldown_after_alarm: Optional[int] = None,
    verbose: bool = True,
    progress_every: int = 2000,
) -> tuple[Path, Path, list[Path]]:
    """
    Запускает CUSUM и формирует аналитические отчёты для каждого срабатывания.

    Параметры:
        csv_path           — CSV текущего года (вход для CUSUM)
        delta_target       — целевое δ = λ1/λ0
        window_size        — размер скользящего окна для оценки λ
        h_json_dir         — папка с JSON-справочниками порогов h
        out_dir            — папка для результатов CUSUM
        prev_csv_path      — CSV предыдущего года; None → автоопределение
        reports_dir        — папка для JSON/CSV отчётов;
                             None → out_dir / "alarm_reports"
        arl0_target        — целевой ARL0 для выбора h
        arl1_target        — целевой ARL1 для выбора h
        cooldown_after_alarm — cooldown шагов после тревоги
        verbose            — подробный вывод CUSUM
        progress_every     — шаг прогресс-вывода CUSUM

    Возвращает:
        (events_path, full_path, report_paths)
        events_path  — *_cusum_events.csv
        full_path    — *_cusum_full.csv
        report_paths — список JSON-отчётов по алармам
    """
    from modules.cusum_exp_seasonal import run_cusum_exp_from_csv

    events_path, full_path = run_cusum_exp_from_csv(
        csv_path,
        delta_target=delta_target,
        window_size=window_size,
        h_json_dir=h_json_dir,
        out_dir=out_dir,
        arl0_target=arl0_target,
        arl1_target=arl1_target,
        cooldown_after_alarm=cooldown_after_alarm,
        verbose=verbose,
        progress_every=progress_every,
    )

    rpt_dir = reports_dir or (out_dir / "alarm_reports")

    report_paths = generate_all_alarm_reports(
        events_csv=events_path,
        curr_csv_path=csv_path,
        out_dir=rpt_dir,
        prev_csv_path=prev_csv_path,
    )

    return events_path, full_path, report_paths
