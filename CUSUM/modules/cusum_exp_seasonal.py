from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ============================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def estimate_lambda_window(time_diffs: np.ndarray) -> float:
    """
    MLE-оценка λ для Exp(λ):
        λ̂ = 1 / mean(x)
    """
    x = time_diffs[time_diffs > 0]
    if len(x) == 0:
        return np.nan
    return 1.0 / np.mean(x)


def llr_increment_exp(time_diff: float, lambda0: float, delta: float) -> float:
    """
    Приращение решающей функции CUSUM (Филаретов–Репин):
        z = ln(delta) - (delta - 1) * (lambda0 * x)
    """
    y = lambda0 * time_diff
    return np.log(delta) - (delta - 1.0) * y


def cusum_update(S_prev: float, z: float) -> float:
    """
    Рекурсия CUSUM:
        S_i = max(0, S_{i-1} + z_i)
    """
    return max(0.0, S_prev + z)


def _normalize_stem(stem: str) -> str:
    """
    Нормализует имя файла к формату «Название-Год»:
        • заменяет пробелы и символы подчёркивания на дефис
        • схлопывает несколько дефисов в один
        • убирает ведущие/замыкающие дефисы

    Примеры:
        Дальневосточная_2025  →  Дальневосточная-2025
        CSH 2025              →  CSH-2025
        МСК__2025             →  МСК-2025
    """
    normalized = re.sub(r"[ _]+", "-", stem)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


# ============================================================
# 2. ВЫБОР ПОРОГА h ИЗ JSON
# ============================================================

def select_h_from_json(
    *,
    delta_target: float,
    arl0_target: Optional[float],
    arl1_target: Optional[float],
    h_json_dir: Path,
    verbose: bool = True,
) -> float:
    """
    Выбор оптимального порога h по приоритетной (лексикографической) логике:

        1. Ближайший delta_target          (высший приоритет)
        2. Ближайший arl1_target           (если задан)
        3. Ближайший arl0_target           (низший приоритет)

    Файл JSON выбирается по комбинации заданных целей:
        • (δ, ARL0)        → h_from_delta_arl0.json
        • (δ, ARL1)        → h_from_delta_arl1.json
        • (δ, ARL0, ARL1)  → h_from_delta_arl0_arl1.json
    """

    _EPS = 1e-9  # допуск для сравнения вещественных чисел

    # ── выбор JSON ────────────────────────────────────────────────────────────
    if arl0_target is not None and arl1_target is not None:
        json_path = h_json_dir / "h_from_delta_arl0_arl1.json"
    elif arl0_target is not None:
        json_path = h_json_dir / "h_from_delta_arl0.json"
    elif arl1_target is not None:
        json_path = h_json_dir / "h_from_delta_arl1.json"
    else:
        raise ValueError("Нужно задать arl0_target и/или arl1_target")

    with open(json_path, "r", encoding="utf-8") as f:
        table = json.load(f)

    if not table:
        raise ValueError(f"JSON-файл пуст: {json_path}")

    candidates = list(table)

    # ── шаг 1: фильтр по delta_target (наивысший приоритет) ──────────────────
    best_delta_dist = min(abs(r["delta_target"] - delta_target) for r in candidates)
    candidates = [
        r for r in candidates
        if abs(r["delta_target"] - delta_target) <= best_delta_dist + _EPS
    ]

    # ── шаг 2: фильтр по arl1_target (второй приоритет) ─────────────────────
    if arl1_target is not None and "arl1_target" in candidates[0]:
        best_arl1_dist = min(abs(r["arl1_target"] - arl1_target) for r in candidates)
        candidates = [
            r for r in candidates
            if abs(r["arl1_target"] - arl1_target) <= best_arl1_dist + _EPS
        ]

    # ── шаг 3: фильтр по arl0_target (низший приоритет) ─────────────────────
    if arl0_target is not None and "arl0_target" in candidates[0]:
        best_arl0_dist = min(abs(r["arl0_target"] - arl0_target) for r in candidates)
        candidates = [
            r for r in candidates
            if abs(r["arl0_target"] - arl0_target) <= best_arl0_dist + _EPS
        ]

    best = candidates[0]

    if verbose:
        parts = [f"delta={best['delta_target']}"]
        if "arl1_target" in best:
            parts.append(f"arl1={best['arl1_target']}")
        if "arl0_target" in best:
            parts.append(f"arl0={best['arl0_target']}")
        print(
            f"[SELECT_H] JSON: {json_path.name} | "
            f"Выбрана запись: {', '.join(parts)} | "
            f"h_mc={best['h_mc']:.4f} "
            f"(из {len(table)} записей, осталось {len(candidates)} кандидатов)"
        )

    return float(best["h_mc"])


# ============================================================
# 3. ОСНОВНАЯ ФУНКЦИЯ CUSUM (один CSV)
# ============================================================

def run_cusum_exp_from_csv(
    csv_path: Path,
    *,
    delta_target: float,
    window_size: int,
    h_json_dir: Path,
    out_dir: Path,
    arl0_target: Optional[float] = None,
    arl1_target: Optional[float] = None,
    cooldown_after_alarm: Optional[int] = None,
    out_stem: Optional[str] = None,
    verbose: bool = True,
    progress_every: int = 2000,
) -> tuple[Path, Path]:
    """
    Экспоненциальный CUSUM (Филаретов–Репин) для одного CSV-файла.

    Параметры
    ---------
    csv_path            : путь к входному CSV
    delta_target        : целевой δ = λ₁/λ₀
    window_size         : размер скользящего окна для оценки λ̂
    h_json_dir          : папка с JSON-таблицами порогов
    out_dir             : папка для сохранения результатов
    arl0_target         : целевой ARL0 (ложные тревоги)
    arl1_target         : целевой ARL1 (задержка обнаружения)
    cooldown_after_alarm: шагов «молчания» после тревоги (по умолчанию = window_size)
    out_stem            : базовое имя выходных файлов без расширения
                          (по умолчанию — нормализованное имя csv_path)
    verbose             : печатать ли прогресс и тревоги
    progress_every      : шаг вывода прогресса (0 — отключить)

    Возвращает
    ----------
    (events_path, full_path)
        events_path : CSV с событиями тревоги
        full_path   : полный CSV с колонкой CUSUM_ALARM
    """

    # ========================================================
    # 1. Загрузка данных
    # ========================================================
    df = pd.read_csv(csv_path)
    df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
    df = df.sort_values("START_TIME").reset_index(drop=True)

    required_cols = {"TIME_DIFF", "SEASON", "LAMBDA0"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют колонки: {missing}")

    has_spike_flag = "SPIKE_FLAG" in df.columns

    # ── базовое имя выходных файлов ───────────────────────────────────────────
    stem = out_stem if out_stem else _normalize_stem(csv_path.stem)

    # ========================================================
    # 2. Порог h
    # ========================================================
    h = select_h_from_json(
        delta_target=delta_target,
        arl0_target=arl0_target,
        arl1_target=arl1_target,
        h_json_dir=h_json_dir,
        verbose=verbose,
    )

    cooldown = cooldown_after_alarm or window_size

    if verbose:
        print("\n================ CUSUM START ================")
        print(f"CSV:          {csv_path.name}")
        print(f"out_stem:     {stem}")
        print(f"delta_target: {delta_target}")
        print(f"window_size:  {window_size}")
        print(f"arl0_target:  {arl0_target}")
        print(f"arl1_target:  {arl1_target}")
        print(f"h:            {h:.4f}")
        print(f"cooldown:     {cooldown}")
        print(f"SPIKE_FLAG:   {has_spike_flag}")
        print("============================================\n")

    # ========================================================
    # 3. Инициализация
    # ========================================================
    S = 0.0
    last_season = None
    cooldown_steps = 0

    alarm_flag = np.zeros(len(df), dtype=int)
    events: list[dict] = []

    time_diff = df["TIME_DIFF"].to_numpy()

    # ========================================================
    # 4. Основной цикл
    # ========================================================
    for i in range(window_size, len(df)):

        season = df.loc[i, "SEASON"]
        lambda0 = df.loc[i, "LAMBDA0"]

        # --- прогресс ---
        if verbose and progress_every > 0 and i % progress_every == 0:
            print(
                f"[PROGRESS] i={i}, time={df.loc[i,'START_TIME']}, "
                f"S={S:.3f}, cooldown={cooldown_steps}"
            )

        # --- reset по сезону ---
        if last_season is None or season != last_season:
            if verbose and last_season is not None:
                print(f"[SEASON RESET] {last_season} → {season}")
            S = 0.0
            cooldown_steps = 0
            last_season = season

        # --- cooldown ---
        if cooldown_steps > 0:
            cooldown_steps -= 1
            continue

        # --- оценка λ̂ ---
        lambda_hat = estimate_lambda_window(time_diff[i - window_size:i])
        if not np.isfinite(lambda_hat) or lambda_hat <= lambda0:
            continue

        # --- CUSUM ---
        z = llr_increment_exp(
            time_diff=time_diff[i],
            lambda0=lambda0,
            delta=delta_target,
        )
        S = cusum_update(S, z)

        # --- тревога ---
        if S >= h:
            alarm_flag[i] = 1
            delta_hat = lambda_hat / lambda0

            in_spike = None
            if has_spike_flag:
                in_spike = int(df.loc[i, "SPIKE_FLAG"] == 1)

            events.append({
                "INDEX": i,
                "START_TIME": df.loc[i, "START_TIME"],
                "SEASON": season,
                "LAMBDA0": float(lambda0),
                "LAMBDA_HAT": float(lambda_hat),
                "DELTA_HAT": float(delta_hat),
                "CUSUM_S": float(S),
                "H": float(h),
                "COOLDOWN": int(cooldown),
                "IN_SPIKE_PERIOD": in_spike,
            })

            if verbose:
                print(
                    f"[ALARM] i={i}, time={df.loc[i,'START_TIME']}, "
                    f"delta_hat={delta_hat:.2f}, "
                    f"in_spike={in_spike}"
                )

            S = 0.0
            cooldown_steps = cooldown

    # ========================================================
    # 5. Сохранение результатов
    # ========================================================
    out_dir.mkdir(parents=True, exist_ok=True)

    df_events = pd.DataFrame(events)
    events_path = out_dir / f"{stem}_cusum_events.csv"
    df_events.to_csv(events_path, index=False, encoding="utf-8-sig")

    df_full = df.copy()
    df_full["CUSUM_ALARM"] = alarm_flag
    full_path = out_dir / f"{stem}.csv"
    df_full.to_csv(full_path, index=False, encoding="utf-8-sig")

    if verbose and has_spike_flag and len(df_events) > 0:
        hit_rate = df_events["IN_SPIKE_PERIOD"].mean()
        print(f"\nHit rate (CUSUM in SPIKE): {hit_rate:.2%}")

    if verbose:
        print("\n================ CUSUM END ==================")
        print(f"Total alarms: {int(alarm_flag.sum())}")
        print(f"Events CSV:   {events_path}")
        print(f"Full CSV:     {full_path}")
        print("============================================")

    return events_path, full_path


# ============================================================
# 4. БАТЧ-ФУНКЦИЯ CUSUM (папки дорог и департаментов)
# ============================================================

def run_cusum_batch(
    *,
    delta_target: float,
    window_size: int,
    h_json_dir: Path,
    roads_csv_dir: Optional[Path] = None,
    departments_csv_dir: Optional[Path] = None,
    roads_out_dir: Optional[Path] = None,
    departments_out_dir: Optional[Path] = None,
    arl0_target: Optional[float] = None,
    arl1_target: Optional[float] = None,
    cooldown_after_alarm: Optional[int] = None,
    summary_out_dir: Optional[Path] = None,
    verbose: bool = True,
    progress_every: int = 0,
) -> list[dict]:
    """
    Батч-запуск CUSUM по папкам с CSV-файлами дорог и/или департаментов.

    Структура входных папок
    -----------------------
        roads_csv_dir/       ← CSV по дорогам  (напр. Дальневосточная_2025.csv)
        departments_csv_dir/ ← CSV по департаментам (напр. CSH_2025.csv)

    Структура выходных папок
    ------------------------
        roads_out_dir/
            Дальневосточная-2025_cusum_events.csv
            Дальневосточная-2025_cusum_full.csv
        departments_out_dir/
            CSH-2025_cusum_events.csv
            CSH-2025_cusum_full.csv

    Параметры
    ---------
    roads_csv_dir       : папка с CSV по дорогам (None — пропустить)
    departments_csv_dir : папка с CSV по департаментам (None — пропустить)
    roads_out_dir       : папка результатов для дорог
                          (обязателен, если roads_csv_dir задан)
    departments_out_dir : папка результатов для департаментов
                          (обязателен, если departments_csv_dir задан)
    summary_out_dir     : папка для сводного CSV (batch_summary.csv);
                          если None — сохраняется рядом с первым out_dir
    progress_every      : шаг вывода прогресса внутри каждого файла
                          (0 — отключить; по умолчанию отключён для батча)

    Возвращает
    ----------
    list[dict] — сводная таблица по всем обработанным файлам:
        source        : "road" | "department"
        csv_name      : имя входного файла
        out_stem      : нормализованное имя (Дальневосточная-2025)
        alarm_count   : количество тревог
        events_path   : путь к events-CSV
        full_path     : путь к full-CSV
        error         : текст ошибки или None
    """

    if roads_csv_dir is None and departments_csv_dir is None:
        raise ValueError(
            "Необходимо указать хотя бы одну из папок: "
            "roads_csv_dir или departments_csv_dir"
        )

    # ── сборка задач: (csv_path, out_dir, source_label) ──────────────────────
    tasks: list[tuple[Path, Path, str]] = []

    if roads_csv_dir is not None:
        if roads_out_dir is None:
            raise ValueError("roads_out_dir обязателен при заданном roads_csv_dir")
        csv_files = sorted(Path(roads_csv_dir).glob("*.csv"))
        if not csv_files:
            print(f"[WARN] В папке дорог не найдено CSV: {roads_csv_dir}")
        for f in csv_files:
            tasks.append((f, Path(roads_out_dir), "road"))

    if departments_csv_dir is not None:
        if departments_out_dir is None:
            raise ValueError(
                "departments_out_dir обязателен при заданном departments_csv_dir"
            )
        csv_files = sorted(Path(departments_csv_dir).glob("*.csv"))
        if not csv_files:
            print(f"[WARN] В папке департаментов не найдено CSV: {departments_csv_dir}")
        for f in csv_files:
            tasks.append((f, Path(departments_out_dir), "department"))

    if not tasks:
        print("[WARN] Нет файлов для обработки.")
        return []

    total = len(tasks)
    print(f"\n{'='*52}")
    print(f"  CUSUM BATCH — файлов к обработке: {total}")
    print(f"{'='*52}\n")

    summary: list[dict] = []

    for idx, (csv_path, out_dir, source) in enumerate(tasks, start=1):
        stem = _normalize_stem(csv_path.stem)
        label = f"[{idx}/{total}] {source.upper()} | {stem}"

        if verbose:
            print(f"\n{'-'*52}")
            print(f"  {label}")
            print(f"{'-'*52}")

        record: dict = {
            "source":      source,
            "csv_name":    csv_path.name,
            "out_stem":    stem,
            "alarm_count": 0,
            "events_path": None,
            "full_path":   None,
            "error":       None,
        }

        try:
            events_path, full_path = run_cusum_exp_from_csv(
                csv_path,
                delta_target=delta_target,
                window_size=window_size,
                h_json_dir=h_json_dir,
                out_dir=out_dir,
                arl0_target=arl0_target,
                arl1_target=arl1_target,
                cooldown_after_alarm=cooldown_after_alarm,
                out_stem=stem,
                verbose=verbose,
                progress_every=progress_every,
            )

            # считаем тревоги из events-файла
            df_ev = pd.read_csv(events_path)
            alarm_count = len(df_ev)

            record["alarm_count"] = alarm_count
            record["events_path"] = str(events_path)
            record["full_path"]   = str(full_path)

        except Exception as exc:
            record["error"] = str(exc)
            print(f"[ERROR] {label} — {exc}")

        summary.append(record)

    # ── сводный CSV ───────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary)

    _first_out = tasks[0][1]
    _summary_dir = Path(summary_out_dir) if summary_out_dir else _first_out
    _summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _summary_dir / "batch_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # ── итог ─────────────────────────────────────────────────────────────────
    ok_count    = df_summary["error"].isna().sum()
    err_count   = df_summary["error"].notna().sum()
    total_alarms = df_summary["alarm_count"].sum()

    print(f"\n{'='*52}")
    print(f"  CUSUM BATCH ЗАВЕРШЁН")
    print(f"  Успешно:      {ok_count} / {total}")
    print(f"  Ошибок:       {err_count}")
    print(f"  Тревог всего: {int(total_alarms)}")
    print(f"  Сводный CSV:  {summary_path}")
    print(f"{'='*52}\n")

    return summary
