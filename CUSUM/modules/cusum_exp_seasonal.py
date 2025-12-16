from __future__ import annotations

import json
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


# ============================================================
# 2. ВЫБОР ПОРОГА h ИЗ JSON
# ============================================================

def select_h_from_json(
    *,
    delta_target: float,
    arl0_target: Optional[float],
    arl1_target: Optional[float],
    h_json_dir: Path,
) -> float:
    """
    Выбор оптимального порога h:

    • (δ, ARL0)        → h_from_delta_arl0.json
    • (δ, ARL1)        → h_from_delta_arl1.json
    • (δ, ARL0, ARL1)  → h_from_delta_arl0_arl1.json
    """

    if arl0_target is not None and arl1_target is not None:
        json_path = h_json_dir / "h_from_delta_arl0_arl1.json"
        keys = ("delta_target", "arl0_target", "arl1_target")

    elif arl0_target is not None:
        json_path = h_json_dir / "h_from_delta_arl0.json"
        keys = ("delta_target", "arl0_target")

    elif arl1_target is not None:
        json_path = h_json_dir / "h_from_delta_arl1.json"
        keys = ("delta_target", "arl1_target")

    else:
        raise ValueError("Нужно задать arl0_target и/или arl1_target")

    with open(json_path, "r", encoding="utf-8") as f:
        table = json.load(f)

    def distance(rec: dict) -> float:
        d = abs(rec["delta_target"] - delta_target)
        if "arl0_target" in keys:
            d += abs(rec["arl0_target"] - arl0_target)
        if "arl1_target" in keys:
            d += abs(rec["arl1_target"] - arl1_target)
        return d

    best = min(table, key=distance)
    return float(best["h_mc"])


# ============================================================
# 3. ОСНОВНАЯ ФУНКЦИЯ CUSUM
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
    verbose: bool = True,
    progress_every: int = 2000,
) -> tuple[Path, Path]:
    """
    Экспоненциальный CUSUM (Филаретов–Репин) для CSV.

    В events.csv добавляется флаг:
        IN_SPIKE_PERIOD ∈ {0,1}
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

    # ========================================================
    # 2. Порог h
    # ========================================================
    h = select_h_from_json(
        delta_target=delta_target,
        arl0_target=arl0_target,
        arl1_target=arl1_target,
        h_json_dir=h_json_dir,
    )

    cooldown = cooldown_after_alarm or window_size

    if verbose:
        print("\n================ CUSUM START ================")
        print(f"CSV:          {csv_path.name}")
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
    events_path = out_dir / f"{csv_path.stem}_cusum_events.csv"
    df_events.to_csv(events_path, index=False, encoding="utf-8-sig")

    df_full = df.copy()
    df_full["CUSUM_ALARM"] = alarm_flag
    full_path = out_dir / f"{csv_path.stem}_cusum_full.csv"
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
