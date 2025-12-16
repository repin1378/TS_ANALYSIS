# modules/cusum_threshold.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Literal
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import json


# =============================================================================
# 1. АНАЛИТИЧЕСКИЙ МЕТОД
# =============================================================================
# Формулы h(ARL0, δ) и ARL1(ARL0, δ) аппроксимируют большие табличные
# результаты, полученные авторами статьи методом моделирования.

_COEFFS = {
    1.25: {
        "h":    (-2.380, 0.798),
        "arl1": (-108.2, 30.9),
    },
    1.5: {
        "h":    (-2.383, 0.933),
        "arl1": (-35.7, 12.4),
    },
    2.0: {
        "h":    (-1.690, 0.941),
        "arl1": (-8.86, 4.84),
    },
    2.5: {
        "h":    (-1.542, 0.969),
        "arl1": (-4.09, 3.08),
    },
    3.0: {
        "h":    (-1.404, 0.978),
        "arl1": (-2.11, 2.26),
    },
}


def _nearest_delta(delta: float) -> float:

    # Выбирает ближайшее табличное значение δ из _COEFFS.

    arr = np.array(sorted(_COEFFS.keys()))
    return float(arr[np.argmin(np.abs(arr - delta))])


def h_from_arl0_delta(arl0: float, delta: float) -> float:

    # Возвращает порог h по целевому ARL0 и заданному δ
    # с использованием аппроксимирующих формул из статьи.
    # Если delta не совпадает точно с одним из табличных δ,
    # используется ближайшее табличное значение.

    if arl0 <= 0:
        raise ValueError("ARL0 должен быть > 0")

    d_used = _nearest_delta(delta)
    a, b = _COEFFS[d_used]["h"]
    return a + b * np.log(arl0)


def arl1_from_arl0_delta(arl0: float, delta: float) -> float:


    # Возвращает оценку ARL1 (среднее время запаздывания обнаружения)
    # по целевому ARL0 и δ.
    # Тоже использует ближайшее табличное значение δ.

    if arl0 <= 0:
        raise ValueError("ARL0 должен быть > 0")

    d_used = _nearest_delta(delta)
    a, b = _COEFFS[d_used]["arl1"]
    return a + b * np.log(arl0)


def efficiency_from_arl0_delta(arl0: float, delta: float) -> float:

    # Показатель эффективности:
    #     E = ARL0 / ARL1.

    arl1 = arl1_from_arl0_delta(arl0, delta)
    return arl0 / arl1


@dataclass
class AnalyticDesignResult:

    # Результат аналитического расчёта порога:
    #   - delta        — твой δ = λ1 / λ0
    #   - used_delta   — фактическое табличное δ, по которому брались коэффициенты
    #   - arl0         — заданный целевой ARL0
    #   - h            — рассчитанный порог CUSUM
    #   - arl1         — оценка среднего запаздывания обнаружения
    #   - efficiency   — E = ARL0 / ARL1

    delta: float
    used_delta: float
    arl0: float
    h: float
    arl1: float
    efficiency: float


def design_cusum_threshold_analytic(delta: float, arl0_target: float) -> AnalyticDesignResult:

    # Основная функция для аналитического метода.
    # Вход:
    #     delta       — отношение λ1 / λ0 (из постановки задачи)
    #     arl0_target — желаемый ARL0 (среднее число наблюдений до ложной тревоги)
    #
    # Выход:
    #     Аналитически рассчитанный порог h и связанные величины.
    #
    # Логика:
    #     1) По arl0_target и delta (через ближайшее табличное δ) считаем h.
    #     2) По тем же arl0_target и delta считаем ARL1.
    #     3) E = ARL0 / ARL1.


    h = h_from_arl0_delta(arl0_target, delta)
    arl1 = arl1_from_arl0_delta(arl0_target, delta)
    eff = arl0_target / arl1
    d_used = _nearest_delta(delta)

    return AnalyticDesignResult(
        delta=delta,
        used_delta=d_used,
        arl0=arl0_target,
        h=h,
        arl1=arl1,
        efficiency=eff
    )


# =============================================================================
# 2. МЕТОД MONTE CARLO (имитационный)
# =============================================================================
# Здесь мы проверяем/уточняем параметры CUSUM путём моделирования
# последовательностей из Exp(λ0) и Exp(λ1).


def llr_increment_exp(y: float, delta: float) -> float:

    # Приращение логарифма отношения правдоподобия (LLR)
    # для экспоненциального распределения в безразмерном виде:
    #
    #     Y = λ0 * X,
    #     под H0: Y ~ Exp(1),
    #     под H1: Y ~ Exp(δ).
    #
    # Тогда:
    #     z = ln(delta) - (delta - 1) * Y.

    return np.log(delta) - (delta - 1.0) * y


def simulate_cusum_run(
    delta: float,
    h: float,
    mode: Literal["H0", "H1"],
    max_steps: int = 200_000,
    rng: Optional[np.random.Generator] = None
) -> int:


    # Один прогон CUSUM до срабатывания порога h.
    #
    # Вход:
    #     delta     — отношение λ1 / λ0.
    #     h         — порог CUSUM.
    #     mode      — "H0" (имитация ложной тревоги) или "H1" (после разладки).
    #     max_steps — защитный максимум шагов (чтобы не зациклиться).
    #     rng       — генератор случайных чисел (для воспроизводимости, опционально).
    #
    # Выход:
    #     Число шагов (наблюдений), потребовавшихся до первого пересечения порога.
    #
    # Логика:
    #     1) Инициализируем S = 0.
    #     2) На каждом шаге:
    #         - генерируем Y:
    #             * под H0: Y ~ Exp(1)
    #             * под H1: Y ~ Exp(δ)
    #         - считаем приращение z = LLR(Y)
    #         - обновляем S = max(0, S + z)
    #         - как только S >= h → возвращаем номер шага.

    if rng is None:
        rng = np.random.default_rng()

    # лёгкое уведомление (только при первых вызовах)
    # можно отключить в будущем
    # print(f"[simulate] mode={mode}, h={h}, max_steps={max_steps}")

    s = 0.0

    for i in range(1, max_steps + 1):
        if mode == "H0":
            y = rng.exponential(scale=1.0)          # Exp(1)
        else:
            y = rng.exponential(scale=1.0 / delta)  # Exp(δ)

        s = max(0.0, s + llr_increment_exp(y, delta))

        if s >= h:
            return i

    return max_steps


@dataclass
class MCStatistics:

    # Результаты имитации для заданного h:
    #   - h          — порог
    #   - arl0       — оценка ARL0 (среднее число шагов до ложной тревоги)
    #   - arl1       — оценка ARL1 (среднее число шагов до обнаружения)
    #   - efficiency — E = ARL0 / ARL1

    h: float
    arl0: float
    arl1: float
    efficiency: float


def estimate_metrics_mc(
    delta: float,
    h: float,
    n_runs_arl0: int = 2000,
    n_runs_arl1: int = 2000,
    max_steps: int = 200_000,
    rng: Optional[np.random.Generator] = None
) -> MCStatistics:

    if rng is None:
        rng = np.random.default_rng()

    print(f"\n[MC] Оценка метрик для h={h:.3f}, delta={delta}")
    print(f"[MC] Прогоны: ARL0={n_runs_arl0}, ARL1={n_runs_arl1}")

    # === ARL0 ===
    t0 = []
    checkpoint0 = max(1, n_runs_arl0 // 10)

    for i in range(n_runs_arl0):
        t0.append(simulate_cusum_run(delta, h, "H0", max_steps, rng))
        if (i + 1) % checkpoint0 == 0:
            print(f"[MC] ARL0 progress: {i+1}/{n_runs_arl0}")

    arl0 = float(np.mean(t0))

    # === ARL1 ===
    t1 = []
    checkpoint1 = max(1, n_runs_arl1 // 10)

    for i in range(n_runs_arl1):
        t1.append(simulate_cusum_run(delta, h, "H1", max_steps, rng))
        if (i + 1) % checkpoint1 == 0:
            print(f"[MC] ARL1 progress: {i+1}/{n_runs_arl1}")

    arl1 = float(np.mean(t1))
    eff = arl0 / arl1

    print(f"[MC] Готово для h={h:.3f}: ARL0={arl0:.1f}, ARL1={arl1:.1f}, E={eff:.3f}")

    return MCStatistics(h=h, arl0=arl0, arl1=arl1, efficiency=eff)


@dataclass
class MCSearchResult:

    # Результат поиска оптимального порога по сетке:
    #   - delta      — использованное δ
    #   - best_h     — лучший найденный порог
    #   - best_stats — MCStatistics для этого порога
    #   - grid_h     — список всех проверенных порогов

    delta: float
    best_h: float
    best_stats: MCStatistics
    grid_h: List[float]


def find_optimal_h_mc(
    delta: float,
    h_grid: Optional[np.ndarray] = None,
    target_arl0_min: Optional[float] = None,
    n_runs_arl0: int = 2000,
    n_runs_arl1: int = 2000,
    max_steps: int = 200_000,
    rng: Optional[np.random.Generator] = None
) -> MCSearchResult:


    # Поиск "оптимального" порога h по сетке h_grid при заданном δ:
    #
    #     - Считаем метрики (ARL0, ARL1, E) для каждого h.
    #     - Оставляем только те h, где ARL0 >= target_arl0_min (если задано).
    #     - Выбираем h, при котором эффективность E = ARL0 / ARL1 максимальна.
    #
    # Вход:
    #     delta          — λ1 / λ0
    #     h_grid         — массив порогов, если None → берём np.arange(0.5, 8.05, 0.1)
    #     target_arl0_min — минимально допустимый ARL0 (может быть None)
    #     n_runs_arl0, n_runs_arl1 — число прогонов
    #     max_steps      — максимум шагов одного прогона
    #     rng            — генератор случайных чисел
    #
    # Выход:
    #     MCSearchResult с лучшим порогом и статистикой.


    if h_grid is None:
        h_grid = np.arange(0.5, 8.05, 0.1)

    if rng is None:
        rng = np.random.default_rng()

    print(f"\n[MC-SEARCH] Поиск оптимального h для delta={delta}")
    print(f"[MC-SEARCH] Сетка h: {h_grid}")

    best_stats = None
    best_h = None

    for h in h_grid:
        print(f"\n[MC-SEARCH] Проверяю h={h:.3f}...")
        stats = estimate_metrics_mc(
            delta=delta,
            h=h,
            n_runs_arl0=n_runs_arl0,
            n_runs_arl1=n_runs_arl1,
            max_steps=max_steps,
            rng=rng,
        )

        if target_arl0_min is not None:
            if stats.arl0 < target_arl0_min:
                print(f"[MC-SEARCH] h={h:.3f} отклонён: ARL0={stats.arl0:.1f} < {target_arl0_min}")
                continue

        if best_stats is None or stats.efficiency > best_stats.efficiency:
            print(f"[MC-SEARCH] h={h:.3f} стал лучшим! E={stats.efficiency:.3f}")
            best_stats = stats
            best_h = h

    if best_stats is None:
        raise RuntimeError("Не найден подходящий порог h.")

    print(f"\n[MC-SEARCH] Лучший порог: h={best_h:.3f}")
    print(f"[MC-SEARCH] Лучшие метрики: ARL0={best_stats.arl0:.1f}, ARL1={best_stats.arl1:.1f}, E={best_stats.efficiency:.3f}")

    return MCSearchResult(
        delta=delta,
        best_h=best_h,
        best_stats=best_stats,
        grid_h=list(h_grid)
    )


# =============================================================================
# 3. ФУНКЦИЯ СРАВНЕНИЯ АНАЛИТИЧЕСКОГО МЕТОДА И MONTE CARLO
# =============================================================================

@dataclass
class ComparisonRow:
    delta: float
    arl0_target: float
    h_analytic: float
    arl1_analytic: float
    eff_analytic: float
    arl0_mc: float
    arl1_mc: float
    eff_mc: float
    h_mc: float   # ← лучший порог по MC



def compare_analytic_vs_mc_extended(
    deltas: list[float] | None = None,
    arl0_targets: list[float] | None = None,
    n_runs_arl0: int = 2000,
    n_runs_arl1: int = 2000,
    max_steps: int = 200_000,
    seed: int = 42,
) -> list[ComparisonRow]:


    if deltas is None:
        deltas = [1.25, 1.5, 2.0, 2.5, 3.0]

    if arl0_targets is None:
        arl0_targets = [200.0, 500.0, 1000.0]

    rng = np.random.default_rng(seed)
    results = []

    for delta in deltas:
        print(f"\n===============================")
        print(f"[COMPARE] Начинаю сравнение для delta={delta}")
        print(f"===============================")

        for arl0_target in arl0_targets:
            print(f"\n[COMPARE] → ARL0_target={arl0_target}")

            ana = design_cusum_threshold_analytic(delta=delta, arl0_target=arl0_target)
            print(f"[COMPARE] Аналитический порог: h={ana.h:.3f}")

            mc = estimate_metrics_mc(
                delta=delta,
                h=ana.h,
                n_runs_arl0=n_runs_arl0,
                n_runs_arl1=n_runs_arl1,
                max_steps=max_steps,
                rng=rng,
            )

            print("[COMPARE] Запускаю поиск лучшего h_mc в окрестности...")

            h_grid = np.arange(ana.h - 0.6, ana.h + 0.6, 0.1)
            mc_search = find_optimal_h_mc(
                delta=delta,
                h_grid=h_grid,
                target_arl0_min=None,
                n_runs_arl0=n_runs_arl0,
                n_runs_arl1=n_runs_arl1,
                max_steps=max_steps,
                rng=rng,
            )

            row = ComparisonRow(
                delta=delta,
                arl0_target=arl0_target,
                h_analytic=ana.h,
                arl1_analytic=ana.arl1,
                eff_analytic=ana.efficiency,
                arl0_mc=mc.arl0,
                arl1_mc=mc.arl1,
                eff_mc=mc.efficiency,
                h_mc=mc_search.best_h
            )

            results.append(row)

    print("\n[COMPARE] Готово!")
    return results

def save_comparison_to_csv(rows: list[ComparisonRow], path: Path):
    df = pd.DataFrame([r.__dict__ for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Сравнительный отчёт сохранён в CSV: {path}")


def save_comparison_to_excel(rows: list[ComparisonRow], path: Path):
    df = pd.DataFrame([r.__dict__ for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    print(f"Сравнительный отчёт сохранён в Excel: {path}")


@dataclass
class HSearchResult:
    delta: float
    best_h: float
    arl0: float
    arl1: float
    efficiency: float
    objective: float   # здесь мы будем хранить целевую метрику E

def find_h_from_delta_arl1(
    delta: float,
    arl1_target: float,
    h_grid: Optional[np.ndarray] = None,
    n_runs: int = 2000,
    max_steps: int = 200_000,
    rng: Optional[np.random.Generator] = None
) -> HSearchResult:
    """
    Поиск порога h при котором:
        ARL1(h) <= arl1_target
    и эффективность E(h)=ARL0/ARL1 максимальна.
    Аналогично h(δ,ARL0) из статьи, но 'зеркально'.
    """

    if h_grid is None:
        h_grid = np.arange(0.5, 8.05, 0.1)

    if rng is None:
        rng = np.random.default_rng()

    best_h = None
    best_eff = None
    best_stats = None

    print(f"\n[SEARCH_ARL1] delta={delta}, ARL1_target={arl1_target}")
    print(f"[SEARCH_ARL1] h_grid size={len(h_grid)}")

    for h in h_grid:
        print(f"  [SEARCH_ARL1] → h={h:.3f}")

        stats = estimate_metrics_mc(
            delta=delta,
            h=h,
            n_runs_arl0=n_runs,
            n_runs_arl1=n_runs,
            max_steps=max_steps,
            rng=rng
        )

        if stats.arl1 > arl1_target:
            print(f"     skip (ARL1={stats.arl1:.1f} > {arl1_target})")
            continue

        if best_eff is None or stats.efficiency > best_eff:
            best_eff = stats.efficiency
            best_h = h
            best_stats = stats
            print(f"     ✓ new best h={h:.3f}, E={best_eff:.3f}")

    if best_h is None:
        raise RuntimeError("Нет порога h, удовлетворяющего ARL1_target.")

    return HSearchResult(
        delta=delta,
        best_h=best_h,
        arl0=best_stats.arl0,
        arl1=best_stats.arl1,
        efficiency=best_stats.efficiency,
        objective=best_stats.efficiency
    )

def compute_arl0_from_delta_arl1(
    delta: float,
    arl1_target: float,
    h_grid: Optional[np.ndarray] = None,
    n_runs: int = 2000,
    max_steps: int = 200_000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Аналог ARL1(δ,ARL0) из статьи — но зеркальное:
    ARL0(δ,ARL1_target).

    Мы сначала вычисляем h(δ,ARL1_target),
    затем возвращаем ARL0(h) и ARL1(h) на этом пороге.
    """

    res = find_h_from_delta_arl1(
        delta=delta,
        arl1_target=arl1_target,
        h_grid=h_grid,
        n_runs=n_runs,
        max_steps=max_steps,
        rng=rng
    )

    return {
        "delta": delta,
        "ARL1_target": arl1_target,
        "h_best": res.best_h,
        "ARL0_best": res.arl0,
        "ARL1_best": res.arl1,
        "E_best": res.efficiency
    }


def build_tables_h_delta_arl1(
    deltas: list[float],
    arl1_targets: list[float],
    h_grid: np.ndarray,
    n_runs: int = 2000,
    max_steps: int = 200_000,
    seed: int = 123,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Формирует таблицу аналогичную статье:
        (δ, ARL1_target) → h_best, ARL0_best, ARL1_best, E_best
    """

    rng = np.random.default_rng(seed)
    rows = []

    for delta in deltas:
        print(f"\n=== Δ={delta} ===")

        for arl1_t in arl1_targets:
            print(f"-- ARL1_target={arl1_t} --")

            res = compute_arl0_from_delta_arl1(
                delta=delta,
                arl1_target=arl1_t,
                h_grid=h_grid,
                n_runs=n_runs,
                max_steps=max_steps,
                rng=rng
            )

            rows.append(res)

    df = pd.DataFrame(rows)

    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n[Saved] {save_path}")

    return df

def fit_new_approximations(df: pd.DataFrame):
    """
    df должен содержать:
        delta, ARL1_target, h_best, ARL0_best

    Возвращает словари аппроксимационных коэффициентов:
        h ≈ a_h + b_h * ln(ARL1)
        ARL0 ≈ a_0 + b_0 * ln(ARL1)
    """

    results_h = {}
    results_ARL0 = {}

    for delta, df_d in df.groupby("delta"):

        X = np.log(df_d["ARL1_target"].astype(float))
        y_h = df_d["h_best"].astype(float)
        y_ARL0 = df_d["ARL0_best"].astype(float)

        # линейная регрессия y = a + b * log(ARL1)
        b_h, a_h = np.polyfit(X, y_h, deg=1)
        b_0, a_0 = np.polyfit(X, y_ARL0, deg=1)

        results_h[delta] = (a_h, b_h)
        results_ARL0[delta] = (a_0, b_0)

        print(f"\nΔ={delta}:")
        print(f"  h(ARL1) ≈ {a_h:.4f} + {b_h:.4f} ln(ARL1)")
        print(f"  ARL0(ARL1) ≈ {a_0:.4f} + {b_0:.4f} ln(ARL1)")

    return results_h, results_ARL0

def find_h_from_delta_arl1_arl0(
    delta: float,
    arl1_target: float,
    arl0_target: float,
    h_grid: Optional[np.ndarray] = None,
    n_runs: int = 2000,
    max_steps: int = 200_000,
    rng: Optional[np.random.Generator] = None
) -> HSearchResult:
    """
    Поиск порога h при тройном критерии:
        1) ARL1(h) <= arl1_target
        2) ARL0(h) >= arl0_target
        3) среди таких h максимизируем эффективность: E(h) = ARL0(h)/ARL1(h)

    Это наиболее универсальный и строгий вариант:
    аналог поиска оптимального порога по двум ограничениям одновременно.
    """

    if h_grid is None:
        h_grid = np.arange(0.5, 10.0, 0.1)

    if rng is None:
        rng = np.random.default_rng()

    best_h: Optional[float] = None
    best_eff: Optional[float] = None
    best_stats: Optional[MCStatistics] = None

    print(f"\n[SEARCH_ARL1_ARL0] delta={delta}, ARL1_target={arl1_target}, ARL0_target={arl0_target}")
    print(f"[SEARCH_ARL1_ARL0] h_grid size={len(h_grid)}")

    for h in h_grid:
        print(f"\n[SEARCH_ARL1_ARL0] → h={h:.3f}")

        stats = estimate_metrics_mc(
            delta=delta,
            h=h,
            n_runs_arl0=n_runs,
            n_runs_arl1=n_runs,
            max_steps=max_steps,
            rng=rng
        )

        # 1) Ограничение по скорости обнаружения
        if stats.arl1 > arl1_target:
            print(f"    skip ARL1={stats.arl1:.1f} > {arl1_target}")
            continue

        # 2) Ограничение по ложным тревогам
        if stats.arl0 < arl0_target:
            print(f"    skip ARL0={stats.arl0:.1f} < {arl0_target}")
            continue

        # 3) Целевая функция — максимальная эффективность
        E = stats.efficiency
        print(f"    ✓ acceptable: ARL0={stats.arl0:.1f}, ARL1={stats.arl1:.1f}, E={E:.3f}")

        if best_eff is None or E > best_eff:
            best_eff = E
            best_h = h
            best_stats = stats
            print(f"    ★ new best h={h:.3f} (E={best_eff:.3f})")

    if best_h is None:
        raise RuntimeError(
            f"Не найден порог h, удовлетворяющий двум условиям: "
            f"ARL1 <= {arl1_target} и ARL0 >= {arl0_target}."
        )

    print(f"\n[SEARCH_ARL1_ARL0] Лучший порог: h={best_h:.3f}")
    print(f"  ARL0={best_stats.arl0:.1f}, ARL1={best_stats.arl1:.1f}, E={best_stats.efficiency:.3f}")

    return HSearchResult(
        delta=delta,
        best_h=best_h,
        arl0=best_stats.arl0,
        arl1=best_stats.arl1,
        efficiency=best_stats.efficiency,
        objective=best_eff
    )




# =============================================================================
# Расчет экспериментальных значений
# =============================================================================

def _mc_single_arg(args):
    """Wrapper для multiprocessing."""
    return simulate_cusum_run(*args)


def estimate_metrics_mc_parallel(
    delta: float,
    h: float,
    n_runs_arl0: int,
    n_runs_arl1: int,
    max_steps: int = 200_000,
    n_workers: int | None = None,
):
    """
    Параллельная версия Monte-Carlo оценки ARL0, ARL1.
    """

    if n_workers is None:
        n_workers = cpu_count()

    # ======= H0 =======
    args0 = [(delta, h, "H0", max_steps, None) for _ in range(n_runs_arl0)]
    with Pool(n_workers) as p:
        t0 = p.map(_mc_single_arg, args0)

    # ======= H1 =======
    args1 = [(delta, h, "H1", max_steps, None) for _ in range(n_runs_arl1)]
    with Pool(n_workers) as p:
        t1 = p.map(_mc_single_arg, args1)

    arl0 = float(np.mean(t0))
    arl1 = float(np.mean(t1))
    eff = arl0 / arl1

    return arl0, arl1, eff


def _auto_h_grid_for_delta_arl0(delta: float, arl0_target: float) -> np.ndarray:
    """
    Автоподбор диапазона h для задачи h(δ, ARL0_target).
    """
    if delta <= 1.3:
        h_max = 4.5
    elif delta <= 1.8:
        h_max = 6.0
    elif delta <= 2.5:
        h_max = 8.0
    else:
        h_max = 10.0

    # небольшая поправка по ARL0
    h_max *= min(1.0 + 0.001 * arl0_target, 1.5)

    return np.arange(0.5, min(h_max, 12.0) + 1e-9, 0.1)



def run_arl0_delta_experiment(
    arl0_target: float,
    delta_target: float,
    n_runs_mc: int,
    csv_path: Path,
    json_path: Path,
    mode: str = "mc_optimal_E",
    h_grid: np.ndarray | None = None,
    max_steps: int = 200_000,
    n_workers: int | None = None,
    arl0_tolerance: float = 0.20,

    # --- новые практические ограничения ---
    arl1_min_factor: float = 3.0,   # ARL1 ≥ ARL0 / factor
    arl1_min_abs: float = 5.0,      # абсолютный минимум ARL1
    arl1_max_factor: float = 1.2,   # ARL1 ≤ ARL0 * factor
):
    """
    Финальная инженерная версия run_arl0_delta_experiment.

    mode="analytic":
        • повторяет статью Филаретова–Репина
        • использует аналитический h
        • проверяет ARL0, ARL1, E через Monte-Carlo
        • сохраняет таблицу + аппроксимацию в CSV

    mode="mc_optimal_E":
        • ищет h, который:
            - даёт ARL0_mc ≈ ARL0_target (± tolerance)
            - имеет разумный ARL1 (не слишком маленький и не слишком большой)
            - максимизирует E = ARL0 / ARL1
        • автоподбор h_grid
        • fallback по ARL0
        • сохраняет результат в JSON (без дублей)
        • строит аппроксимацию h = a + b ln(ARL0)
    """

    # =====================================================================
    # MODE 1: ANALYTIC — строго по статье
    # =====================================================================
    if mode == "analytic":

        print(f"[analytic] δ={delta_target}, ARL0_target={arl0_target}")

        # 1) Аналитическое значение порога h из статьи
        ana = design_cusum_threshold_analytic(delta_target, arl0_target)
        h_an = ana.h
        arl1_an = ana.arl1
        E_an = ana.efficiency

        # 2) Монте-Карло оценка метрик для ФИКСИРОВАННОГО h_an
        arl0_mc, arl1_mc, E_mc = estimate_metrics_mc_parallel(
            delta_target,
            h_an,
            n_runs_arl0=n_runs_mc,
            n_runs_arl1=n_runs_mc,
            max_steps=max_steps,
            n_workers=n_workers,
        )

        # ===== CSV =====
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=[
                "arl0_target", "delta_target", "n_runs_mc",
                "h_an", "arl1_an", "E_an",
                "h_mc", "ARL0_mc", "ARL1_mc", "E_mc",
                "a", "b",
            ])

        # В analytic режиме h_mc = h_an (мы НЕ оптимизируем h, а берём его из статьи)
        h_mc = h_an

        df = pd.concat([
            df,
            pd.DataFrame([{
                "arl0_target": arl0_target,
                "delta_target": delta_target,
                "n_runs_mc": n_runs_mc,
                "h_an": h_an,
                "arl1_an": arl1_an,
                "E_an": E_an,
                "h_mc": h_mc,
                "ARL0_mc": arl0_mc,
                "ARL1_mc": arl1_mc,
                "E_mc": E_mc,
                "a": np.nan,
                "b": np.nan,
            }])
        ], ignore_index=True)

        # ========================================================
        # Аппроксимация ДЛЯ h:  h ≈ a + b ln(ARL0)
        # (как в статье для таблицы порогов)
        # ========================================================
        df_delta = df[df["delta_target"] == delta_target].dropna(subset=["h_an"])

        if df_delta.shape[0] >= 2:
            X = np.log(df_delta["arl0_target"].astype(float))
            y = df_delta["h_an"].astype(float)  # аппроксимируем именно h
            b, a = np.polyfit(X, y, 1)

            df.loc[df["delta_target"] == delta_target, "a"] = a
            df.loc[df["delta_target"] == delta_target, "b"] = b

            print(f"[FIT analytic] h ≈ {a:.4f} + {b:.4f} ln(ARL0)")
        else:
            print("[FIT analytic] Недостаточно точек для аппроксимации h.")

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE CSV] {csv_path}")
        return

    # =====================================================================
    # MODE 2: MC_OPTIMAL_E — практический режим
    # =====================================================================
    print(f"[mc_optimal_E] δ={delta_target}, ARL0_target={arl0_target}")

    if h_grid is None:
        h_grid = _auto_h_grid_for_delta_arl0(delta_target, arl0_target)

    arl0_min = arl0_target * (1 - arl0_tolerance)
    arl0_max = arl0_target * (1 + arl0_tolerance)

    arl1_min = max(arl0_target / arl1_min_factor, arl1_min_abs)
    arl1_max = arl0_target * arl1_max_factor

    best_h = None
    best_E = -np.inf
    best_arl0 = None
    best_arl1 = None

    candidates = []

    for h in h_grid:
        arl0_mc, arl1_mc, E_mc = estimate_metrics_mc_parallel(
            delta_target,
            h,
            n_runs_arl0=n_runs_mc,
            n_runs_arl1=n_runs_mc,
            max_steps=max_steps,
            n_workers=n_workers,
        )

        candidates.append((h, arl0_mc, arl1_mc, E_mc))

        if not (arl0_min <= arl0_mc <= arl0_max):
            continue
        if not (arl1_min <= arl1_mc <= arl1_max):
            continue

        if E_mc > best_E:
            best_E = E_mc
            best_h = h
            best_arl0 = arl0_mc
            best_arl1 = arl1_mc

    # ---------- Fallback ----------
    if best_h is None:
        print("[WARN] Нет h, удовлетворяющих ограничениям. Fallback по ARL0.")
        best_dist = np.inf
        for h, a0, a1, E in candidates:
            dist = abs(a0 - arl0_target)
            if dist < best_dist:
                best_dist = dist
                best_h = h
                best_arl0 = a0
                best_arl1 = a1
                best_E = E

    print(
        f"[BEST] h={best_h:.3f}, "
        f"ARL0_mc={best_arl0:.1f}, "
        f"ARL1_mc={best_arl1:.1f}, "
        f"E={best_E:.3f}"
    )

    # ================= JSON =================
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    new_record = {
        "arl0_target": float(arl0_target),
        "delta_target": float(delta_target),
        "n_runs_mc": int(n_runs_mc),
        "h_grid": {
            "start": float(h_grid[0]),
            "stop": float(h_grid[-1]),
            "step": float(h_grid[1] - h_grid[0]),
            "n_points": int(len(h_grid)),
        },
        "h_mc": float(best_h),
        "E_mc": float(best_E),
        "ARL0_mc": float(best_arl0),
        "ARL1_mc": float(best_arl1),
        "a": None,
        "b": None,
    }

    existing = [
        r for r in data
        if r["delta_target"] == delta_target
        and r["arl0_target"] == arl0_target
    ]

    if existing:
        old = existing[0]
        if new_record["E_mc"] > old["E_mc"]:
            data.remove(old)
            data.append(new_record)
            print("[UPDATE] Запись улучшена.")
        else:
            print("[SKIP] Старая запись лучше.")
    else:
        data.append(new_record)
        print("[ADD] Новая запись добавлена.")

    same_delta = [r for r in data if r["delta_target"] == delta_target]

    if len(same_delta) >= 2:
        X = np.log([r["arl0_target"] for r in same_delta])
        y = [r["h_mc"] for r in same_delta]
        b, a = np.polyfit(X, y, 1)
        for r in same_delta:
            r["a"] = float(a)
            r["b"] = float(b)
        print(f"[FIT mc_optimal_E] h ≈ {a:.4f} + {b:.4f} ln(ARL0)")
    else:
        print("[FIT mc_optimal_E] Недостаточно точек для аппроксимации.")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[SAVE JSON] {json_path}")


def run_arl1_delta_experiment(
    arl0_target: float,
    delta_target: float,
    n_runs_mc: int,
    csv_path: Path,
    json_path: Path,
    mode: str = "mc_optimal_E",
    h_grid: np.ndarray | None = None,
    max_steps: int = 200_000,
    n_workers: int | None = None,
    arl0_tolerance: float = 0.20,
):
    """
    Расчёт ARL1(δ, ARL0) аналогично Таблице 2 Филаретова–Репина–Лецкого.

    Два режима:
    -------------------------------------------------------------------
    mode="analytic"
        — использует аналитический порог h(δ, ARL0)
        — вычисляет ARL1 через MC
        — сохраняет таблицу ARL1(δ,ARL0) в CSV
        — строит аппроксимацию ARL1 = a + b ln(ARL0)

    mode="mc_optimal_E"
        — выбирает h_mc с максимальным E = ARL0/ARL1
          среди порогов, удовлетворяющих ARL0_mc ≈ ARL0_target
        — ARL1_mc является практическим ARL1(δ, ARL0)
        — сохраняет в JSON-справочник без дублей
        — добавляет аппроксимацию ARL1 = a + b ln(ARL0)
    -------------------------------------------------------------------
    """

    # =====================================================================
    # MODE 1 — АНАЛИТИКА (точное повторение Таблицы 2)
    # =====================================================================
    if mode == "analytic":

        print(f"[analytic ARL1] δ={delta_target}, ARL0={arl0_target}")

        # Получаем аналитический порог и аналитический ARL1
        ana = design_cusum_threshold_analytic(delta_target, arl0_target)
        h_an = ana.h
        arl1_an = ana.arl1  # Табличное значение ARL1
        E_an = ana.efficiency

        # Monte Carlo для проверки ARL1
        arl0_mc, arl1_mc, E_mc = estimate_metrics_mc_parallel(
            delta_target,
            h_an,
            n_runs_arl0=n_runs_mc,
            n_runs_arl1=n_runs_mc,
            max_steps=max_steps,
            n_workers=n_workers,
        )

        # ==== CSV ====
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=[
                "arl0_target", "delta_target", "n_runs_mc",
                "h_an", "ARL1_an", "E_an",
                "ARL0_mc", "ARL1_mc", "E_mc",
                "a", "b"
            ])

        df = pd.concat([
            df,
            pd.DataFrame([{
                "arl0_target": arl0_target,
                "delta_target": delta_target,
                "n_runs_mc": n_runs_mc,
                "h_an": h_an,
                "ARL1_an": arl1_an,
                "E_an": E_an,
                "ARL0_mc": arl0_mc,
                "ARL1_mc": arl1_mc,
                "E_mc": E_mc,
                "a": np.nan,
                "b": np.nan,
            }])
        ], ignore_index=True)

        # Аппроксимация ARL1 = a + b ln(ARL0)
        df_delta = df[df["delta_target"] == delta_target]

        if df_delta.shape[0] >= 2:
            X = np.log(df_delta["arl0_target"].astype(float))
            y = df_delta["ARL1_mc"].astype(float)
            b, a = np.polyfit(X, y, 1)
            df.loc[df["delta_target"] == delta_target, "a"] = a
            df.loc[df["delta_target"] == delta_target, "b"] = b
            print(f"[FIT analytic ARL1] ARL1 ≈ {a:.4f} + {b:.4f} ln(ARL0)")
        else:
            print("[FIT analytic ARL1] Недостаточно точек.")

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE CSV] {csv_path}")
        return

    # =====================================================================
    # MODE 2 — MC_OPTIMAL_E
    # =====================================================================
    if mode == "mc_optimal_E":

        print(f"[mc_optimal_E ARL1] δ={delta_target}, ARL0={arl0_target}")

        if h_grid is None:
            h_grid = np.arange(0.3, 7.01, 0.1)

        arl0_min = arl0_target * (1 - arl0_tolerance)
        arl0_max = arl0_target * (1 + arl0_tolerance)

        best_h = None
        best_E = -np.inf
        best_arl0 = None
        best_arl1 = None

        # Перебор h
        for h in h_grid:
            arl0_mc, arl1_mc, E_mc = estimate_metrics_mc_parallel(
                delta_target,
                h,
                n_runs_arl0=n_runs_mc,
                n_runs_arl1=n_runs_mc,
                max_steps=max_steps,
                n_workers=n_workers,
            )

            # Берём только допустимые ARL0
            if not (arl0_min <= arl0_mc <= arl0_max):
                continue

            # Максимизируем эффективность среди допустимых
            if E_mc > best_E:
                best_E = E_mc
                best_h = h
                best_arl0 = arl0_mc
                best_arl1 = arl1_mc

        # Fallback — ближайший ARL0
        if best_h is None:
            print("[WARN ARL1] Нет h удовлетворяющих ARL0_target. Ищу ближайший.")
            best_dist = np.inf
            for h in h_grid:
                arl0_mc, arl1_mc, E_mc = estimate_metrics_mc_parallel(
                    delta_target,
                    h,
                    n_runs_arl0=n_runs_mc,
                    n_runs_arl1=n_runs_mc,
                    max_steps=max_steps,
                    n_workers=n_workers,
                )
                dist = abs(arl0_mc - arl0_target)
                if dist < best_dist:
                    best_dist = dist
                    best_h = h
                    best_E = E_mc
                    best_arl0 = arl0_mc
                    best_arl1 = arl1_mc

        print(f"[BEST ARL1] h={best_h:.3f}, ARL0_mc={best_arl0:.1f}, ARL1_mc={best_arl1:.1f}, E={best_E:.3f}")

        # ===== JSON-СПРАВОЧНИК =====
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        new_record = {
            "arl0_target": float(arl0_target),
            "delta_target": float(delta_target),
            "n_runs_mc": int(n_runs_mc),

            "h_grid": {
                "start": float(h_grid[0]),
                "stop": float(h_grid[-1]),
                "step": float(h_grid[1] - h_grid[0]),
                "n_points": int(len(h_grid)),
            },

            "h_mc": float(best_h),
            "E_mc": float(best_E),
            "ARL0_mc": float(best_arl0),
            "ARL1_mc": float(best_arl1),
            "a": None,
            "b": None,
        }

        # Удаляем старые записи
        existing = [
            r for r in data
            if r["delta_target"] == delta_target
            and r["arl0_target"] == arl0_target
        ]

        if existing:
            old = existing[0]
            if new_record["E_mc"] > old["E_mc"]:
                data.remove(old)
                data.append(new_record)
                print("[UPDATE ARL1] Запись улучшена.")
            else:
                print("[SKIP ARL1] Старая запись лучше — пропущено.")
        else:
            data.append(new_record)
            print("[ADD ARL1] Новая запись добавлена.")

        # ===== Аппроксимация ARL1 = a + b ln(ARL0) =====
        same_delta = [r for r in data if r["delta_target"] == delta_target]

        if len(same_delta) >= 2:
            X = np.log([r["arl0_target"] for r in same_delta])
            y = [r["ARL1_mc"] for r in same_delta]
            b, a = np.polyfit(X, y, 1)

            for r in same_delta:
                r["a"] = float(a)
                r["b"] = float(b)

            print(f"[FIT ARL1 mc_optimal_E] ARL1 ≈ {a:.4f} + {b:.4f} ln(ARL0)")
        else:
            print("[FIT ARL1 mc_optimal_E] Недостаточно точек.")

        # Сохранение JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"[SAVE JSON] {json_path}")
        return

def _auto_h_grid_for_delta_arl1(delta: float, arl1_target: float) -> np.ndarray:
    """
    Грубый, но практичный автоподбор диапазона h по delta и целевому ARL1.

    Идея: при росте delta и ARL1_target разумный максимум h растёт.
    """
    # Базовый максимум по delta
    if delta <= 1.3:
        base_max = 5.1
    elif delta <= 2.1:
        base_max = 6.1
    elif delta <= 3.1:
        base_max = 7.1
    else:
        base_max = 9.1

    # Лёгкая поправка по целевому ARL1: чем больше ARL1_target, тем больше h_max
    factor = 1.0 + 0.01 * max(0.0, arl1_target - 20.0)   # +1% за каждую единицу выше 20
    factor = min(factor, 2.0)                            # не более чем ×2
    h_max = min(base_max * factor, 15.0)

    return np.arange(0.5, h_max + 1e-9, 0.1)


def run_arl1_target_delta_experiment(
    arl1_target: float,
    delta_target: float,
    n_runs_mc: int,
    json_path: Path,
    h_grid: np.ndarray | None = None,
    max_steps: int = 200_000,
    n_workers: int | None = None,
    arl1_tolerance: float = 0.20,   # допуск по ARL1 (±20%)
    arl0_max: float = 300.0,        # верхняя разумная граница ARL0
    arl0_min_factor: float = 1.5,   # ARL0_min = max(ARL1_target * factor, 10)
):
    """
    Практическая оптимизация порога h при заданных (δ, ARL1_target).

    Задача:
        1) найти пороги h, для которых:
               ARL1_mc(h) ≈ ARL1_target (в пределах arl1_tolerance)
               ARL0_min <= ARL0_mc(h) <= arl0_max
        2) среди них выбрать h_mc с максимальной эффективностью:
               E(h) = ARL0(h) / ARL1(h)
        3) если подходящих h нет — выбрать h с минимальной |ARL1_mc - ARL1_target|
           (fallback).

    В отличие от вариантов с ARL0_target, здесь нет режима analytic,
    т.к. в статье нет таблиц h(δ, ARL1).
    Результаты сохраняются в JSON-справочник (без дублей по (δ, ARL1_target)).
    """

    print(f"[mc_optimal_E ARL1-target] δ={delta_target}, ARL1_target={arl1_target}")

    # 1) Автоподбор сетки h, если пользователь её не задал
    if h_grid is None:
        h_grid = _auto_h_grid_for_delta_arl1(delta_target, arl1_target)

    # 2) Границы по ARL1 и ARL0
    arl1_min = arl1_target * (1 - arl1_tolerance)
    arl1_max = arl1_target * (1 + arl1_tolerance)
    arl0_min = max(arl1_target * arl0_min_factor, 10.0)

    best_h: float | None = None
    best_E: float = -np.inf
    best_arl0: float | None = None
    best_arl1: float | None = None

    # Для fallback-режима будем сохранять все точки
    all_candidates: list[dict[str, float]] = []

    # 3) Перебор порогов h
    for h in h_grid:
        arl0_mc, arl1_mc, E_mc = estimate_metrics_mc_parallel(
            delta_target,
            h,
            n_runs_arl0=n_runs_mc,
            n_runs_arl1=n_runs_mc,
            max_steps=max_steps,
            n_workers=n_workers,
        )

        all_candidates.append(
            {
                "h": h,
                "ARL0_mc": arl0_mc,
                "ARL1_mc": arl1_mc,
                "E_mc": E_mc,
            }
        )

        # Фильтр по ARL1_target
        if not (arl1_min <= arl1_mc <= arl1_max):
            continue

        # Фильтр по ARL0 (двойное ограничение)
        if not (arl0_min <= arl0_mc <= arl0_max):
            continue

        # Максимизация E среди допустимых
        if E_mc > best_E:
            best_E = E_mc
            best_h = h
            best_arl0 = arl0_mc
            best_arl1 = arl1_mc

    # 4) Fallback, если не найдено ни одного h, удовлетворяющего обоим ограничениям
    if best_h is None:
        print("[WARN ARL1-target] Нет h, удовлетворяющих ARL1 и ARL0 ограничениям. Ищу ближайший по ARL1.")

        best_score = np.inf
        for cand in all_candidates:
            # штраф: как далеко ARL1_mc от цели + штраф за превышение ARL0_max
            d_arl1 = abs(cand["ARL1_mc"] - arl1_target) / max(arl1_target, 1e-6)
            penalty_arl0 = 0.0
            if cand["ARL0_mc"] > arl0_max:
                penalty_arl0 = (cand["ARL0_mc"] - arl0_max) / max(arl0_max, 1e-6)

            score = d_arl1 + penalty_arl0

            if score < best_score:
                best_score = score
                best_h = cand["h"]
                best_E = cand["E_mc"]
                best_arl0 = cand["ARL0_mc"]
                best_arl1 = cand["ARL1_mc"]

    print(
        f"[BEST ARL1-target] h={best_h:.3f}, "
        f"ARL0_mc={best_arl0:.1f}, ARL1_mc={best_arl1:.1f}, E={best_E:.3f}"
    )

    # 5) Работа с JSON-справочником
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    new_record = {
        "arl1_target": float(arl1_target),
        "delta_target": float(delta_target),
        "n_runs_mc": int(n_runs_mc),
        "h_grid": {
            "start": float(h_grid[0]),
            "stop": float(h_grid[-1]),
            "step": float(h_grid[1] - h_grid[0]) if len(h_grid) > 1 else None,
            "n_points": int(len(h_grid)),
        },
        "h_mc": float(best_h),
        "E_mc": float(best_E),
        "ARL0_mc": float(best_arl0),
        "ARL1_mc": float(best_arl1),
        "a": None,
        "b": None,
    }

    # 6) Обновление/добавление записи (без дублей по (δ, ARL1_target))
    existing = [
        r for r in data
        if r.get("delta_target") == new_record["delta_target"]
        and r.get("arl1_target") == new_record["arl1_target"]
    ]

    if existing:
        old = existing[0]
        if new_record["E_mc"] > old.get("E_mc", -np.inf):
            data.remove(old)
            data.append(new_record)
            print("[UPDATE ARL1-target] Запись улучшена.")
        else:
            print("[SKIP ARL1-target] Старая запись лучше — не обновляем.")
    else:
        data.append(new_record)
        print("[ADD ARL1-target] Добавлена новая запись.")

    # 7) Аппроксимация h = a + b ln(ARL1_target) для данной δ
    same_delta = [r for r in data if r.get("delta_target") == delta_target]

    if len(same_delta) >= 2:
        X = np.log([r["arl1_target"] for r in same_delta])
        y = [r["h_mc"] for r in same_delta]
        b, a = np.polyfit(X, y, 1)
        for r in same_delta:
            r["a"] = float(a)
            r["b"] = float(b)
        print(f"[FIT ARL1-target] h ≈ {a:.4f} + {b:.4f} ln(ARL1_target)")
    else:
        print("[FIT ARL1-target] Недостаточно точек для аппроксимации.")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[SAVE JSON] {json_path}")

def run_arl0_arl1_delta_experiment(
    delta_target: float,
    arl0_target: float,
    arl1_target: float,
    n_runs_mc: int,
    json_path: Path,
    h_grid: np.ndarray | None = None,
    max_steps: int = 200_000,
    n_workers: int | None = None,

    arl0_tolerance: float = 0.20,
    arl1_tolerance: float = 0.20,

    arl0_max: float | None = None,
    arl1_min_abs: float = 5.0,
):
    """
    Компромиссный поиск h при заданных (δ, ARL0_target, ARL1_target).

    Цель:
        • ARL0 ≈ ARL0_target
        • ARL1 ≈ ARL1_target
        • max E = ARL0 / ARL1
    """

    print(
        f"[mc_compromise] δ={delta_target}, "
        f"ARL0≈{arl0_target}, ARL1≈{arl1_target}"
    )

    # --- 1. h_grid ---
    if h_grid is None:
        h0 = h_from_arl0_delta(arl0_target, delta_target)
        h1 = h_from_arl0_delta(max(arl1_target * 2, 10), delta_target)
        h_min = max(0.3, min(h0, h1) - 0.8)
        h_max = max(h0, h1) + 0.8
        h_grid = np.arange(h_min, h_max + 1e-9, 0.1)

    # --- 2. Допуски ---
    arl0_min = arl0_target * (1 - arl0_tolerance)
    arl0_max_eff = (
        min(arl0_target * (1 + arl0_tolerance), arl0_max)
        if arl0_max is not None else
        arl0_target * (1 + arl0_tolerance)
    )

    arl1_min = max(arl1_target * (1 - arl1_tolerance), arl1_min_abs)
    arl1_max = arl1_target * (1 + arl1_tolerance)

    best = None
    candidates = []

    # --- 3. Перебор h ---
    for h in h_grid:
        arl0_mc, arl1_mc, E_mc = estimate_metrics_mc_parallel(
            delta_target,
            h,
            n_runs_arl0=n_runs_mc,
            n_runs_arl1=n_runs_mc,
            max_steps=max_steps,
            n_workers=n_workers,
        )

        candidates.append((h, arl0_mc, arl1_mc, E_mc))

        if not (arl0_min <= arl0_mc <= arl0_max_eff):
            continue
        if not (arl1_min <= arl1_mc <= arl1_max):
            continue

        if best is None or E_mc > best["E_mc"]:
            best = {
                "h": h,
                "ARL0_mc": arl0_mc,
                "ARL1_mc": arl1_mc,
                "E_mc": E_mc,
            }

    # --- 4. Fallback ---
    if best is None:
        print("[WARN] Нет h, удовлетворяющих обоим условиям. Использую компромисс.")
        best_score = np.inf
        for h, a0, a1, E in candidates:
            score = (
                abs(a0 - arl0_target) / arl0_target +
                abs(a1 - arl1_target) / arl1_target
            )
            if score < best_score:
                best_score = score
                best = {
                    "h": h,
                    "ARL0_mc": a0,
                    "ARL1_mc": a1,
                    "E_mc": E,
                }

    print(
        f"[BEST] h={best['h']:.3f}, "
        f"ARL0={best['ARL0_mc']:.1f}, "
        f"ARL1={best['ARL1_mc']:.1f}, "
        f"E={best['E_mc']:.3f}"
    )

    # --- 5. JSON ---
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    record = {
        "delta_target": delta_target,
        "arl0_target": arl0_target,
        "arl1_target": arl1_target,
        "n_runs_mc": n_runs_mc,
        "h_mc": best["h"],
        "ARL0_mc": best["ARL0_mc"],
        "ARL1_mc": best["ARL1_mc"],
        "E_mc": best["E_mc"],
    }

    data.append(record)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[SAVE JSON] {json_path}")









