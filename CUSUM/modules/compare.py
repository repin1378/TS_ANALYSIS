from __future__ import annotations

"""
compare.py — сравнение исторических данных (processed) с результатами CUSUM.
Генерирует HTML-отчёты о срабатываниях CUSUM по каждой дороге / подразделению.

Публичные функции:
    compare_road_by_year(...)
    compare_department_by_year(...)
"""

import html
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================
# Константы
# ============================================================

_SEASON_MAP: dict[int, str] = {
    12: "Зима", 1: "Зима",  2: "Зима",
    3:  "Весна", 4: "Весна", 5: "Весна",
    6:  "Лето",  7: "Лето",  8: "Лето",
    9:  "Осень", 10: "Осень", 11: "Осень",
}

# Месяц начала каждого сезона
_SEASON_START_MONTH: dict[str, int] = {
    "Зима":  12,
    "Весна": 3,
    "Лето":  6,
    "Осень": 9,
}

_CATEGORIES = [1, 2, 3]

_TEMPLATE_DIR = Path(__file__).parent.parent / "html_template"

# Маппинг кодов подразделений → отображаемые названия (только для HTML)
_DEPT_DISPLAY: dict[str, str] = {
    "CT":  "ЦТ",
    "CV":  "ЦВ",
    "CSH": "ЦШ",
}


# ============================================================
# Вспомогательные утилиты
# ============================================================

def _dept_display(name: str) -> str:
    """Возвращает отображаемое название подразделения для HTML-отчёта.

        "CT"  → "ЦТ"
        "CV"  → "ЦВ"
        "CSH" → "ЦШ"
        всё остальное → без изменений
    """
    return _DEPT_DISPLAY.get(str(name).strip(), name)


def _extract_entity_name(filepath: Path) -> str:
    """
    Извлекает имя сущности из имени файла (убирает суффикс -YYYY / _YYYY).

        Дальневосточная-2025.csv  →  Дальневосточная
        CSH-2025.csv              →  CSH
    """
    return re.sub(r"[-_ ]\d{4}$", "", filepath.stem)


def _find_historical_file(
    entity_name: str,
    hist_dir: Path,
    hist_year: int,
) -> Optional[Path]:
    """
    Ищет исторический CSV в hist_dir по шаблону:
        {entity_name}_{hist_year}.csv  или  {entity_name}-{hist_year}.csv
    """
    for sep in ("_", "-", " "):
        candidate = hist_dir / f"{entity_name}{sep}{hist_year}.csv"
        if candidate.exists():
            return candidate
    return None


def _find_cusum_full_csvs(directory: Path) -> list[Path]:
    """
    Возвращает полные CUSUM-файлы (с колонкой CUSUM_ALARM),
    исключая *_cusum_events.csv и batch_summary.csv.
    """
    return sorted(
        p for p in Path(directory).glob("*.csv")
        if not p.name.endswith("_cusum_events.csv")
        and p.name != "batch_summary.csv"
    )


def _find_events_csv(cusum_full_path: Path) -> Optional[Path]:
    """
    По пути полного CUSUM-файла находит соответствующий events-файл.

        {stem}.csv              →  {stem}_cusum_events.csv
        {stem}_cusum_full.csv   →  {stem}_cusum_events.csv
    """
    stem = cusum_full_path.stem
    if stem.endswith("_cusum_full"):
        stem = stem[: -len("_cusum_full")]
    events_path = cusum_full_path.parent / f"{stem}_cusum_events.csv"
    return events_path if events_path.exists() else None


def _season_start(alarm_dt: pd.Timestamp, season: str) -> pd.Timestamp:
    """
    Возвращает дату начала сезона для аларма.

    Особый случай: Зима начинается 1 декабря предыдущего года,
    если аларм приходится на январь или февраль (month < 3).
    """
    start_month = _SEASON_START_MONTH[season]
    year = alarm_dt.year

    if season == "Зима" and alarm_dt.month < 3:
        # Январь/февраль — зима началась в декабре прошлого года
        year = alarm_dt.year - 1

    return pd.Timestamp(year=year, month=start_month, day=1)


def _filter_period(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Фильтрует DataFrame по START_TIME в диапазоне [start, end] включительно."""
    mask = (df["START_TIME"] >= start) & (df["START_TIME"] <= end)
    return df.loc[mask]


def _prepare_hist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет MONTH и SEASON в исторический DataFrame (если их нет).
    """
    df = df.copy()
    df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
    if "MONTH" not in df.columns:
        df["MONTH"] = df["START_TIME"].dt.month
    if "SEASON" not in df.columns:
        df["SEASON"] = df["MONTH"].map(_SEASON_MAP)
    return df


def _fmt_train_hours(v: float) -> str:
    """
    Форматирует продолжительность в поездо-часах с пробелом как разделителем тысяч.

        31246.28  →  "31 246.3 поездо-ч"
        26.0      →  "26 поездо-ч"
    """
    hours = float(v or 0.0)
    if abs(hours - round(hours)) < 0.05:
        int_v = int(round(hours))
        formatted = f"{int_v:,}".replace(",", "\u00a0")
        return f"{formatted}\u00a0поездо-ч"

    formatted = f"{hours:,.1f}".replace(",", "\u00a0")
    return f"{formatted}\u00a0поездо-ч"


def _parse_train_hours(value: object) -> float:
    """
    Преобразует значение таймаута к числу поездо-часов.

    Поддерживает форматы:
      - "0,67ч"  →  0.67
      - "1.18 ч" →  1.18
      - "11"     →  11.0
      - 1.5      →  1.5   (int/float/numpy scalar)
      - NaN / None / "" / "nan" / "none"  →  0.0
    """
    # 1. NaN / None / pd.NA / np.nan
    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    # 2. Числовые скаляры (int, float, np.int64, np.float64 и т.д.)
    try:
        fv = float(value)  # type: ignore[arg-type]
        if not (fv != fv):  # проверка на NaN через self-inequality
            return fv
        return 0.0
    except (TypeError, ValueError):
        pass

    # 3. Строковые значения
    s = str(value).strip().replace("\xa0", "").replace("\u00a0", "").replace(" ", "")
    if not s or s.lower() in ("nan", "none", "null", "na", "<na>"):
        return 0.0

    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
    if not match:
        return 0.0

    return float(match.group(0).replace(",", "."))


# ============================================================
# HTML-рендеринг
# ============================================================

def _pct_badge(curr: int, prev: int) -> str:
    """
    Возвращает HTML-бейдж процентного изменения.

        curr=87, prev=64  →  '<span class="pct up">▲ +35.9%</span>'
        curr=50, prev=64  →  '<span class="pct down">▼ −21.9%</span>'
        curr=64, prev=64  →  '<span class="pct zero">→ 0.0%</span>'
        curr=5,  prev=0   →  '<span class="pct up">▲ новые</span>'
    """
    if prev == 0 and curr == 0:
        return '<span class="pct zero">→ 0.0%</span>'
    if prev == 0:
        return '<span class="pct up">▲ новый фактор</span>'
    pct = (curr - prev) / prev * 100
    if abs(pct) < 0.05:
        return '<span class="pct zero">→ 0.0%</span>'
    if pct > 0:
        return f'<span class="pct up">▲ +{pct:.1f}%</span>'
    return f'<span class="pct down">▼ −{abs(pct):.1f}%</span>'


def _pct_growth_key(t: tuple[str, int, int]) -> float:
    """Ключ сортировки: процент роста по убыванию.

    Новые факторы (prev=0, curr>0) → +inf (идут первыми).
    Исчезнувшие факторы (curr=0, prev>0) → -100%.
    Оба нуля → 0%.
    """
    _, curr, prev = t
    if prev == 0:
        return float("inf") if curr > 0 else 0.0
    return (curr - prev) / prev * 100


def _render_comparison_rows(
    data: list[tuple[str, int, int]],
    curr_total: int,
    *,
    bold_label: bool = False,
    max_rows: int = 15,
) -> str:
    """
    Генерирует HTML <tr> строки для таблицы сравнения.

    Parameters
    ----------
    data       : список (label, curr_count, prev_count)
    curr_total : суммарное количество текущих событий (для расчёта доли)
    bold_label : обернуть ячейку label в <strong> (для категорий)
    max_rows   : максимум строк

    Сортировка: сначала факторы с наибольшим % роста (curr/prev - 1),
    новые факторы (prev=0) идут первыми.
    """
    sorted_data = sorted(data, key=_pct_growth_key, reverse=True)[:max_rows]

    if not sorted_data:
        return ""

    max_curr = max(t[1] for t in sorted_data) or 1
    rows: list[str] = []

    for label, curr, prev in sorted_data:
        safe_label = html.escape(str(label))
        label_html = f"<strong>{safe_label}</strong>" if bold_label else safe_label

        # Доля от суммы текущего периода
        share_pct = round(curr / curr_total * 100) if curr_total > 0 else 0
        # Ширина бара относительно максимального curr
        bar_w = round(curr / max_curr * 100) if max_curr > 0 else 0

        badge = _pct_badge(curr, prev)

        rows.append(
            f"            <tr>\n"
            f"              <td>{label_html}</td>\n"
            f"              <td class=\"num\">{curr}</td>\n"
            f"              <td class=\"num\">{prev}</td>\n"
            f"              <td class=\"bar-cell\">\n"
            f"                <div class=\"bar-wrap\">\n"
            f"                  <div class=\"bar-bg\">"
            f"<div class=\"bar-fill curr\" style=\"width:{bar_w}%\"></div></div>\n"
            f"                  <span class=\"bar-num\">{share_pct}%</span>\n"
            f"                </div>\n"
            f"              </td>\n"
            f"              <td class=\"num\">{badge}</td>\n"
            f"            </tr>"
        )

    return "\n".join(rows)


def _render_downtime_card(
    label: str,
    icon: str,
    curr: float,
    prev: float,
    cusum_year: str,
    hist_year: str,
) -> str:
    """
    Генерирует полный <div class="dt-card">...</div> для раздела простоев в поездо-часах.
    """
    curr_fmt = _fmt_train_hours(curr)
    prev_fmt = _fmt_train_hours(prev)

    if prev == 0 and curr == 0:
        change_html = '<span class="val-zero">→ 0%</span>'
    elif prev == 0:
        change_html = '<span class="val-up">▲ новый фактор</span>'
    else:
        pct = (curr - prev) / prev * 100
        if abs(pct) < 0.05:
            change_html = '<span class="val-zero">→ 0.0%</span>'
        elif pct > 0:
            change_html = f'<span class="val-up">▲ +{pct:.1f}%</span>'
        else:
            change_html = f'<span class="val-down">▼ −{abs(pct):.1f}%</span>'

    return (
        f'        <div class="dt-card">\n'
        f'          <div class="dt-title">{icon} {html.escape(label)}</div>\n'
        f'          <div class="dt-row">\n'
        f'            <span class="dt-year">{html.escape(cusum_year)}</span>\n'
        f'            <span class="dt-val">{curr_fmt}</span>\n'
        f'          </div>\n'
        f'          <div class="dt-row">\n'
        f'            <span class="dt-year">{html.escape(hist_year)}</span>\n'
        f'            <span class="dt-val">{prev_fmt}</span>\n'
        f'          </div>\n'
        f'          <div class="dt-divider"></div>\n'
        f'          <div class="dt-change">\n'
        f'            <span class="label">Изменение</span>\n'
        f'            {change_html}\n'
        f'          </div>\n'
        f'        </div>'
    )


# ============================================================
# Загрузка шаблона
# ============================================================

def _load_template(entity_type: str) -> str:
    """
    Загружает HTML-шаблон в зависимости от типа сущности.

        entity_type='road'       →  road_alarm_report.html
        entity_type='department' →  department_alarm_report.html
    """
    name = "road_alarm_report.html" if entity_type == "road" else "department_alarm_report.html"
    path = _TEMPLATE_DIR / name
    return path.read_text(encoding="utf-8")


# ============================================================
# Вспомогательные функции для подготовки данных периода
# ============================================================

def _pct_change(new: float, old: float) -> Optional[float]:
    """Процентное изменение. None если old=0."""
    if old == 0:
        return None
    return round((new - old) / old * 100, 1)


def _count_by_dim(
    df: pd.DataFrame,
    col: str,
) -> dict[str, int]:
    """Группировка по колонке col → словарь {значение: количество}."""
    if col not in df.columns or df.empty:
        return {}
    return df.groupby(col).size().to_dict()


def _downtime_sum(
    df: pd.DataFrame,
    cnt_col: str,
    tmt_col: str,
    *,
    label: str = "",
    verbose: bool = False,
) -> float:
    """Суммарный простой в поездо-часах = SUM(COUNT * TIMEOUT).

    В исходных CSV поля *_TIMEOUT могут храниться как:
      - строки вида "0,67ч" или "1.18ч"
      - числа (int или float)
      - пустые строки / NaN — считаются как 0

    Если колонки отсутствуют — возвращает 0.0 и при verbose=True печатает предупреждение.
    """
    tag = f" [{label}]" if label else ""

    if df.empty:
        return 0.0

    missing = [c for c in (cnt_col, tmt_col) if c not in df.columns]
    if missing:
        if verbose:
            print(
                f"    [WARN]{tag} колонки {missing} не найдены "
                f"(доступны: {sorted(df.columns.tolist())})"
            )
        return 0.0

    cnt = pd.to_numeric(df[cnt_col], errors="coerce").fillna(0.0)
    tmt = df[tmt_col].apply(_parse_train_hours)

    result = float((cnt * tmt).sum())

    if verbose and result == 0.0:
        cnt_nonzero = int((cnt > 0).sum())
        tmt_nonzero = int((tmt > 0).sum())
        sample_cnt = df[cnt_col].dropna().head(3).tolist()
        sample_tmt = df[tmt_col].dropna().head(3).tolist()
        print(
            f"    [DEBUG]{tag} результат=0 | строк={len(df)} | "
            f"cnt>0={cnt_nonzero} dtype={df[cnt_col].dtype} примеры={sample_cnt} | "
            f"tmt>0={tmt_nonzero} dtype={df[tmt_col].dtype} примеры={sample_tmt}"
        )

    return result


# ============================================================
# Генерация HTML одного аларма
# ============================================================

def _generate_alarm_html(
    *,
    entity_name: str,
    entity_type: str,
    alarm_idx: int,           # 1-based
    alarm_row: pd.Series,     # строка из events CSV
    cusum_df: pd.DataFrame,   # полный CUSUM-файл (уже pd.to_datetime START_TIME)
    hist_df: pd.DataFrame,    # исторический (уже подготовлен, может быть пустым)
    cusum_year: int,
    hist_year: int,
    top_n: int,
    template: str,
) -> str:
    """
    Заполняет HTML-шаблон данными для одного аларма и возвращает строку HTML.
    """
    alarm_dt: pd.Timestamp = pd.to_datetime(alarm_row["START_TIME"])
    season: str = str(alarm_row.get("SEASON", ""))

    # ── временные границы ────────────────────────────────────────────────────
    season_start_curr = _season_start(alarm_dt, season)

    # Аналогичная дата и начало сезона в предыдущем году
    analogous_dt = alarm_dt.replace(year=alarm_dt.year - 1)
    season_start_prev = season_start_curr.replace(year=season_start_curr.year - 1)

    # ── фильтрация данных по периодам ────────────────────────────────────────
    curr_df = _filter_period(cusum_df, season_start_curr, alarm_dt)
    prev_df = _filter_period(hist_df, season_start_prev, analogous_dt) if not hist_df.empty else hist_df

    curr_events = len(curr_df)
    prev_events = len(prev_df)

    # ── CATEGORY rows ─────────────────────────────────────────────────────────
    curr_cat = _count_by_dim(curr_df, "CATEGORY")
    prev_cat = _count_by_dim(prev_df, "CATEGORY")
    cat_data = [
        (str(c), curr_cat.get(c, 0), prev_cat.get(c, 0))
        for c in _CATEGORIES
    ]
    category_rows = _render_comparison_rows(cat_data, curr_events, bold_label=True)

    # ── REASON rows ───────────────────────────────────────────────────────────
    curr_reason = _count_by_dim(curr_df, "REASON")
    prev_reason = _count_by_dim(prev_df, "REASON")
    all_reasons = sorted(set(curr_reason) | set(prev_reason))
    reason_data_full = [
        (r, curr_reason.get(r, 0), prev_reason.get(r, 0))
        for r in all_reasons
    ]
    # Отбираем топ top_n по abs_delta, затем сортируем по abs_delta desc
    reason_data_sorted = sorted(
        reason_data_full,
        key=lambda t: abs(t[1] - t[2]),
        reverse=True,
    )[:top_n]
    reason_data_sorted.sort(key=lambda t: t[1] - t[2], reverse=True)
    reason_rows = _render_comparison_rows(reason_data_sorted, curr_events)

    # ── CROSS-DIM rows (DEPARTMENT for road / ROAD for department) ───────────
    cross_col = "DEPARTMENT" if entity_type == "road" else "ROAD"
    curr_cross = _count_by_dim(curr_df, cross_col)
    prev_cross = _count_by_dim(prev_df, cross_col)
    all_cross = sorted(set(curr_cross) | set(prev_cross))
    # Для дорог: метки — названия подразделений → применяем display-маппинг
    cross_data = [
        (
            _dept_display(v) if entity_type == "road" else v,
            curr_cross.get(v, 0),
            prev_cross.get(v, 0),
        )
        for v in all_cross
    ]
    cross_dim_rows = _render_comparison_rows(cross_data, curr_events)

    # ── RESPONSIBILITY rows ───────────────────────────────────────────────────
    curr_resp = _count_by_dim(curr_df, "RESPONSIBILITY")
    prev_resp = _count_by_dim(prev_df, "RESPONSIBILITY")
    all_resp = sorted(set(curr_resp) | set(prev_resp))
    resp_data = [
        (r, curr_resp.get(r, 0), prev_resp.get(r, 0))
        for r in all_resp
    ]
    responsibility_rows = _render_comparison_rows(resp_data, curr_events)

    # ── KPI ───────────────────────────────────────────────────────────────────
    total_pct = _pct_change(curr_events, prev_events)
    if total_pct is None:
        total_pct_fmt = "н/д"
        kpi_total_class = "neutral"
    elif total_pct > 0:
        total_pct_fmt = f"+{total_pct:.1f}%"
        kpi_total_class = "danger"
    elif total_pct < 0:
        total_pct_fmt = f"{total_pct:.1f}%"
        kpi_total_class = "success"
    else:
        total_pct_fmt = "0.0%"
        kpi_total_class = "neutral"

    # ── Downtime cards ────────────────────────────────────────────────────────
    cusum_year_str = str(cusum_year)
    hist_year_str = str(hist_year)

    freight_curr  = _downtime_sum(curr_df, "FREIGHT_COUNT",   "FREIGHT_TIMEOUT",   label="грузовые curr",  verbose=True)
    freight_prev  = _downtime_sum(prev_df, "FREIGHT_COUNT",   "FREIGHT_TIMEOUT",   label="грузовые prev",  verbose=True)
    passenger_curr = _downtime_sum(curr_df, "PASSENGER_COUNT", "PASSENGER_TIMEOUT", label="пасс curr",      verbose=True)
    passenger_prev = _downtime_sum(prev_df, "PASSENGER_COUNT", "PASSENGER_TIMEOUT", label="пасс prev",      verbose=True)
    commuter_curr  = _downtime_sum(curr_df, "COMMUTER_COUNT",  "COMMUTER_TIMEOUT",  label="пригор curr",    verbose=True)
    commuter_prev  = _downtime_sum(prev_df, "COMMUTER_COUNT",  "COMMUTER_TIMEOUT",  label="пригор prev",    verbose=True)

    freight_card = _render_downtime_card(
        "Грузовые перевозки", "🚂",
        freight_curr, freight_prev, cusum_year_str, hist_year_str,
    )
    passenger_card = _render_downtime_card(
        "Пассажирские перевозки", "🚆",
        passenger_curr, passenger_prev, cusum_year_str, hist_year_str,
    )
    commuter_card = _render_downtime_card(
        "Пригородные перевозки", "🚋",
        commuter_curr, commuter_prev, cusum_year_str, hist_year_str,
    )

    # ── CUSUM параметры аларма ────────────────────────────────────────────────
    delta_hat = float(alarm_row.get("DELTA_HAT", 0.0))
    cusum_s   = float(alarm_row.get("CUSUM_S", 0.0))
    h_val     = float(alarm_row.get("H", 0.0))

    # ── Форматирование строк ──────────────────────────────────────────────────
    alarm_num_str = f"{alarm_idx:02d}"
    alarm_dt_fmt  = alarm_dt.strftime("%d.%m.%Y") + "&nbsp;" + alarm_dt.strftime("%H:%M:%S")

    season_start_curr_fmt = season_start_curr.strftime("%d.%m.%Y")
    analogous_dt_fmt      = analogous_dt.strftime("%d.%m.%Y")
    season_start_prev_fmt = season_start_prev.strftime("%d.%m.%Y")

    curr_period_str = (
        season_start_curr.strftime("%d.%m") + " – " +
        alarm_dt.strftime("%d.%m.%Y")
    )
    prev_period_str = (
        season_start_prev.strftime("%d.%m") + " – " +
        analogous_dt.strftime("%d.%m.%Y")
    )

    # Отображаемое имя сущности (для департаментов применяем маппинг кодов)
    display_name = (
        entity_name if entity_type == "road" else _dept_display(entity_name)
    )

    subtitle = (
        f"Дорога: {html.escape(display_name)} · Сезон: {html.escape(season)}"
        if entity_type == "road"
        else f"Подразделение: {html.escape(display_name)} · Сезон: {html.escape(season)}"
    )

    footer_text = (
        f"Сгенерировано автоматически · CUSUM Alarm Report · "
        f"{html.escape(display_name)} · {alarm_dt.strftime('%d.%m.%Y %H:%M:%S')}"
    )

    # ── Подстановка в шаблон ─────────────────────────────────────────────────
    replacements = {
        "{{ENTITY_NAME}}":           html.escape(display_name),
        "{{SUBTITLE}}":              subtitle,
        "{{ALARM_NUMBER}}":          alarm_num_str,
        "{{ALARM_DATETIME_FMT}}":    alarm_dt_fmt,
        "{{ALARM_SEASON}}":          html.escape(season),
        "{{SEASON_START_CURR_FMT}}": season_start_curr_fmt,
        "{{ANALOGOUS_DATE_FMT}}":    analogous_dt_fmt,
        "{{SEASON_START_PREV_FMT}}": season_start_prev_fmt,
        "{{DELTA_HAT_FMT}}":         f"{delta_hat:.2f}",
        "{{CUSUM_S_FMT}}":           f"{cusum_s:.2f}",
        "{{H_FMT}}":                 f"{h_val:.2f}",
        "{{CURR_EVENTS}}":           str(curr_events),
        "{{PREV_EVENTS}}":           str(prev_events),
        "{{TOTAL_PCT_FMT}}":         total_pct_fmt,
        "{{TOTAL_PCT_SUB}}":         f"{curr_events} vs {prev_events} событий",
        "{{KPI_TOTAL_CLASS}}":       kpi_total_class,
        "{{CURR_PERIOD_STR}}":       curr_period_str,
        "{{PREV_PERIOD_STR}}":       prev_period_str,
        "{{CUSUM_YEAR}}":            cusum_year_str,
        "{{HIST_YEAR}}":             hist_year_str,
        "{{CATEGORY_ROWS}}":         category_rows,
        "{{REASON_ROWS}}":           reason_rows,
        "{{CROSS_DIM_ROWS}}":        cross_dim_rows,
        "{{RESPONSIBILITY_ROWS}}":   responsibility_rows,
        "{{FREIGHT_CARD}}":          freight_card,
        "{{PASSENGER_CARD}}":        passenger_card,
        "{{COMMUTER_CARD}}":         commuter_card,
        "{{FOOTER_TEXT}}":           footer_text,
    }

    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    return result


# ============================================================
# Обработка одной сущности
# ============================================================

def _process_entity(
    cusum_path: Path,
    hist_df: pd.DataFrame,
    entity_name: str,
    entity_type: str,
    out_entity_dir: Path,
    cusum_year: int,
    hist_year: int,
    top_n: int,
    verbose: bool,
) -> list[dict]:
    """
    Для каждого аларма в events CSV генерирует и сохраняет HTML-отчёт.

    Возвращает список summary-dict (по одному на аларм).
    """
    events_path = _find_events_csv(cusum_path)
    if events_path is None:
        if verbose:
            print(f"    [WARN] events-файл не найден для {cusum_path.name}")
        return []

    try:
        events_df = pd.read_csv(events_path, encoding="utf-8-sig")
    except Exception:
        if verbose:
            print(f"    [WARN] events-файл пуст или повреждён: {events_path.name}")
        return []
    if events_df.empty:
        if verbose:
            print(f"    [WARN] events-файл не содержит алармов: {events_path.name}")
        return []

    events_df["START_TIME"] = pd.to_datetime(events_df["START_TIME"], errors="coerce")
    events_df = events_df.sort_values("START_TIME").reset_index(drop=True)

    # Загрузка полного CUSUM-файла
    cusum_df = pd.read_csv(cusum_path, encoding="utf-8-sig")
    cusum_df["START_TIME"] = pd.to_datetime(cusum_df["START_TIME"], errors="coerce")

    # Только события с CUSUM_ALARM=True (если колонка есть)
    if "CUSUM_ALARM" in cusum_df.columns:
        # Для фильтрации периода используем все события (не только аларм)
        pass  # cusum_df содержит все строки, нужны все для фильтрации

    template = _load_template(entity_type)
    out_entity_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    alarm_dates: list[str] = []

    for alarm_idx, alarm_row in events_df.iterrows():
        alarm_num = int(alarm_idx) + 1  # 1-based
        alarm_dt = alarm_row["START_TIME"]
        if pd.isna(alarm_dt):
            continue

        alarm_date_str = alarm_dt.strftime("%Y-%m-%d")
        alarm_dates.append(alarm_date_str)

        out_filename = f"alarm_{alarm_num:02d}_{alarm_date_str}.html"
        out_path = out_entity_dir / out_filename

        try:
            html_content = _generate_alarm_html(
                entity_name=entity_name,
                entity_type=entity_type,
                alarm_idx=alarm_num,
                alarm_row=alarm_row,
                cusum_df=cusum_df,
                hist_df=hist_df,
                cusum_year=cusum_year,
                hist_year=hist_year,
                top_n=top_n,
                template=template,
            )
            out_path.write_text(html_content, encoding="utf-8")
            if verbose:
                print(f"      Аларм {alarm_num:02d} [{alarm_date_str}] → {out_filename}")
        except Exception as exc:
            print(f"    [ERROR] Аларм {alarm_num:02d} {alarm_date_str}: {exc}")
            continue

    summary_rows.append({
        "entity_name":  entity_name,
        "entity_type":  entity_type,
        "hist_year":    hist_year,
        "cusum_year":   cusum_year,
        "alarm_count":  len(alarm_dates),
        "alarm_dates":  "; ".join(alarm_dates),
    })

    return summary_rows


# ============================================================
# Публичный API
# ============================================================

def compare_road_by_year(
    *,
    historical_road_dir: Path,
    historical_year: int,
    cusum_road_dir: Path,
    out_dir: Path,
    top_n_reasons: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Для каждой дороги в cusum_road_dir:
      - находит соответствующий исторический CSV в historical_road_dir
      - для каждого CUSUM-аларма генерирует HTML-отчёт
      - сохраняет в out_dir/{entity_name}/alarm_NN_YYYY-MM-DD.html

    Возвращает сводный DataFrame (также сохраняется в out_dir/compare_summary.csv).
    """
    historical_road_dir = Path(historical_road_dir)
    cusum_road_dir      = Path(cusum_road_dir)
    out_dir             = Path(out_dir)

    cusum_files = _find_cusum_full_csvs(cusum_road_dir)
    if not cusum_files:
        if verbose:
            print(f"[WARN] Нет CUSUM-файлов в {cusum_road_dir}")
        return pd.DataFrame()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPARE ROADS — файлов к обработке: {len(cusum_files)}")
        print(f"{'='*60}\n")

    summary_rows: list[dict] = []

    for cusum_path in cusum_files:
        entity_name = _extract_entity_name(cusum_path)
        if verbose:
            print(f"  Дорога: {entity_name}")

        # Исторические данные
        hist_path = _find_historical_file(entity_name, historical_road_dir, historical_year)
        if hist_path is not None:
            hist_df = _prepare_hist(
                pd.read_csv(hist_path, encoding="utf-8-sig")
            )
            if verbose:
                print(f"    Исторический файл: {hist_path.name} ({len(hist_df)} строк)")
        else:
            hist_df = pd.DataFrame()
            if verbose:
                print(f"    [WARN] Исторический файл не найден: "
                      f"{entity_name}_{historical_year}.csv")

        # Получаем cusum_year из файла
        tmp = pd.read_csv(cusum_path, encoding="utf-8-sig", nrows=5)
        cusum_year = (
            int(tmp["YEAR"].mode()[0]) if "YEAR" in tmp.columns else historical_year + 1
        )

        out_entity_dir = out_dir / entity_name
        rows = _process_entity(
            cusum_path=cusum_path,
            hist_df=hist_df,
            entity_name=entity_name,
            entity_type="road",
            out_entity_dir=out_entity_dir,
            cusum_year=cusum_year,
            hist_year=historical_year,
            top_n=top_n_reasons,
            verbose=verbose,
        )
        summary_rows.extend(rows)

    return _save_summary(summary_rows, out_dir, verbose)


def compare_department_by_year(
    *,
    historical_dept_dir: Path,
    historical_year: int,
    cusum_dept_dir: Path,
    out_dir: Path,
    top_n_reasons: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    То же самое, что compare_road_by_year, но для подразделений
    (cross-dimension = ROAD вместо DEPARTMENT).

    Для каждого подразделения в cusum_dept_dir:
      - находит соответствующий исторический CSV в historical_dept_dir
      - для каждого CUSUM-аларма генерирует HTML-отчёт
      - сохраняет в out_dir/{entity_name}/alarm_NN_YYYY-MM-DD.html

    Возвращает сводный DataFrame (также сохраняется в out_dir/compare_summary.csv).
    """
    historical_dept_dir = Path(historical_dept_dir)
    cusum_dept_dir      = Path(cusum_dept_dir)
    out_dir             = Path(out_dir)

    cusum_files = _find_cusum_full_csvs(cusum_dept_dir)
    if not cusum_files:
        if verbose:
            print(f"[WARN] Нет CUSUM-файлов в {cusum_dept_dir}")
        return pd.DataFrame()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPARE DEPARTMENTS — файлов к обработке: {len(cusum_files)}")
        print(f"{'='*60}\n")

    summary_rows: list[dict] = []

    for cusum_path in cusum_files:
        entity_name = _extract_entity_name(cusum_path)
        if verbose:
            print(f"  Подразделение: {entity_name}")

        hist_path = _find_historical_file(entity_name, historical_dept_dir, historical_year)
        if hist_path is not None:
            hist_df = _prepare_hist(
                pd.read_csv(hist_path, encoding="utf-8-sig")
            )
            if verbose:
                print(f"    Исторический файл: {hist_path.name} ({len(hist_df)} строк)")
        else:
            hist_df = pd.DataFrame()
            if verbose:
                print(f"    [WARN] Исторический файл не найден: "
                      f"{entity_name}_{historical_year}.csv")

        tmp = pd.read_csv(cusum_path, encoding="utf-8-sig", nrows=5)
        cusum_year = (
            int(tmp["YEAR"].mode()[0]) if "YEAR" in tmp.columns else historical_year + 1
        )

        out_entity_dir = out_dir / entity_name
        rows = _process_entity(
            cusum_path=cusum_path,
            hist_df=hist_df,
            entity_name=entity_name,
            entity_type="department",
            out_entity_dir=out_entity_dir,
            cusum_year=cusum_year,
            hist_year=historical_year,
            top_n=top_n_reasons,
            verbose=verbose,
        )
        summary_rows.extend(rows)

    return _save_summary(summary_rows, out_dir, verbose)


# ============================================================
# Вспомогательная функция сохранения сводного CSV
# ============================================================

def _save_summary(
    summary_rows: list[dict],
    out_dir: Path,
    verbose: bool,
) -> pd.DataFrame:
    """Сохраняет compare_summary.csv и возвращает DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["entity_name", "entity_type", "hist_year", "cusum_year",
                 "alarm_count", "alarm_dates"],
    ) if summary_rows else pd.DataFrame(
        columns=["entity_name", "entity_type", "hist_year", "cusum_year",
                 "alarm_count", "alarm_dates"]
    )

    summary_path = out_dir / "compare_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if verbose:
        total_alarms = int(summary_df["alarm_count"].sum()) if not summary_df.empty else 0
        print(f"\n{'='*60}")
        print(f"  COMPARE ЗАВЕРШЁН")
        print(f"  Сущностей обработано: {len(summary_df)}")
        print(f"  Алармов всего:        {total_alarms}")
        print(f"  Сводный CSV:          {summary_path}")
        print(f"{'='*60}\n")

    return summary_df
