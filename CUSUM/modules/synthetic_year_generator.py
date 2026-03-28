import numpy as np
import pandas as pd
from pathlib import Path


SEASON_TO_MONTHS = {
    "Зима": [1, 2],
    "Весна": [3, 4, 5],
    "Лето": [6, 7, 8],
    "Осень": [9, 10, 11],
}

SEASON_INDEX_MAP = {
    "Зима": 1,
    "Весна": 2,
    "Лето": 3,
    "Осень": 4,
}


def generate_synthetic_year(
    source_year: int,
    synthetic_year: int,
    input_department_dir: Path,
    input_road_dir: Path,
    output_department_dir: Path,
    output_road_dir: Path,
    random_seed: int | None = None,
) -> None:

    if random_seed is not None:
        np.random.seed(random_seed)

    def _extract_entity_name(csv_file: Path) -> str:
        stem = csv_file.stem

        if stem.startswith("lambda_0_"):
            tail = stem[len("lambda_0_"):]
        else:
            tail = stem

        parts = tail.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]

        return tail

    def _generate_one_file(lambda_file: Path, entity_type: str, out_dir: Path):

        df_lambdas = pd.read_csv(lambda_file, encoding="utf-8-sig")

        required_cols = {"SEASON", "LAMBDA_FITTER", "LAMBDA_MONTH"}
        if not required_cols.issubset(df_lambdas.columns):
            print(f"⚠️ Пропуск {lambda_file.name}: нет нужных колонок")
            return

        # ===== Проверка всех сезонов =====
        missing_seasons = [
            s for s in SEASON_TO_MONTHS.keys()
            if s not in df_lambdas["SEASON"].values
        ]

        if missing_seasons:
            print(f"❌ Пропуск {lambda_file.name}: нет сезонов {missing_seasons}")
            return

        entity_name = (
            str(df_lambdas["ENTITY_NAME"].iloc[0]).strip()
            if "ENTITY_NAME" in df_lambdas.columns
            else _extract_entity_name(lambda_file)
        )

        all_rows = []

        for season, months in SEASON_TO_MONTHS.items():
            df_s = df_lambdas[df_lambdas["SEASON"] == season]

            lambda_minute = float(df_s["LAMBDA_FITTER"].iloc[0])
            lambda_month = float(df_s["LAMBDA_MONTH"].iloc[0])

            if lambda_minute <= 0 or lambda_month <= 0:
                print(f"❌ Пропуск {lambda_file.name}: некорректные λ ({season})")
                return

            season_index = (
                int(df_s["SEASON_INDEX"].iloc[0])
                if "SEASON_INDEX" in df_s.columns
                else SEASON_INDEX_MAP[season]
            )

            n_events = max(1, int(round(lambda_month)))
            scale = 1.0 / lambda_minute

            for month in months:
                start = pd.Timestamp(year=synthetic_year, month=month, day=1)
                next_month = start + pd.offsets.MonthBegin(1)

                td = np.random.exponential(scale=scale, size=n_events)

                df = pd.DataFrame({"TIME_DIFF_RAW": td})
                df["DELTA_MINUTES_MONTH"] = df["TIME_DIFF_RAW"].cumsum()
                df["START_TIME"] = start + pd.to_timedelta(df["DELTA_MINUTES_MONTH"], unit="m")

                df = df[df["START_TIME"] < next_month]

                if df.empty:
                    continue

                df["SEASON"] = season
                df["SEASON_INDEX"] = season_index
                df["MONTH"] = month
                df["LAMBDA_FITTER"] = lambda_minute
                df["LAMBDA_MONTH"] = lambda_month

                all_rows.append(df)

        if not all_rows:
            print(f"⚠️ Пропуск {lambda_file.name}: нет событий")
            return

        df_all = pd.concat(all_rows, ignore_index=True)
        df_all = df_all.sort_values("START_TIME").reset_index(drop=True)

        df_all["DELTA_TIME"] = df_all["START_TIME"] - df_all["START_TIME"].iloc[0]
        df_all["DELTA_MINUTES"] = df_all["DELTA_TIME"].dt.total_seconds() / 60
        df_all["TIME_DIFF"] = df_all["DELTA_MINUTES"].diff().fillna(0)
        df_all["INDEX"] = df_all.index / max(1, len(df_all) - 1)

        df_all["YEAR"] = synthetic_year
        df_all["SOURCE_YEAR"] = source_year
        df_all["CATEGORY"] = ""
        df_all["ENTITY_TYPE"] = entity_type

        if entity_type == "department":
            df_all["DEPARTMENT"] = entity_name
            df_all["ROAD"] = ""
        else:
            df_all["ROAD"] = entity_name
            df_all["DEPARTMENT"] = "SYNTHETIC"

        df_all = df_all[
            [
                "CATEGORY", "START_TIME", "ROAD", "DEPARTMENT",
                "YEAR", "SOURCE_YEAR",
                "DELTA_TIME", "DELTA_MINUTES", "TIME_DIFF", "INDEX",
                "SEASON", "SEASON_INDEX", "MONTH",
                "LAMBDA_FITTER", "LAMBDA_MONTH",
                "ENTITY_TYPE"
            ]
        ]

        out_dir.mkdir(parents=True, exist_ok=True)

        # ✅ исправленное имя файла
        out_path = out_dir / f"{entity_name}-{synthetic_year}.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"✅ Сгенерирован: {out_path}")

    def _process_dir(src: Path, out: Path, entity_type: str):
        files = sorted(src.glob(f"*_{source_year}.csv"))

        if not files:
            print(f"⚠️ Нет файлов за {source_year} в {src}")
            return

        for f in files:
            try:
                _generate_one_file(f, entity_type, out)
            except Exception as e:
                print(f"❌ Ошибка {f.name}: {e}")

    _process_dir(input_department_dir, output_department_dir, "department")
    _process_dir(input_road_dir, output_road_dir, "road")

    print("\n🎉 Генерация завершена")

def generate_synthetic_year_with_spikes_smooth(
    source_year: int,
    synthetic_year: int,
    input_department_dir: Path,
    input_road_dir: Path,
    output_department_dir: Path,
    output_road_dir: Path,
    delta: float,
    spike_days: int = 20,
    transition_days: int = 5,
    k: float = 2.0,
    random_seed: int | None = None,
) -> None:
    """
    Генерирует синтетические годовые ряды с 4 всплесками
    (по одному в каждом сезоне) на основе файлов lambda_0.

    Если хотя бы один сезон отсутствует, файл пропускается.

    Параметры:
        source_year:
            год исходной статистики lambda_0
        synthetic_year:
            год, для которого создаётся синтетический ряд
        input_department_dir:
            папка с lambda_0 по департаментам
        input_road_dir:
            папка с lambda_0 по дорогам
        output_department_dir:
            папка для сохранения синтетики по департаментам
        output_road_dir:
            папка для сохранения синтетики по дорогам
        delta:
            множитель разладки, λ1 = delta * λ0
        spike_days:
            длительность всплеска в днях
        transition_days:
            длительность плавного перехода
        k:
            скорость экспоненциального перехода
        random_seed:
            seed для воспроизводимой генерации
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    def _extract_entity_name(csv_file: Path) -> str:
        stem = csv_file.stem

        if stem.startswith("lambda_0_"):
            tail = stem[len("lambda_0_"):]
        else:
            tail = stem

        parts = tail.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]

        return tail

    def _generate_one_file(lambda_file: Path, entity_type: str, out_dir: Path) -> None:
        df_lambdas = pd.read_csv(lambda_file, encoding="utf-8-sig")

        required_cols = {"SEASON", "LAMBDA_FITTER", "LAMBDA_MONTH"}
        if not required_cols.issubset(df_lambdas.columns):
            print(f"⚠️ Пропуск {lambda_file.name}: нет нужных колонок")
            return

        missing_seasons = [
            season for season in SEASON_TO_MONTHS.keys()
            if season not in df_lambdas["SEASON"].values
        ]
        if missing_seasons:
            print(f"❌ Пропуск {lambda_file.name}: нет сезонов {missing_seasons}")
            return

        entity_name = (
            str(df_lambdas["ENTITY_NAME"].iloc[0]).strip()
            if "ENTITY_NAME" in df_lambdas.columns and not df_lambdas.empty
            else _extract_entity_name(lambda_file)
        )

        all_rows = []

        for season, months in SEASON_TO_MONTHS.items():
            df_s = df_lambdas[df_lambdas["SEASON"] == season].copy()

            lambda0 = float(df_s["LAMBDA_FITTER"].iloc[0])
            lambda_month = float(df_s["LAMBDA_MONTH"].iloc[0])

            if lambda0 <= 0 or lambda_month <= 0:
                print(f"❌ Пропуск {lambda_file.name}: некорректные λ ({season})")
                return

            season_index = (
                int(df_s["SEASON_INDEX"].iloc[0])
                if "SEASON_INDEX" in df_s.columns
                else SEASON_INDEX_MAP[season]
            )

            lambda1 = lambda0 * delta

            season_start = pd.Timestamp(year=synthetic_year, month=months[0], day=1)

            last_month = months[-1]
            season_end = pd.Timestamp(year=synthetic_year, month=last_month, day=1) + pd.offsets.MonthEnd(1)

            season_minutes = (season_end - season_start).total_seconds() / 60
            n_events = max(1, int(round(lambda_month * len(months))))

            df = pd.DataFrame()
            df["TIME_DIFF"] = np.random.exponential(scale=1 / lambda0, size=n_events)
            df["DELTA_MINUTES"] = df["TIME_DIFF"].cumsum()
            df["START_TIME"] = season_start + pd.to_timedelta(df["DELTA_MINUTES"], unit="m")

            season_mid = season_start + pd.Timedelta(minutes=season_minutes / 2)
            spike_start = season_mid - pd.Timedelta(days=spike_days / 2)
            spike_end = season_mid + pd.Timedelta(days=spike_days / 2)
            transition_start = spike_start - pd.Timedelta(days=transition_days)

            df["SPIKE_FLAG"] = (
                (df["START_TIME"] >= spike_start) &
                (df["START_TIME"] <= spike_end)
            ).astype(int)

            df["TRANSITION_FLAG"] = (
                (df["START_TIME"] >= transition_start) &
                (df["START_TIME"] < spike_start)
            ).astype(int)

            lambda_dynamic = np.full(len(df), lambda0)

            trans_mask = df["TRANSITION_FLAG"] == 1
            if trans_mask.any():
                t_min = (df.loc[trans_mask, "START_TIME"] - transition_start).dt.total_seconds() / 60
                T = transition_days * 1440
                idx = df.index[trans_mask]
                lambda_dynamic[idx] = (
                    lambda0 + (lambda1 - lambda0) * (1 - np.exp(-k * (t_min / T)))
                )

            lambda_dynamic[df["SPIKE_FLAG"] == 1] = lambda1

            df["LAMBDA_DYNAMIC"] = lambda_dynamic
            df["LAMBDA0"] = lambda0
            df["LAMBDA1"] = lambda1
            df["LAMBDA_MONTH"] = lambda_month
            df["SEASON"] = season
            df["SEASON_INDEX"] = season_index

            df["TIME_DIFF"] = [np.random.exponential(scale=1 / lam) for lam in lambda_dynamic]
            df["DELTA_MINUTES"] = df["TIME_DIFF"].cumsum()
            df["START_TIME"] = season_start + pd.to_timedelta(df["DELTA_MINUTES"], unit="m")
            df["MONTH"] = df["START_TIME"].dt.month

            all_rows.append(df)

        if not all_rows:
            print(f"⚠️ Пропуск {lambda_file.name}: нет событий")
            return

        df_all = pd.concat(all_rows, ignore_index=True)
        df_all = df_all.sort_values("START_TIME").reset_index(drop=True)

        df_all["DELTA_TIME"] = df_all["START_TIME"] - df_all["START_TIME"].iloc[0]
        df_all["DELTA_MINUTES"] = df_all["DELTA_TIME"].dt.total_seconds() / 60
        df_all["TIME_DIFF"] = df_all["DELTA_MINUTES"].diff().fillna(0)
        df_all["INDEX"] = df_all.index / max(1, len(df_all) - 1)

        df_all["YEAR"] = synthetic_year
        df_all["SOURCE_YEAR"] = source_year
        df_all["ENTITY_TYPE"] = entity_type

        # Общие пустые поля
        df_all["CATEGORY"] = ""
        df_all["RESPONSIBILITY"] = ""
        df_all["FREIGHT_COUNT"] = ""
        df_all["FREIGHT_TIMEOUT"] = ""
        df_all["PASSENGER_COUNT"] = ""
        df_all["PASSENGER_TIMEOUT"] = ""
        df_all["COMMUTER_COUNT"] = ""
        df_all["COMMUTER_TIMEOUT"] = ""

        if entity_type == "road":
            df_all["ROAD"] = entity_name
            df_all["DEPARTMENT"] = ""
        else:
            df_all["DEPARTMENT"] = entity_name
            df_all["ROAD"] = ""

        df_all = df_all[
            [
                "CATEGORY",
                "START_TIME",
                "ROAD",
                "DEPARTMENT",
                "RESPONSIBILITY",
                "YEAR",
                "SOURCE_YEAR",
                "DELTA_TIME",
                "DELTA_MINUTES",
                "TIME_DIFF",
                "INDEX",
                "SEASON",
                "SEASON_INDEX",
                "MONTH",
                "LAMBDA_DYNAMIC",
                "LAMBDA0",
                "LAMBDA1",
                "LAMBDA_MONTH",
                "SPIKE_FLAG",
                "TRANSITION_FLAG",
                "FREIGHT_COUNT",
                "FREIGHT_TIMEOUT",
                "PASSENGER_COUNT",
                "PASSENGER_TIMEOUT",
                "COMMUTER_COUNT",
                "COMMUTER_TIMEOUT",
                "ENTITY_TYPE",
            ]
        ]

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{entity_name}-{synthetic_year}.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"🔥 Сгенерирован synthetic CSV со всплесками: {out_path}")

    def _process_dir(src: Path, out: Path, entity_type: str) -> None:
        if not src.exists():
            print(f"⚠️ Папка не найдена: {src}")
            return

        files = sorted(src.glob(f"*_{source_year}.csv"))
        if not files:
            print(f"⚠️ Нет файлов за {source_year} в {src}")
            return

        for f in files:
            try:
                _generate_one_file(f, entity_type, out)
            except Exception as e:
                print(f"❌ Ошибка {f.name}: {e}")

    _process_dir(input_department_dir, output_department_dir, "department")
    _process_dir(input_road_dir, output_road_dir, "road")

    print("\n🎉 Генерация synthetic со всплесками завершена.")

def generate_spike_report(
    input_department_dir: Path,
    input_road_dir: Path,
    output_department_dir: Path,
    output_road_dir: Path,
) -> None:
    """
    Формирует отчёты по всплескам для всех synthetic CSV
    из папок департаментов и дорог.

    Для каждого входного CSV создаётся отдельный report CSV.

    В отчёт попадают:
        - сущность (дорога или департамент)
        - тип сущности
        - год synthetic ряда
        - source_year
        - сезон
        - season_index
        - время начала всплеска
        - время окончания всплеска
        - λ0 и λ1 в минуту
        - λ0 и λ1 в пересчёте на месяц
    """

    def _extract_entity_name(csv_file: Path) -> str:
        stem = csv_file.stem
        parts = stem.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return stem

    def _process_one_file(csv_file: Path, out_dir: Path, entity_type: str) -> None:
        df = pd.read_csv(csv_file, encoding="utf-8-sig")

        required_cols = {
            "SEASON",
            "SPIKE_FLAG",
            "START_TIME",
            "LAMBDA0",
            "LAMBDA1",
            "LAMBDA_MONTH",
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"⚠️ Пропуск {csv_file.name}: нет колонок {sorted(missing_cols)}")
            return

        df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
        df = df[df["START_TIME"].notna()].copy()

        if df.empty:
            print(f"⚠️ Пропуск {csv_file.name}: нет валидных START_TIME")
            return

        entity_name = (
            str(df["ROAD"].iloc[0]).strip()
            if entity_type == "road" and "ROAD" in df.columns and str(df["ROAD"].iloc[0]).strip() != ""
            else str(df["DEPARTMENT"].iloc[0]).strip()
            if entity_type == "department" and "DEPARTMENT" in df.columns and str(df["DEPARTMENT"].iloc[0]).strip() != ""
            else _extract_entity_name(csv_file)
        )

        year = int(df["YEAR"].iloc[0]) if "YEAR" in df.columns and pd.notna(df["YEAR"].iloc[0]) else None
        source_year = (
            int(df["SOURCE_YEAR"].iloc[0])
            if "SOURCE_YEAR" in df.columns and pd.notna(df["SOURCE_YEAR"].iloc[0])
            else None
        )

        rows = []

        for season, df_season in df.groupby("SEASON", sort=False):
            spike_mask = pd.to_numeric(df_season["SPIKE_FLAG"], errors="coerce").fillna(0).astype(int) == 1

            if not spike_mask.any():
                continue

            spike_start = df_season.loc[spike_mask, "START_TIME"].min()
            spike_end = df_season.loc[spike_mask, "START_TIME"].max()

            lambda0 = float(df_season["LAMBDA0"].iloc[0])
            lambda1 = float(df_season["LAMBDA1"].iloc[0])
            lambda0_month = float(df_season["LAMBDA_MONTH"].iloc[0])
            lambda1_month = lambda0_month * (lambda1 / lambda0) if lambda0 != 0 else None

            season_index = (
                int(df_season["SEASON_INDEX"].iloc[0])
                if "SEASON_INDEX" in df_season.columns and pd.notna(df_season["SEASON_INDEX"].iloc[0])
                else None
            )

            rows.append({
                "ENTITY_TYPE": entity_type,
                "ENTITY_NAME": entity_name,
                "YEAR": year,
                "SOURCE_YEAR": source_year,
                "SEASON": season,
                "SEASON_INDEX": season_index,
                "SPIKE_START": spike_start,
                "SPIKE_END": spike_end,
                "LAMBDA0": lambda0,
                "LAMBDA1": lambda1,
                "LAMBDA0_MONTH": lambda0_month,
                "LAMBDA1_MONTH": lambda1_month,
            })

        if not rows:
            print(f"⚠️ Пропуск {csv_file.name}: всплески не найдены")
            return

        report_df = pd.DataFrame(rows)

        if "SEASON_INDEX" in report_df.columns:
            report_df = report_df.sort_values(
                by=["SEASON_INDEX", "SEASON"],
                kind="stable"
            ).reset_index(drop=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"report_{csv_file.stem}.csv"
        report_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"📄 Мини-отчёт по всплескам сохранён: {out_path}")

    def _process_dir(src: Path, out: Path, entity_type: str) -> None:
        if not src.exists():
            print(f"⚠️ Папка не найдена: {src}")
            return

        files = sorted(src.glob("*.csv"))
        if not files:
            print(f"⚠️ Нет CSV-файлов в папке: {src}")
            return

        print(f"\n📂 Формирование spike-report для папки: {src}")

        for f in files:
            try:
                _process_one_file(f, out, entity_type)
            except Exception as e:
                print(f"❌ Ошибка обработки {f.name}: {e}")

    _process_dir(input_department_dir, output_department_dir, "department")
    _process_dir(input_road_dir, output_road_dir, "road")

    print("\n🎉 Формирование отчётов по всплескам завершено.")


