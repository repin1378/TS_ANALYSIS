from pathlib import Path
import pandas as pd
import numpy as np
from fitter import Fitter
from scipy.stats import kstest


def _month_to_season(m: int) -> str:
    if m in (3, 4, 5):
        return "Весна"
    if m in (6, 7, 8):
        return "Лето"
    if m in (9, 10, 11):
        return "Осень"
    return "Зима"


def estimate_lambda_for_season(
    by_department_year_dir: Path,
    by_road_year_dir: Path,
    output_department_dir: Path,
    output_road_dir: Path,
    min_points: int = 30,
    ks_alpha: float = 0.05,
) -> None:
    """
    Оценивает λ₀ экспоненциального распределения отдельно по сезонам
    для CSV-файлов из папок by_department_year и by_road_year.

    Для каждого исходного файла формируется один CSV-отчёт,
    в котором каждая строка соответствует отдельному сезону.

    Если в сезоне недостаточно событий, этот сезон пропускается.

    Параметры:
        by_department_year_dir: папка с CSV по департаментам
        by_road_year_dir: папка с CSV по дорогам
        output_department_dir: папка для сохранения результатов по департаментам
        output_road_dir: папка для сохранения результатов по дорогам
        min_points: минимальное число положительных значений TIME_DIFF для сезона
        ks_alpha: уровень значимости для KS-теста
    """

    season_order = ["Зима", "Весна", "Лето", "Осень"]
    season_index_map = {
        "Зима": 1,
        "Весна": 2,
        "Лето": 3,
        "Осень": 4,
    }

    def _process_directory(source_dir: Path, save_dir: Path, entity_type: str) -> None:
        if not source_dir.exists():
            print(f"⚠️ Папка не найдена: {source_dir}")
            return

        save_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(source_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️ Нет CSV файлов в папке: {source_dir}")
            return

        print(f"\n📂 Оценка λ₀ по сезонам для папки: {source_dir}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")

                if "START_TIME" not in df.columns and "TIME_DIFF" not in df.columns:
                    print(f"⚠️ Пропуск {csv_file.name}: нет START_TIME и TIME_DIFF")
                    continue

                if "START_TIME" not in df.columns:
                    print(f"⚠️ Пропуск {csv_file.name}: нет START_TIME, сезон определить невозможно")
                    continue

                df["START_TIME"] = pd.to_datetime(df["START_TIME"], errors="coerce")
                df = df[df["START_TIME"].notna()].copy()

                if df.empty:
                    print(f"⚠️ Пропуск {csv_file.name}: нет валидных START_TIME")
                    continue

                df = df.sort_values("START_TIME").reset_index(drop=True)
                df["MONTH"] = df["START_TIME"].dt.month
                df["SEASON"] = df["MONTH"].apply(_month_to_season)

                if "TIME_DIFF" not in df.columns:
                    delta = df["START_TIME"] - df["START_TIME"].iloc[0]
                    delta_minutes = delta.dt.total_seconds() / 60
                    df["TIME_DIFF"] = delta_minutes.diff().fillna(0)

                stem_parts = csv_file.stem.rsplit("_", 1)
                if len(stem_parts) == 2 and stem_parts[1].isdigit():
                    entity_name = stem_parts[0]
                    year = int(stem_parts[1])
                else:
                    entity_name = csv_file.stem
                    year = None

                result_rows = []

                for season in season_order:
                    season_df = df[df["SEASON"] == season].copy()
                    if season_df.empty:
                        continue

                    data = pd.to_numeric(season_df["TIME_DIFF"], errors="coerce").to_numpy()
                    data = data[np.isfinite(data)]
                    data = data[data > 0]

                    if len(data) < min_points:
                        print(
                            f"⚠️ {csv_file.name} / {season}: "
                            f"недостаточно данных ({len(data)} < {min_points})"
                        )
                        continue

                    mu = float(np.mean(data))
                    lambda_mle = 1.0 / mu

                    f = Fitter(data, distributions=["expon"], timeout=10)
                    f.fit()
                    loc, scale = f.fitted_param["expon"]
                    lambda_fitter = 1.0 / scale

                    ks_stat, ks_pvalue = kstest(data, "expon", args=(0, 1 / lambda_fitter))

                    minutes_in_month = 30 * 24 * 60
                    lambda_month = lambda_fitter * minutes_in_month

                    result_rows.append({
                        "FILE_NAME": csv_file.name,
                        "ENTITY_TYPE": entity_type,
                        "ENTITY_NAME": entity_name,
                        "YEAR": year,
                        "SEASON": season,
                        "SEASON_INDEX": season_index_map[season],
                        "N": len(data),
                        "TIME_DIFF_MEAN": mu,
                        "LAMBDA_MLE": lambda_mle,
                        "LAMBDA_FITTER": lambda_fitter,
                        "LAMBDA_MONTH": lambda_month,
                        "KS_STAT": ks_stat,
                        "KS_PVALUE": ks_pvalue,
                        "KS_OK": ks_pvalue >= ks_alpha,
                        "KS_ALPHA": ks_alpha,
                    })

                if not result_rows:
                    print(f"⚠️ Пропуск {csv_file.name}: нет сезонов с достаточным числом событий")
                    continue

                result_df = pd.DataFrame(result_rows)
                result_df["SEASON"] = pd.Categorical(
                    result_df["SEASON"],
                    categories=season_order,
                    ordered=True,
                )
                result_df = result_df.sort_values(["SEASON_INDEX", "SEASON"]).reset_index(drop=True)

                out_path = save_dir / f"lambda_0_{entity_name}_{year}.csv"
                result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

                print(f"✅ Сохранён: {out_path}")

            except Exception as e:
                print(f"❌ Ошибка обработки {csv_file.name}: {e}")

    _process_directory(
        source_dir=by_department_year_dir,
        save_dir=output_department_dir,
        entity_type="department",
    )

    _process_directory(
        source_dir=by_road_year_dir,
        save_dir=output_road_dir,
        entity_type="road",
    )

    print("\n🎉 Оценка λ₀ по сезонам завершена.")