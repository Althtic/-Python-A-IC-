from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_DIR = PROJECT_ROOT / "功能模块"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dolphindb_factor_loader import load_factor_panel_from_dolphindb
from project_paths import result_dir
from ValidationTest import (  # noqa: E402
    IC_calculate,
    data_preprocessing,
    monthly_processing,
    factor_cumuret_rank,
    ic_ma_period,
    plot_validation_analysis,
    plot_validation_monthly_series_bar,
    plot_validation_yearly_series_bar,
    test_period,
    test_window_end,
    test_window_start,
)


def effective_window(df: pd.DataFrame, start_date: str, end_date: str) -> tuple[str, str]:
    dates = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    start = max(pd.to_datetime(start_date, format="%Y%m%d"), dates.min())
    end = min(pd.to_datetime(end_date, format="%Y%m%d"), dates.max())
    if start > end:
        raise ValueError(f"Requested window {start_date}-{end_date} has no overlap with loaded data.")
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IC validation from DolphinDB factor_daily and market tables.")
    parser.add_argument("--factor-name", default="alpha_60")
    parser.add_argument("--factor-version", default="v1")
    parser.add_argument("--start-date", default=test_window_start)
    parser.add_argument("--end-date", default=test_window_end)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="123456")
    args = parser.parse_args()

    save_dir = result_dir(args.factor_name) / "dolphindb_validation"
    save_dir.mkdir(parents=True, exist_ok=True)
    df = load_factor_panel_from_dolphindb(
        factor_name=args.factor_name,
        factor_version=args.factor_version,
        start_date=args.start_date,
        end_date=args.end_date,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )
    print(f"loaded rows: {len(df):,}")
    start_date, end_date = effective_window(df, args.start_date, args.end_date)
    print(f"effective window: {start_date} ~ {end_date}")

    df_initial = data_preprocessing(df, start_date, end_date)
    df_ranked = factor_cumuret_rank(df_initial, args.factor_name, test_period)
    result_df, ic_decay, ttest_result = IC_calculate(df_ranked, ic_ma_period, test_period)

    monthly_mean_dict, monthly_icir_dict = monthly_processing(result_df)

    plot_validation_analysis(result_df, ic_decay, test_period, save_dir, result_dict=None)
    plot_validation_yearly_series_bar(result_df, save_dir, ttest_result, result_dict=None)
    plot_validation_monthly_series_bar(monthly_mean_dict, monthly_icir_dict, save_dir, result_dict=None)

    print({
        "factor": args.factor_name,
        "version": args.factor_version,
        "rank_ic_mean": float(result_df["Rank_IC"].mean()),
        "icir": float(result_df["ICIR"].iloc[0]),
        "ic_win_rate": float(result_df["IC_Win_Rate"].iloc[0]),
        "rows_after_preprocess": len(df_ranked),
        "save_dir": str(save_dir),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




