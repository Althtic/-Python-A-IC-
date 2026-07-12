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
from QuantileSpreadTest import (  # noqa: E402
    calculate_turnover_rate,
    data_preprocessing,
    holding_period,
    layers,
    plot_multiple_return_metrics,
    plot_turnover,
    process_group_by_date,
    spread_ret_cumsum_calculate,
    t_test_spread_ret,
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
    parser = argparse.ArgumentParser(description="Run quantile spread backtest from DolphinDB factor_daily and market tables.")
    parser.add_argument("--factor-name", default="alpha_60")
    parser.add_argument("--factor-version", default="v1")
    parser.add_argument("--start-date", default=test_window_start)
    parser.add_argument("--end-date", default=test_window_end)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="123456")
    args = parser.parse_args()

    save_dir = result_dir(args.factor_name) / "dolphindb_quantile"
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
    if "circ_mv" not in df.columns:
        raise ValueError("DolphinDB factor panel must include circ_mv for quantile spread test.")

    df_processed = data_preprocessing(df, start_date, end_date, holding_period)
    layer_cols = ["trade_date", "ts_code", args.factor_name, "holding_lndret", "circ_mv"]
    df_layers_processed = (
        df_processed[layer_cols]
        .groupby("trade_date", group_keys=False)
        .apply(lambda group: process_group_by_date(group, layers))
        .reset_index(drop=True)
    )

    df_ew, spread_ew, cum_ew = spread_ret_cumsum_calculate(df_layers_processed, layers, "mean_lndret", "_ew")
    df_vw, spread_vw, cum_vw = spread_ret_cumsum_calculate(df_layers_processed, layers, "mean_lndret_vw", "_vw")
    turnover_df, avg_ew, std_ew, ann_ew, avg_vw, std_vw, ann_vw, group_turnovers = calculate_turnover_rate(
        df_layers_processed, layers, holding_period
    )

    t_ew = t_test_spread_ret(spread_ew, run_label="等权L-S")
    t_vw = t_test_spread_ret(spread_vw, run_label="市值加权L-S")
    plot_multiple_return_metrics(df_ew, cum_ew, df_vw, cum_vw, layers, args.factor_name, save_dir, t_ew, t_vw, "_ew", "_vw")
    plot_turnover(
        turnover_df,
        group_turnovers,
        args.factor_name,
        layers,
        save_dir,
        std_ew=std_ew,
        std_vw=std_vw,
        holding_period=holding_period,
    )

    print({
        "factor": args.factor_name,
        "version": args.factor_version,
        "rows_after_preprocess": len(df_processed),
        "rows_after_layering": len(df_layers_processed),
        "ls_ew_final": float(cum_ew["L-S"]),
        "ls_vw_final": float(cum_vw["L-S"]),
        "avg_turnover_ew": float(avg_ew),
        "avg_turnover_vw": float(avg_vw),
        "save_dir": str(save_dir),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


