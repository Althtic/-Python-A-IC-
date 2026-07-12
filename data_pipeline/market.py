from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .io import audit_frame, read_csv_checked, write_audit
from .paths import backtest_data_path, ensure_parent


logger = logging.getLogger(__name__)


RAW_MARKET_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
]

PIPE_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "adj_factor",
    "amount",
    "dret",
    "suspend_timing",
    "suspend_type",
    "is_trading",
    "market",
    "industry_name",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]


def apply_forward_adjustment(market_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    market = market_df.copy()
    adj = adj_df[["ts_code", "trade_date", "adj_factor"]].drop_duplicates(["ts_code", "trade_date"])
    merged = market.merge(adj, how="left", on=["ts_code", "trade_date"])

    missing = merged["adj_factor"].isna().sum()
    if missing:
        raise ValueError(f"Missing adj_factor for {missing:,} market rows")

    latest_factor = merged.groupby("ts_code")["adj_factor"].transform("last")
    merged["forward_factor"] = merged["adj_factor"] / latest_factor

    price_cols = ["open", "high", "low", "close", "pre_close"]
    merged[price_cols] = merged[price_cols].mul(merged["forward_factor"], axis=0).round(2)
    merged["vol"] = (merged["vol"] / merged["forward_factor"]).round(2)

    return merged.drop(columns=["forward_factor"])


def remove_st_rows(market_df: pd.DataFrame, st_df: pd.DataFrame) -> pd.DataFrame:
    st = st_df[["ts_code", "trade_date", "type"]].drop_duplicates(["ts_code", "trade_date"])
    merged = market_df.merge(st, how="left", on=["ts_code", "trade_date"])
    cleaned = merged[merged["type"] != "ST"].copy()
    return cleaned.drop(columns=["type"])


def remove_new_stock_initial_days(market_df: pd.DataFrame, min_history_days: int = 120) -> pd.DataFrame:
    df = market_df.sort_values(["ts_code", "trade_date"]).copy()
    sample_start = df["trade_date"].min()
    first_trade = df.groupby("ts_code")["trade_date"].transform("min")
    trade_day_num = df.groupby("ts_code").cumcount() + 1
    is_new_in_sample = first_trade > sample_start
    cleaned = df[(~is_new_in_sample) | (trade_day_num > min_history_days)].copy()
    return cleaned.reset_index(drop=True)


def calculate_daily_return(market_df: pd.DataFrame) -> pd.DataFrame:
    df = market_df.sort_values(["ts_code", "trade_date"]).copy()
    df["dret"] = df.groupby("ts_code")["close"].pct_change().round(5)
    return df.dropna(subset=["dret"]).reset_index(drop=True)


def add_trading_status(market_df: pd.DataFrame, suspension_df: pd.DataFrame) -> pd.DataFrame:
    suspension = suspension_df.drop_duplicates(["ts_code", "trade_date"])
    df = market_df.merge(suspension, how="left", on=["ts_code", "trade_date"])

    is_suspended = df["suspend_timing"].isna() & (df["suspend_type"] == "S")
    is_one_word_limit = (
        df["dret"].notna()
        & (df["open"] == df["close"])
        & (df["high"] == df["low"])
        & (df["open"] == df["high"])
    )

    df["is_trading"] = np.select([is_suspended, is_one_word_limit], [-1, -2], default=1)
    return df


def identify_market(ts_code: str) -> str:
    code = str(ts_code).split(".")[0]
    if code.startswith(("8", "9")):
        return "北交所"
    if code.startswith(("688", "689")):
        return "科创板"
    if code.startswith(("300", "301", "302")):
        return "创业板"
    if code.startswith(("60", "000", "001", "002", "003")):
        return "沪深主板"
    return "其他"


def cap_abnormal_returns(market_df: pd.DataFrame) -> pd.DataFrame:
    df = market_df.copy()
    df["market"] = df["ts_code"].map(identify_market)
    limits = {"北交所": 0.30, "科创板": 0.20, "创业板": 0.20, "沪深主板": 0.10, "其他": 0.30}
    limit = df["market"].map(limits).fillna(0.30)
    df["dret"] = np.where(df["dret"].abs() > limit, np.sign(df["dret"]) * limit, df["dret"])
    return df


def add_industry(market_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    industry = industry_df.rename(columns={"stock_code": "ts_code"})[
        ["trade_date", "ts_code", "industry_name"]
    ].drop_duplicates(["trade_date", "ts_code"])
    df = market_df.merge(industry, how="left", on=["trade_date", "ts_code"])
    df["industry_name"] = df["industry_name"].fillna("未分类")
    return df


def add_daily_basic(market_df: pd.DataFrame, daily_basic_df: pd.DataFrame) -> pd.DataFrame:
    daily = daily_basic_df.drop(columns=["close"], errors="ignore").copy()
    daily = daily.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"])
    fill_cols = daily.columns.difference(["ts_code", "trade_date"])
    daily[fill_cols] = daily.groupby("ts_code")[fill_cols].ffill()
    return market_df.merge(daily, how="left", on=["ts_code", "trade_date"])


def build_market_panel(
    market_df: pd.DataFrame,
    adj_df: pd.DataFrame,
    st_df: pd.DataFrame,
    suspension_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    min_history_days: int = 120,
) -> pd.DataFrame:
    df = market_df[RAW_MARKET_COLUMNS].copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df = apply_forward_adjustment(df, adj_df)
    df = remove_st_rows(df, st_df)
    df = remove_new_stock_initial_days(df, min_history_days=min_history_days)
    df = calculate_daily_return(df)
    df = add_trading_status(df, suspension_df)
    df = cap_abnormal_returns(df)
    df = add_industry(df, industry_df)
    df = add_daily_basic(df, daily_basic_df)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    available_columns = [col for col in PIPE_COLUMNS if col in df.columns]
    return df[available_columns]


def run_market_pipeline(
    output_path: Path | None = None,
    audit_path: Path | None = None,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    output_path = output_path or backtest_data_path("20170930-20251231_pipe.csv")

    read_kwargs = {"low_memory": False}
    market_kwargs = dict(read_kwargs)
    if sample_rows is not None:
        market_kwargs["nrows"] = sample_rows

    market_df = read_csv_checked(backtest_data_path("20170930-20251231_ori.csv"), RAW_MARKET_COLUMNS, **market_kwargs)
    adj_df = read_csv_checked(backtest_data_path("adj_factor_ori.csv"), ["ts_code", "trade_date", "adj_factor"], **read_kwargs)
    st_df = read_csv_checked(backtest_data_path("ST-Stocks_ori.csv"), ["ts_code", "trade_date", "type"], **read_kwargs)
    suspension_df = read_csv_checked(
        backtest_data_path("Suspension_data17-25_ori.csv"),
        ["ts_code", "trade_date", "suspend_timing", "suspend_type"],
        **read_kwargs,
    )
    industry_df = read_csv_checked(
        backtest_data_path("SWlevel1_sorted_ori.csv"),
        ["trade_date", "stock_code", "industry_name"],
        **read_kwargs,
    )
    daily_basic_df = read_csv_checked(backtest_data_path("daily_basic_ori.csv"), ["ts_code", "trade_date"], **read_kwargs)

    result = build_market_panel(market_df, adj_df, st_df, suspension_df, industry_df, daily_basic_df)

    if sample_rows is None:
        ensure_parent(output_path)
        result.to_csv(output_path, index=False)
        logger.info("Saved market panel to %s", output_path)

    if audit_path:
        write_audit(
            audit_path,
            [
                audit_frame("market_panel", result, ["ts_code", "trade_date"], ["dret", "industry_name", "circ_mv"]),
            ],
        )

    return result



