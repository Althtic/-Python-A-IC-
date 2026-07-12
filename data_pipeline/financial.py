from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import audit_frame, read_csv_checked, write_audit
from .paths import backtest_data_path, cashflow_path, ensure_parent


logger = logging.getLogger(__name__)


def normalize_date_column(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: str(int(float(x))) if pd.notna(x) and x != "" else pd.NA)


def fill_first_nan_only(group: pd.DataFrame, value_col: str) -> pd.DataFrame:
    values = group[value_col].to_numpy()
    cleaned = []
    for i, value in enumerate(values):
        if pd.notna(value):
            cleaned.append(value)
        elif i > 0 and pd.notna(values[i - 1]):
            cleaned.append(values[i - 1])
        else:
            cleaned.append(np.nan)
    group[value_col] = cleaned
    return group


def load_and_clean_disclosure_files(
    source_dir: Path,
    date_col: str,
    value_col: str,
    output_path: Path | None = None,
    include_pattern: str = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].csv",
    max_announcement_date: str = "20251231",
) -> pd.DataFrame:
    files = sorted(source_dir.glob(include_pattern))
    if not files:
        raise FileNotFoundError(f"No disclosure files matching {include_pattern!r} under {source_dir}")

    target_columns = ["ts_code", date_col, "end_date", value_col]
    frames = []
    for path in files:
        frame = read_csv_checked(path, target_columns, usecols=target_columns, dtype={"ts_code": str})
        frame[date_col] = normalize_date_column(frame[date_col])
        frame["end_date"] = normalize_date_column(frame["end_date"])
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates()
    result = result[target_columns]
    result = result[result[date_col].notna() & (result[date_col] <= max_announcement_date)]
    result = result[~result["ts_code"].str.startswith("A", na=False)]
    result = result.sort_values(["ts_code", date_col, "end_date"]).reset_index(drop=True)
    result = result.groupby("ts_code", group_keys=False).apply(fill_first_nan_only, value_col=value_col)
    result = result.drop_duplicates().reset_index(drop=True)

    if output_path:
        ensure_parent(output_path)
        result.to_csv(output_path, index=False)
        logger.info("Saved cleaned disclosure data to %s", output_path)

    return result


def align_latest_disclosure(
    market_df: pd.DataFrame,
    disclosure_df: pd.DataFrame,
    announcement_col: str,
    value_columns: Iterable[str],
) -> pd.DataFrame:
    market = market_df.sort_values(["ts_code", "trade_date"]).copy()
    disclosure = disclosure_df.sort_values(["ts_code", announcement_col, "end_date"]).copy()

    market["trade_date"] = pd.to_numeric(market["trade_date"], errors="coerce")
    market = market.dropna(subset=["trade_date"])
    market["trade_date"] = market["trade_date"].astype("int64")

    disclosure[announcement_col] = pd.to_numeric(disclosure[announcement_col], errors="coerce")
    disclosure["end_date"] = pd.to_numeric(disclosure["end_date"], errors="coerce")
    disclosure = disclosure.dropna(subset=[announcement_col, "end_date"])
    disclosure = disclosure[disclosure["end_date"] <= disclosure[announcement_col]]
    disclosure[announcement_col] = disclosure[announcement_col].astype("int64")
    disclosure["end_date"] = disclosure["end_date"].astype("int64")

    aligned_parts = []
    value_columns = list(value_columns)
    for ts_code, market_part in market.groupby("ts_code", sort=False):
        fin_part = disclosure[disclosure["ts_code"] == ts_code]
        if fin_part.empty:
            aligned_parts.append(market_part.copy())
            continue

        market_part = market_part.sort_values("trade_date").copy()
        fin_part = fin_part.sort_values(announcement_col).copy()

        merged = pd.merge_asof(
            market_part,
            fin_part[["ts_code", announcement_col, "end_date", *value_columns]],
            by="ts_code",
            left_on="trade_date",
            right_on=announcement_col,
            direction="backward",
            allow_exact_matches=True,
        )

        invalid = merged["end_date"].notna() & (merged["end_date"] > merged["trade_date"])
        if invalid.any():
            cols_to_clear = [announcement_col, "end_date", *value_columns]
            merged.loc[invalid, cols_to_clear] = pd.NA

        aligned_parts.append(merged)

    if not aligned_parts:
        return market.iloc[0:0].copy()

    result = pd.concat(aligned_parts, ignore_index=True)
    return result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def compute_cashflow_ttm_qoq(cashflow_df: pd.DataFrame) -> pd.DataFrame:
    required = ["ts_code", "f_ann_date", "end_date", "c_pay_acq_const_fiolta"]
    missing = [col for col in required if col not in cashflow_df.columns]
    if missing:
        raise ValueError(f"cashflow_df missing required columns: {missing}")

    ratio = cashflow_df[required].copy()
    ratio["end_date"] = ratio["end_date"].astype(int)
    ratio = ratio.sort_values(["ts_code", "end_date", "f_ann_date"]).groupby(
        ["ts_code", "end_date"], as_index=False
    ).last()

    quarter_map = {331: "Q1", 630: "Q2", 930: "Q3", 1231: "Q4"}
    ratio["quarter"] = (ratio["end_date"] % 10000).map(quarter_map)
    ratio["year"] = ratio["end_date"] // 10000
    ratio = ratio[ratio["quarter"].notna()].copy()

    has_quarterly = (
        ratio.groupby(["ts_code", "year"])["quarter"]
        .apply(lambda x: bool(set(x) & {"Q1", "Q3"}))
        .groupby(level=0)
        .any()
        .reindex(ratio["ts_code"].unique())
        .fillna(False)
    )

    def get_ttm(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("end_date").copy()
        q_map = {(row.year, row.quarter): row.c_pay_acq_const_fiolta for row in group.itertuples()}
        has_quarter = bool(has_quarterly.get(group["ts_code"].iloc[0], False))
        ttm_values = []
        for row in group.itertuples():
            q = row.quarter
            y = row.year
            value = row.c_pay_acq_const_fiolta
            if q == "Q4":
                ttm = value
            elif q == "Q2":
                q4_prev = q_map.get((y - 1, "Q4"), np.nan)
                q2_prev = q_map.get((y - 1, "Q2"), np.nan)
                ttm = value + (q4_prev - q2_prev) if pd.notna(q4_prev) and pd.notna(q2_prev) else np.nan
            elif has_quarter and q in ("Q1", "Q3"):
                q4_prev = q_map.get((y - 1, "Q4"), np.nan)
                q_prev = q_map.get((y - 1, q), np.nan)
                ttm = value + (q4_prev - q_prev) if pd.notna(q4_prev) and pd.notna(q_prev) else np.nan
            else:
                ttm = np.nan
            ttm_values.append(ttm)
        group["ttm"] = ttm_values
        return group

    ratio = ratio.groupby("ts_code", group_keys=False).apply(get_ttm).reset_index(drop=True)
    ratio["ttm_prev"] = ratio.groupby("ts_code")["ttm"].shift(1)
    ratio["qoq"] = np.where(
        ratio["ttm_prev"].notna() & (ratio["ttm_prev"] != 0),
        (ratio["ttm"] - ratio["ttm_prev"]) / ratio["ttm_prev"],
        np.nan,
    )
    ratio = ratio.drop(columns=["ttm_prev"])

    years = np.arange(ratio["year"].min(), ratio["year"].max() + 1)
    full_dates = [year * 10000 + suffix for year in years for suffix in [331, 630, 930, 1231]]
    grid = pd.MultiIndex.from_product([ratio["ts_code"].unique(), full_dates], names=["ts_code", "end_date"])
    aligned = pd.DataFrame(index=grid).reset_index()
    aligned["year"] = aligned["end_date"] // 10000
    aligned["quarter"] = (aligned["end_date"] % 10000).map(quarter_map)
    aligned = aligned.merge(
        ratio[["ts_code", "end_date", "ttm", "qoq", "c_pay_acq_const_fiolta", "f_ann_date"]],
        on=["ts_code", "end_date"],
        how="left",
    )
    aligned = aligned.sort_values(["ts_code", "end_date"])
    for col in ["ttm", "qoq"]:
        aligned[col] = aligned.groupby("ts_code")[col].transform(
            lambda s: pd.Series(fill_first_nan_values(s.to_numpy()), index=s.index)
        )
    aligned["f_ann_date"] = aligned["f_ann_date"].apply(lambda x: str(int(x)) if pd.notna(x) else pd.NA)
    return aligned.reset_index(drop=True)


def fill_first_nan_values(values: np.ndarray) -> list[float]:
    out = []
    for i, value in enumerate(values):
        if pd.notna(value):
            out.append(value)
        elif i > 0 and pd.notna(values[i - 1]):
            out.append(values[i - 1])
        else:
            out.append(np.nan)
    return out


def run_cashflow_qoq_pipeline(
    source_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    source_path = source_path or cashflow_path("构建固定资产等支付的现金_clean.csv")
    output_path = output_path or cashflow_path("qoq.csv")
    source = read_csv_checked(
        source_path,
        ["ts_code", "f_ann_date", "end_date", "c_pay_acq_const_fiolta"],
        dtype={"ts_code": str},
    )
    result = compute_cashflow_ttm_qoq(source)
    ensure_parent(output_path)
    result.to_csv(output_path, index=False)
    logger.info("Saved cashflow qoq data to %s", output_path)
    return result


def run_disclosure_alignment(
    cleaned_disclosure_path: Path,
    output_path: Path,
    announcement_col: str,
    value_columns: Iterable[str],
    market_path: Path | None = None,
    audit_path: Path | None = None,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    market_path = market_path or backtest_data_path("20170930-20251231_pipe.csv")
    read_kwargs = {"low_memory": False}
    if sample_rows is not None:
        read_kwargs["nrows"] = sample_rows

    market_df = read_csv_checked(market_path, ["ts_code", "trade_date"], **read_kwargs)
    disclosure_df = read_csv_checked(
        cleaned_disclosure_path,
        ["ts_code", announcement_col, "end_date", *list(value_columns)],
        **read_kwargs,
    )
    result = align_latest_disclosure(market_df, disclosure_df, announcement_col, value_columns)

    if sample_rows is None:
        ensure_parent(output_path)
        result.to_csv(output_path, index=False)
        logger.info("Saved aligned disclosure panel to %s", output_path)

    if audit_path:
        write_audit(
            audit_path,
            [
                audit_frame(
                    output_path.name,
                    result,
                    ["ts_code", "trade_date"],
                    [announcement_col, "end_date", *list(value_columns)],
                )
            ],
        )
    return result




