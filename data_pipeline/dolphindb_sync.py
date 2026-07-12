from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_pipeline.paths import backtest_data_path
from project_paths import FACTORS_DIR


DEFAULT_DB_PATH = "dfs://quant_system"

MARKET_BAR_COLUMNS = [
    "trade_date", "ts_code", "open", "high", "low", "close", "pre_close",
    "vol", "amount", "adj_factor", "dret",
]
MARKET_STATUS_COLUMNS = ["trade_date", "ts_code", "is_trading", "suspend_timing", "suspend_type", "market"]
MARKET_VALUATION_COLUMNS = [
    "trade_date", "ts_code", "turnover_rate", "turnover_rate_f", "volume_ratio",
    "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
    "total_share", "float_share", "free_share", "total_mv", "circ_mv",
]
INDUSTRY_COLUMNS = ["trade_date", "ts_code", "industry_name"]

FINANCIAL_FEATURE_SOURCES = {
    "roe": {
        "path": backtest_data_path("roe.csv"),
        "value_col": "roe",
        "ann_col": "ann_date",
        "feature_name": "roe",
    },
    "equity": {
        "path": backtest_data_path("归母股东权益.csv"),
        "value_col": "total_hldr_eqy_exc_min_int",
        "ann_col": "f_ann_date",
        "feature_name": "total_hldr_eqy_exc_min_int",
    },
    "cashflow_qoq": {
        "path": backtest_data_path("环比购买固定资产支出增长率(TTM).csv"),
        "value_col": "qoq",
        "ann_col": "f_ann_date",
        "feature_name": "cashflow_capex_qoq_ttm",
    },
}


def connect(host: str, port: int, user: str, password: str):
    try:
        import dolphindb as ddb
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install with `pip install dolphindb`.") from exc

    session = ddb.session()
    session.connect(host, port, user, password)
    return session


def _to_date(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return pd.to_datetime(numeric.astype(str), format="%Y%m%d", errors="coerce")


def _prepare_common(df: pd.DataFrame, date_columns: Iterable[str] = ("trade_date",)) -> pd.DataFrame:
    out = df.copy()
    for col in date_columns:
        if col in out.columns:
            out[col] = _to_date(out[col])
    if "ts_code" in out.columns:
        out["ts_code"] = out["ts_code"].astype(str)
    return out

def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _append(session, db_path: str, table_name: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    select_exprs = {
        "core_market_bar_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, open, high, low, close, pre_close, vol, amount, adj_factor, dret",
        "core_market_status_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, is_trading, symbol(string(suspend_timing)) as suspend_timing, symbol(string(suspend_type)) as suspend_type, symbol(market) as market",
        "core_market_valuation_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv",
        "core_industry_sw_l1_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, symbol(industry_name) as industry_name, symbol(source) as source",
        "feature_financial_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, symbol(feature_name) as feature_name, feature_value, date(ann_date) as ann_date, date(end_date) as end_date, symbol(data_version) as data_version",
        "factor_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, symbol(factor_name) as factor_name, factor_value, raw_value, symbol(factor_version) as factor_version, symbol(data_version) as data_version, neutralized, winsorized, symbol(run_id) as run_id, timestamp(created_at) as created_at",
    }
    if table_name not in select_exprs:
        raise ValueError(f"Unsupported DolphinDB table append target: {table_name}")

    session.upload({"qsAppendTmp": df})
    script = f"""
        target = loadTable('{db_path}', `{table_name})
        target.append!(select {select_exprs[table_name]} from qsAppendTmp)
        undef(`qsAppendTmp, VAR)
    """
    session.run(script)
    return len(df)

def _read_chunks(path: Path, usecols: list[str], chunksize: int, sample_rows: int | None):
    kwargs = {"usecols": usecols, "low_memory": False}
    if sample_rows is not None:
        kwargs["nrows"] = sample_rows
        yield pd.read_csv(path, **kwargs)
    else:
        yield from pd.read_csv(path, chunksize=chunksize, **kwargs)


def sync_market(session, db_path: str, chunksize: int, sample_rows: int | None) -> dict[str, int]:
    source = backtest_data_path("20170930-20251231_pipe.csv")
    usecols = sorted(set(MARKET_BAR_COLUMNS + MARKET_STATUS_COLUMNS + MARKET_VALUATION_COLUMNS + INDUSTRY_COLUMNS))
    counts = {
        "core_market_bar_daily": 0,
        "core_market_status_daily": 0,
        "core_market_valuation_daily": 0,
        "core_industry_sw_l1_daily": 0,
    }

    for chunk in _read_chunks(source, usecols, chunksize, sample_rows):
        _require_columns(chunk, usecols, source)
        chunk = _prepare_common(chunk)

        counts["core_market_bar_daily"] += _append(
            session, db_path, "core_market_bar_daily", chunk[MARKET_BAR_COLUMNS]
        )
        counts["core_market_status_daily"] += _append(
            session, db_path, "core_market_status_daily", chunk[MARKET_STATUS_COLUMNS]
        )
        counts["core_market_valuation_daily"] += _append(
            session, db_path, "core_market_valuation_daily", chunk[MARKET_VALUATION_COLUMNS]
        )

        industry = chunk[INDUSTRY_COLUMNS].drop_duplicates(["trade_date", "ts_code"])
        industry = industry.assign(source="SW_L1")
        counts["core_industry_sw_l1_daily"] += _append(session, db_path, "core_industry_sw_l1_daily", industry)

    return counts


def _financial_frame(chunk: pd.DataFrame, spec: dict[str, object], data_version: str) -> pd.DataFrame:
    value_col = str(spec["value_col"])
    ann_col = str(spec["ann_col"])
    frame = chunk[["trade_date", "ts_code", "end_date", value_col, ann_col]].rename(
        columns={value_col: "feature_value", ann_col: "ann_date"}
    )
    frame["feature_name"] = str(spec["feature_name"])
    frame["data_version"] = data_version
    frame = frame[["trade_date", "ts_code", "feature_name", "feature_value", "ann_date", "end_date", "data_version"]]
    frame = _prepare_common(frame, date_columns=("trade_date", "ann_date", "end_date"))
    return frame[["trade_date", "ts_code", "feature_name", "feature_value", "ann_date", "end_date", "data_version"]]


def sync_features(
    session,
    db_path: str,
    selected_features: Iterable[str] | None,
    chunksize: int,
    sample_rows: int | None,
    data_version: str,
) -> dict[str, int]:
    selected = list(selected_features or FINANCIAL_FEATURE_SOURCES)
    counts: dict[str, int] = {}

    for feature in selected:
        if feature not in FINANCIAL_FEATURE_SOURCES:
            raise ValueError(f"Unknown feature {feature!r}; choices: {sorted(FINANCIAL_FEATURE_SOURCES)}")
        spec = FINANCIAL_FEATURE_SOURCES[feature]
        source = Path(spec["path"])
        value_col = str(spec["value_col"])
        ann_col = str(spec["ann_col"])
        usecols = ["trade_date", "ts_code", "end_date", value_col, ann_col]
        counts[feature] = 0

        for chunk in _read_chunks(source, usecols, chunksize, sample_rows):
            _require_columns(chunk, usecols, source)
            frame = _financial_frame(chunk, spec, data_version).dropna(subset=["feature_value"])
            counts[feature] += _append(session, db_path, "feature_financial_daily", frame)

    return counts


def sync_factor(
    session,
    db_path: str,
    factor_name: str,
    factor_path: Path,
    chunksize: int,
    sample_rows: int | None,
    factor_version: str,
    data_version: str,
    run_id: str | None,
) -> dict[str, int | str]:
    run_id = run_id or f"{factor_name}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
    if not factor_path.exists():
        raise FileNotFoundError(f"Missing factor CSV: {factor_path}")
    header = pd.read_csv(factor_path, nrows=0).columns
    usecols = ["trade_date", "ts_code", factor_name]
    if "raw_factor" in header:
        usecols.append("raw_factor")

    total = 0
    for chunk in _read_chunks(factor_path, usecols, chunksize, sample_rows):
        _require_columns(chunk, ["trade_date", "ts_code", factor_name], factor_path)
        frame = chunk.rename(columns={factor_name: "factor_value", "raw_factor": "raw_value"})
        if "raw_value" not in frame.columns:
            frame["raw_value"] = np.nan
        frame["factor_name"] = factor_name
        frame["factor_version"] = factor_version
        frame["data_version"] = data_version
        frame["neutralized"] = True
        frame["winsorized"] = True
        frame["run_id"] = run_id
        frame["created_at"] = pd.Timestamp.now()
        frame = _prepare_common(frame)
        frame = frame[[
            "trade_date", "month", "ts_code", "factor_name", "factor_value", "raw_value",
            "factor_version", "data_version", "neutralized", "winsorized", "run_id", "created_at",
        ]]
        total += _append(session, db_path, "factor_daily", frame)

    return {"factor_daily": total, "run_id": run_id}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync QuantSystem CSV data into DolphinDB.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--sample-rows", type=int, default=None)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("market", help="Sync the cleaned market panel.")

    features = sub.add_parser("features", help="Sync point-in-time financial features.")
    features.add_argument("--features", nargs="+", choices=sorted(FINANCIAL_FEATURE_SOURCES), default=None)
    features.add_argument("--data-version", default="local_csv")

    factor = sub.add_parser("factor", help="Sync one factor CSV.")
    factor.add_argument("--factor-name", required=True)
    factor.add_argument("--factor-path", type=Path, default=None)
    factor.add_argument("--factor-version", default="v1")
    factor.add_argument("--data-version", default="local_csv")
    factor.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = connect(args.host, args.port, args.user, args.password)

    if args.command == "market":
        print(sync_market(session, args.db_path, args.chunk_size, args.sample_rows))
    elif args.command == "features":
        print(sync_features(session, args.db_path, args.features, args.chunk_size, args.sample_rows, args.data_version))
    elif args.command == "factor":
        factor_path = args.factor_path or (FACTORS_DIR / f"{args.factor_name}.csv")
        print(sync_factor(
            session, args.db_path, args.factor_name, factor_path, args.chunk_size,
            args.sample_rows, args.factor_version, args.data_version, args.run_id,
        ))
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
