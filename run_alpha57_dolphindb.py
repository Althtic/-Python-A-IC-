from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_pipeline.dolphindb_sync import DEFAULT_DB_PATH, _append, connect

PROJECT_ROOT = Path(__file__).resolve().parent
FACTOR_CALCULATE_DIR = PROJECT_ROOT / "Factor_Calculate"
if str(FACTOR_CALCULATE_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOR_CALCULATE_DIR))

from WQ_Alpha57 import calculate_alpha  # noqa: E402


FACTOR_NAME = "alpha_57"
REQUIRED_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "dret",
    "vol",
    "amount",
    "industry_name",
    "suspend_type",
    "circ_mv",
]


def _month_start(value: str) -> pd.Timestamp:
    return pd.Period(value, freq="M").start_time


def _month_end(value: str) -> pd.Timestamp:
    return pd.Period(value, freq="M").end_time.normalize()


def list_months(start_month: str, end_month: str) -> list[str]:
    return [str(period) for period in pd.period_range(start=start_month, end=end_month, freq="M")]


def load_market_window(session, db_path: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start = start_date.strftime("%Y.%m.%d")
    end = end_date.strftime("%Y.%m.%d")
    script = f"""
        bar = loadTable('{db_path}', `core_market_bar_daily)
        status = loadTable('{db_path}', `core_market_status_daily)
        val = loadTable('{db_path}', `core_market_valuation_daily)
        ind = loadTable('{db_path}', `core_industry_sw_l1_daily)
        select
            b.ts_code as ts_code,
            b.trade_date as trade_date,
            b.open as open,
            b.high as high,
            b.low as low,
            b.close as close,
            b.pre_close as pre_close,
            b.dret as dret,
            b.vol as vol,
            b.amount as amount,
            i.industry_name as industry_name,
            s.suspend_type as suspend_type,
            v.circ_mv as circ_mv
        from bar as b
        left join status as s on b.trade_date=s.trade_date and b.ts_code=s.ts_code
        left join val as v on b.trade_date=v.trade_date and b.ts_code=v.ts_code
        left join ind as i on b.trade_date=i.trade_date and b.ts_code=i.ts_code
        where b.trade_date between {start} : {end}
        order by b.ts_code, b.trade_date
    """
    df = session.run(script)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    for col in ["ts_code", "industry_name", "suspend_type"]:
        df[col] = df[col].astype(str).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df[REQUIRED_COLUMNS]


def prepare_factor_rows(df: pd.DataFrame, factor_version: str, data_version: str, run_id: str) -> pd.DataFrame:
    out = df[["trade_date", "ts_code", FACTOR_NAME, "raw_factor"]].rename(
        columns={FACTOR_NAME: "factor_value", "raw_factor": "raw_value"}
    )
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    out["factor_name"] = FACTOR_NAME
    out["factor_version"] = factor_version
    out["data_version"] = data_version
    out["neutralized"] = True
    out["winsorized"] = True
    out["run_id"] = run_id
    out["created_at"] = pd.Timestamp.now()
    return out[
        [
            "trade_date",
            "ts_code",
            "factor_name",
            "factor_value",
            "raw_value",
            "factor_version",
            "data_version",
            "neutralized",
            "winsorized",
            "run_id",
            "created_at",
        ]
    ]


def delete_existing_factor(session, db_path: str, factor_name: str, factor_version: str) -> None:
    session.run(
        f"""
        t = loadTable('{db_path}', `factor_daily)
        delete from t where factor_name = `{factor_name}, factor_version = `{factor_version}
        """
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calculate alpha_57 from DolphinDB market tables and sync to factor_daily.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--start-month", default="2017-11")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument("--warmup-days", type=int, default=75)
    parser.add_argument("--factor-version", default="v1")
    parser.add_argument("--data-version", default="dolphindb_core_v1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--replace", action="store_true", help="Delete existing alpha_57 rows for the factor version before writing.")
    args = parser.parse_args(argv)

    session = connect(args.host, args.port, args.user, args.password)
    run_id = args.run_id or f"alpha_57_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

    if args.replace:
        delete_existing_factor(session, args.db_path, FACTOR_NAME, args.factor_version)

    total_rows = 0
    for month in list_months(args.start_month, args.end_month):
        month_start = _month_start(month)
        month_end = _month_end(month)
        window_start = month_start - pd.Timedelta(days=args.warmup_days)
        print(f"calculating {FACTOR_NAME} for {month}: window {window_start.date()} to {month_end.date()}", flush=True)

        market = load_market_window(session, args.db_path, window_start, month_end)
        if market.empty:
            print(f"skip {month}: no market rows", flush=True)
            continue

        try:
            calculated = calculate_alpha(market)
        except ValueError as exc:
            if "trade_date" in str(exc) and "ambiguous" in str(exc):
                print(f"skip {month}: no valid factor rows after rolling-window warm-up", flush=True)
                continue
            raise
        month_mask = (calculated["trade_date"] >= month_start.strftime("%Y%m%d")) & (
            calculated["trade_date"] <= month_end.strftime("%Y%m%d")
        )
        factor_rows = prepare_factor_rows(calculated.loc[month_mask], args.factor_version, args.data_version, run_id)
        written = _append(session, args.db_path, "factor_daily", factor_rows)
        total_rows += written
        print(f"wrote {written:,} rows for {month}; total {total_rows:,}", flush=True)

    print({"factor_daily": total_rows, "run_id": run_id})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
