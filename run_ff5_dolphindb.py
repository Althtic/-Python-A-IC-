from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path


import pandas as pd

from ff5_core import FF5_FACTORS, calculate_ff5

from data_pipeline.dolphindb_sync import DEFAULT_DB_PATH, connect
from data_pipeline.tushare_update import TushareClient, load_tushare_token
from project_paths import BACKTEST_DATA_DIR, FACTORS_DIR, PROJECT_ROOT


RATE_NAME = "shibor_3m"
TRADING_DAYS = 252.0


def _date_literal(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y.%m.%d")


def ensure_schema(session, db_path: str) -> None:
    session.run(
        f"""
        dbPath = '{db_path}';
        if (!existsDatabase(dbPath)) throw "Missing DolphinDB database: " + dbPath;
        db = database(dbPath);
        if (!existsTable(dbPath, `market_factor_daily)) {{
            marketFactor = table(
                1:0,
                `trade_date`month`factor_name`factor_value`factor_version`data_version`run_id`created_at,
                [DATE, MONTH, SYMBOL, DOUBLE, SYMBOL, SYMBOL, SYMBOL, TIMESTAMP]
            );
            db.createPartitionedTable(marketFactor, `market_factor_daily, `month`factor_name);
        }}
        if (!existsTable(dbPath, `macro_rate_daily)) {{
            macroRate = table(
                1:0,
                `trade_date`month`rate_name`rate_value`daily_rate`source`data_version`run_id`created_at,
                [DATE, MONTH, SYMBOL, DOUBLE, DOUBLE, SYMBOL, SYMBOL, SYMBOL, TIMESTAMP]
            );
            db.createPartitionedTable(macroRate, `macro_rate_daily, `month`rate_name);
        }}
        """
    )


def _append_market_factors(session, db_path: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    session.upload({"ff5AppendTmp": frame})
    session.run(
        f"""
        target = loadTable('{db_path}', `market_factor_daily)
        target.append!(select date(trade_date) as trade_date,
                              month(date(trade_date)) as month,
                              symbol(factor_name) as factor_name,
                              factor_value,
                              symbol(factor_version) as factor_version,
                              symbol(data_version) as data_version,
                              symbol(run_id) as run_id,
                              timestamp(created_at) as created_at
                       from ff5AppendTmp)
        undef(`ff5AppendTmp, VAR)
        """
    )
    return len(frame)


def _append_rates(session, db_path: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    session.upload({"rateAppendTmp": frame})
    session.run(
        f"""
        target = loadTable('{db_path}', `macro_rate_daily)
        target.append!(select date(trade_date) as trade_date,
                              month(date(trade_date)) as month,
                              symbol(rate_name) as rate_name,
                              rate_value,
                              daily_rate,
                              symbol(source) as source,
                              symbol(data_version) as data_version,
                              symbol(run_id) as run_id,
                              timestamp(created_at) as created_at
                       from rateAppendTmp)
        undef(`rateAppendTmp, VAR)
        """
    )
    return len(frame)


def _delete_market_window(session, db_path: str, factors: list[str], factor_version: str, start_date: str, end_date: str) -> None:
    start = _date_literal(start_date)
    end = _date_literal(end_date)
    names = "`" + "`".join(factors)
    session.run(
        f"""
        t = loadTable('{db_path}', `market_factor_daily)
        delete from t where factor_name in {names}, factor_version=`{factor_version}, trade_date between {start}:{end}
        """
    )


def _delete_rate_window(
    session,
    db_path: str,
    rate_name: str,
    start_date: str,
    end_date: str,
    data_version: str | None = None,
) -> None:
    start = _date_literal(start_date)
    end = _date_literal(end_date)
    version_filter = f", data_version=`{data_version}" if data_version else ""
    session.run(
        f"""
        t = loadTable('{db_path}', `macro_rate_daily)
        delete from t where rate_name=`{rate_name}{version_filter}, trade_date between {start}:{end}
        """
    )

def _output_rows(wide: pd.DataFrame, factor_version: str, data_version: str, run_id: str) -> pd.DataFrame:
    rows = wide.melt("trade_date", value_vars=list(FF5_FACTORS), var_name="factor_name", value_name="factor_value")
    rows = rows.dropna(subset=["factor_value"])
    rows["trade_date"] = pd.to_datetime(rows["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    rows = rows.dropna(subset=["trade_date"])
    rows["factor_version"] = factor_version
    rows["data_version"] = data_version
    rows["run_id"] = run_id
    rows["created_at"] = pd.Timestamp.now()
    return rows[["trade_date", "factor_name", "factor_value", "factor_version", "data_version", "run_id", "created_at"]]


def import_csv(session, args: argparse.Namespace) -> int:
    ensure_schema(session, args.db_path)
    path = args.path or (FACTORS_DIR / "FF5.csv")
    wide = pd.read_csv(path)
    missing = [column for column in ("trade_date", *FF5_FACTORS) if column not in wide.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    wide["trade_date"] = wide["trade_date"].astype(str)
    if args.start_date:
        wide = wide[wide["trade_date"] >= args.start_date]
    if args.end_date:
        wide = wide[wide["trade_date"] <= args.end_date]
    if wide.empty:
        raise RuntimeError("No FF5 CSV rows selected for import.")
    run_id = args.run_id or f"ff5_csv_{dt.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
    rows = _output_rows(wide, args.factor_version, args.data_version, run_id)
    if args.replace_window:
        start = rows["trade_date"].min().strftime("%Y%m%d")
        end = rows["trade_date"].max().strftime("%Y%m%d")
        _delete_market_window(session, args.db_path, list(FF5_FACTORS), args.factor_version, start, end)
    count = _append_market_factors(session, args.db_path, rows)
    print({"market_factor_daily": count, "run_id": run_id})
    return count


def update_rf(session, args: argparse.Namespace) -> int:
    ensure_schema(session, args.db_path)
    client = TushareClient(args.token or load_tushare_token())
    raw = client.request("shibor", start_date=args.start_date, end_date=args.end_date, fields="date,3m")
    if raw.empty:
        raise RuntimeError("Tushare shibor returned no rows.")
    frame = raw.rename(columns={"date": "trade_date", "3m": "rate_value"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame["rate_value"] = pd.to_numeric(frame["rate_value"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "rate_value"])
    frame["daily_rate"] = frame["rate_value"] / 100.0 / TRADING_DAYS
    frame["rate_name"] = RATE_NAME
    frame["source"] = "tushare_shibor"
    frame["data_version"] = args.data_version
    frame["run_id"] = args.run_id or f"rf_shibor_{dt.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
    frame["created_at"] = pd.Timestamp.now()
    if args.replace_window:
        _delete_rate_window(session, args.db_path, RATE_NAME, args.start_date, args.end_date, args.data_version)
    count = _append_rates(session, args.db_path, frame[["trade_date", "rate_name", "rate_value", "daily_rate", "source", "data_version", "run_id", "created_at"]])
    print({"macro_rate_daily": count, "run_id": frame["run_id"].iloc[0]})
    return count


def import_legacy_rf_csv(session, args: argparse.Namespace) -> int:
    ensure_schema(session, args.db_path)
    path = args.path or (BACKTEST_DATA_DIR / "rf.csv")
    frame = pd.read_csv(path, usecols=["trade_date", "rf"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame["daily_rate"] = pd.to_numeric(frame["rf"], errors="coerce") / 100.0
    frame = frame.dropna(subset=["trade_date", "daily_rate"])
    frame = frame[
        (frame["trade_date"] >= pd.Timestamp(args.start_date))
        & (frame["trade_date"] <= pd.Timestamp(args.end_date))
    ]
    if frame.empty:
        raise RuntimeError("No legacy RF rows selected for import.")
    frame["rate_value"] = frame["daily_rate"] * TRADING_DAYS * 100.0
    frame["rate_name"] = RATE_NAME
    frame["source"] = "legacy_shibor_3m_csv"
    frame["data_version"] = args.data_version
    frame["run_id"] = args.run_id or f"rf_legacy_{dt.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
    frame["created_at"] = pd.Timestamp.now()
    if args.replace_window:
        _delete_rate_window(session, args.db_path, RATE_NAME, args.start_date, args.end_date, args.data_version)
    count = _append_rates(
        session,
        args.db_path,
        frame[["trade_date", "rate_name", "rate_value", "daily_rate", "source", "data_version", "run_id", "created_at"]],
    )
    print({"macro_rate_daily": count, "run_id": frame["run_id"].iloc[0]})
    return count

def _load_panel(
    session,
    db_path: str,
    start_date: str,
    end_date: str,
    include_rf: bool,
    rf_data_version: str | None = None,
) -> pd.DataFrame:
    start = _date_literal(start_date)
    end = _date_literal(end_date)
    rf_join = "p = lj(p, rf, `trade_date)" if include_rf else ""
    rf_version_filter = f", data_version=`{rf_data_version}" if rf_data_version else ""
    rf_defs = f"""
        rates = loadTable('{db_path}', `macro_rate_daily)
        rf = select trade_date, last(daily_rate) as rf from rates where rate_name=`{RATE_NAME}{rf_version_filter}, trade_date between {start}:{end} group by trade_date
    """ if include_rf else "rf = table(1:0, `trade_date`rf, [DATE, DOUBLE])"
    df = session.run(
        f"""
        bar = loadTable('{db_path}', `core_market_bar_daily)
        val = loadTable('{db_path}', `core_market_valuation_daily)
        fin = loadTable('{db_path}', `feature_financial_daily)
        m = select b.trade_date as trade_date, b.ts_code as ts_code, b.dret as dret, v.circ_mv as circ_mv
            from bar as b left join val as v on b.trade_date=v.trade_date and b.ts_code=v.ts_code
            where b.trade_date between {start}:{end}
        eq = select trade_date, ts_code, feature_value as equity from fin
            where feature_name=`total_hldr_eqy_exc_min_int, trade_date between {start}:{end}
        roe = select trade_date, ts_code, feature_value as roe from fin
            where feature_name=`roe, trade_date between {start}:{end}
        inv = select trade_date, ts_code, feature_value as inv from fin
            where feature_name=`cashflow_capex_qoq_ttm, trade_date between {start}:{end}
        {rf_defs}
        p = lj(lj(lj(m, eq, `trade_date`ts_code), roe, `trade_date`ts_code), inv, `trade_date`ts_code)
        {rf_join}
        select * from p order by trade_date, ts_code
        """
    )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    return df


def _month_windows(start_date: str, end_date: str, chunk_months: int):
    if chunk_months < 1:
        raise ValueError("chunk_months must be at least 1.")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date.")
    cursor = start.replace(day=1)
    while cursor <= end:
        next_cursor = cursor + pd.DateOffset(months=chunk_months)
        window_start = max(start, cursor)
        window_end = min(end, next_cursor - pd.Timedelta(days=1))
        yield window_start.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")
        cursor = next_cursor


def _calculate_chunk(session, args: argparse.Namespace, start_date: str, end_date: str) -> pd.DataFrame:
    panel = _load_panel(
        session,
        args.db_path,
        start_date,
        end_date,
        include_rf=not args.allow_missing_rf,
        rf_data_version=getattr(args, "rf_data_version", None),
    )
    if panel.empty:
        return pd.DataFrame(columns=["trade_date", *FF5_FACTORS])
    return calculate_ff5(panel, require_rf=not args.allow_missing_rf)


def compute_ff5(session, args: argparse.Namespace) -> int:
    ensure_schema(session, args.db_path)
    run_id = args.run_id or f"ff5_calc_{dt.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
    if args.replace_window:
        _delete_market_window(
            session,
            args.db_path,
            list(FF5_FACTORS),
            args.factor_version,
            args.start_date,
            args.end_date,
        )

    total = 0
    for start_date, end_date in _month_windows(args.start_date, args.end_date, args.chunk_months):
        wide = _calculate_chunk(session, args, start_date, end_date)
        rows = _output_rows(wide, args.factor_version, args.data_version, run_id)
        total += _append_market_factors(session, args.db_path, rows)
        print({"window": f"{start_date}:{end_date}", "market_factor_daily": len(rows), "total": total})
    if not total:
        raise RuntimeError("No FF5 rows were calculated for the requested window.")
    print({"market_factor_daily": total, "run_id": run_id})
    return total


def compare_csv(session, args: argparse.Namespace) -> dict:
    path = args.path or (FACTORS_DIR / "FF5.csv")
    csv = pd.read_csv(path)
    missing = [column for column in ("trade_date", *FF5_FACTORS) if column not in csv.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    csv["trade_date"] = csv["trade_date"].astype(str)
    csv = csv[(csv["trade_date"] >= args.start_date) & (csv["trade_date"] <= args.end_date)]
    if csv.empty:
        raise RuntimeError("No CSV FF5 rows selected for comparison.")

    calculated = []
    for start_date, end_date in _month_windows(args.start_date, args.end_date, args.chunk_months):
        wide = _calculate_chunk(session, args, start_date, end_date)
        if not wide.empty:
            calculated.append(wide)
        print({"window": f"{start_date}:{end_date}", "calculated_dates": len(wide)})
    if not calculated:
        raise RuntimeError("No DB FF5 rows were calculated for comparison.")
    db = pd.concat(calculated, ignore_index=True)

    summaries = {}
    details = []
    for factor in FF5_FACTORS:
        paired = csv[["trade_date", factor]].merge(
            db[["trade_date", factor]],
            on="trade_date",
            how="outer",
            suffixes=("_csv", "_db"),
            indicator=True,
        )
        common = paired.loc[paired["_merge"] == "both"].copy()
        common["abs_error"] = (common[f"{factor}_csv"] - common[f"{factor}_db"]).abs()
        mismatch = common.loc[common["abs_error"] > args.tolerance + 1e-12].copy()
        mismatch.insert(1, "factor_name", factor)
        details.append(mismatch)
        summaries[factor] = {
            "csv_dates": int((paired["_merge"] != "right_only").sum()),
            "db_dates": int((paired["_merge"] != "left_only").sum()),
            "overlap_dates": int(len(common)),
            "csv_only_dates": int((paired["_merge"] == "left_only").sum()),
            "db_only_dates": int((paired["_merge"] == "right_only").sum()),
            "max_abs_error": float(common["abs_error"].max()) if len(common) else None,
            "mean_abs_error": float(common["abs_error"].mean()) if len(common) else None,
            "rmse": float((common["abs_error"] ** 2).mean() ** 0.5) if len(common) else None,
            "over_tolerance_dates": int(len(mismatch)),
        }

    report = {
        "csv_path": str(path),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "chunk_months": args.chunk_months,
        "tolerance": args.tolerance,
        "factors": summaries,
    }
    output = args.output or (PROJECT_ROOT / "docs" / "ff5_db_vs_csv.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path = output.with_suffix(".csv")
    pd.concat(details, ignore_index=True).to_csv(detail_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print({"report": str(output), "details": str(detail_path)})
    return report

def check(session, args: argparse.Namespace) -> None:
    ensure_schema(session, args.db_path)
    start_filter = f", trade_date >= {_date_literal(args.start_date)}" if args.start_date else ""
    end_filter = f", trade_date <= {_date_literal(args.end_date)}" if args.end_date else ""
    result = session.run(
        f"""
        t = loadTable('{args.db_path}', `market_factor_daily)
        base = select trade_date, factor_name from t where factor_version=`{args.factor_version}{start_filter}{end_filter}
        summary = select count(*) as rows, min(trade_date) as start_date, max(trade_date) as end_date from base group by factor_name order by factor_name
        dupBase = select count(*) as cnt from base group by trade_date, factor_name
        dup = select count(*) as duplicate_keys, sum(cnt - 1) as duplicate_rows from dupBase where cnt > 1
        [summary, dup]
        """
    )
    summary, dup = result
    print(summary.to_string(index=False))
    print(dup.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import or calculate FF5 market risk factors into DolphinDB.")
    parser.add_argument("command", choices=["import-csv", "update-rf", "import-rf-csv", "compute", "compare-csv", "check"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--factor-version", default="v1")
    parser.add_argument("--data-version", default="ff5_csv_legacy")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--replace-window", action="store_true")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--allow-missing-rf", action="store_true")
    parser.add_argument("--chunk-months", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rf-data-version", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = connect(args.host, args.port, args.user, args.password)
    if args.command == "import-csv":
        import_csv(session, args)
    elif args.command == "update-rf":
        if not args.start_date or not args.end_date:
            raise ValueError("update-rf requires --start-date and --end-date")
        update_rf(session, args)
    elif args.command == "import-rf-csv":
        if not args.start_date or not args.end_date:
            raise ValueError("import-rf-csv requires --start-date and --end-date")
        import_legacy_rf_csv(session, args)
    elif args.command == "compute":
        if not args.start_date or not args.end_date:
            raise ValueError("compute requires --start-date and --end-date")
        compute_ff5(session, args)
    elif args.command == "compare-csv":
        if not args.start_date or not args.end_date:
            raise ValueError("compare-csv requires --start-date and --end-date")
        compare_csv(session, args)
    elif args.command == "check":
        check(session, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
