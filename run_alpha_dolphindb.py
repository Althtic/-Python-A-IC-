"""Calculate legacy WQ alpha factors from DolphinDB and write standardized factor_daily rows."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.dolphindb_sync import DEFAULT_DB_PATH, _append, connect

PROJECT_ROOT = Path(__file__).resolve().parent
FACTOR_CALCULATE_DIR = PROJECT_ROOT / "Factor_Calculate"
if str(FACTOR_CALCULATE_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOR_CALCULATE_DIR))

from factor_neutralization import neutralize_factor  # noqa: E402
from factor_suspension_processing import remove_resume_window_data  # noqa: E402
from factor_winsorize import winsorize_factor  # noqa: E402

# alpha_61 is a renamed legacy Alpha_60 formula retained as a separate factor.
FACTOR_SPECS = {
    "alpha_01": ("WQ_Alpha01_numba", "process_alpha_01_features"),
    "alpha_02": ("WQ_Alpha02", "calculate_alpha_02"),
    "alpha_03": ("WQ_Alpha03", "calculate_alpha_03"),
    "alpha_04": ("WQ_Alpha04_numba", "calculate_alpha_04"),
    "alpha_05": ("WQ_Alpha05", "calculate_alpha_05"),
    "alpha_06": ("WQ_Alpha06_numba", "calculate_alpha_06"),
    "alpha_07": ("WQ_Alpha07", "calculate_alpha_07"),
    "alpha_08": ("WQ_Alpha08", "calculate_alpha_08"),
    "alpha_09": ("WQ_Alpha09", "calculate_alpha_09"),
    "alpha_11": ("WQ_Alpha11", "calculate_alpha_11"),
    "alpha_12": ("WQ_Alpha12", "calculate_alpha_12"),
    "alpha_13": ("WQ_Alpha13", "calculate_alpha_13"),
    "alpha_14": ("WQ_Alpha14", "calculate_alpha_14"),
    "alpha_15": ("WQ_Alpha15", "calculate_alpha_15"),
    "alpha_16": ("WQ_Alpha16", "calculate_alpha_16"),
    "alpha_17": ("WQ_Alpha17", "calculate_alpha_17"),
    "alpha_18": ("WQ_Alpha18", "calculate_alpha_18"),
    "alpha_23": ("WQ_Alpha23", "calculate_alpha_23"),
    "alpha_25": ("WQ_Alpha25", "calculate_alpha_25"),
    "alpha_26": ("WQ_Alpha26", "calculate_alpha"),
    "alpha_27": ("WQ_Alpha27", "calculate_alpha"),
    "alpha_28": ("WQ_Alpha28", "calculate_alpha"),
    "alpha_30": ("WQ_Alpha30", "calculate_alpha"),
    "alpha_32": ("WQ_Alpha32", "calculate_alpha_32"),
    "alpha_33": ("WQ_Alpha33", "calculate_alpha_33"),
    "alpha_34": ("WQ_Alpha34", "calculate_alpha"),
    "alpha_35": ("WQ_Alpha35", "calculate_alpha"),
    "alpha_37": ("WQ_Alpha37", "calculate_alpha"),
    "alpha_45": ("WQ_Alpha45", "calculate_alpha"),
    "alpha_49": ("WQ_Alpha49", "calculate_alpha"),
    "alpha_50": ("WQ_Alpha50", "calculate_alpha"),
    "alpha_61": ("WQ_Alpha61", "calculate_alpha"),
}

MIN_WARMUP_DAYS = {"alpha_37": 300}

# These formulas completed the 2017-2025 full-history validation run. The
# remaining legacy formulas are excluded until repaired and revalidated.
AUTO_UPDATE_FACTORS = (
    "alpha_01", "alpha_02", "alpha_03", "alpha_04", "alpha_05", "alpha_06", "alpha_07", "alpha_08", "alpha_09",
    "alpha_11", "alpha_12", "alpha_13", "alpha_14", "alpha_15", "alpha_16", "alpha_17", "alpha_18",
    "alpha_23", "alpha_25", "alpha_26", "alpha_27", "alpha_28", "alpha_30", "alpha_32", "alpha_33", "alpha_34", "alpha_35", "alpha_37", "alpha_45",
    "alpha_49", "alpha_50", "alpha_61",
)

MARKET_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "dret", "vol", "amount",
    "industry_name", "suspend_type", "circ_mv",
]


def list_months(start_month: str, end_month: str) -> list[str]:
    return [str(period) for period in pd.period_range(start=start_month, end=end_month, freq="M")]


def load_market_window(session, db_path: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start = start_date.strftime("%Y.%m.%d")
    end = end_date.strftime("%Y.%m.%d")
    df = session.run(
        f"""
        bar = loadTable('{db_path}', `core_market_bar_daily)
        status = loadTable('{db_path}', `core_market_status_daily)
        val = loadTable('{db_path}', `core_market_valuation_daily)
        ind = loadTable('{db_path}', `core_industry_sw_l1_daily)
        select b.ts_code as ts_code, b.trade_date as trade_date, b.open as open, b.high as high,
               b.low as low, b.close as close, b.pre_close as pre_close, b.dret as dret, b.vol as vol,
               b.amount as amount, i.industry_name as industry_name, s.suspend_type as suspend_type,
               v.circ_mv as circ_mv
        from bar as b
        left join status as s on b.trade_date=s.trade_date and b.ts_code=s.ts_code
        left join val as v on b.trade_date=v.trade_date and b.ts_code=v.ts_code
        left join ind as i on b.trade_date=i.trade_date and b.ts_code=i.ts_code
        where b.trade_date between {start} : {end}
        order by b.ts_code, b.trade_date
        """
    )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    for column in ("ts_code", "industry_name"):
        df[column] = df[column].astype(str).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    df["suspend_type"] = df["suspend_type"].astype(str).replace({"": "N", "nan": "N", "None": "N"})
    return df[MARKET_COLUMNS]


def _formula(factor_name: str):
    module_name, function_name = FACTOR_SPECS[factor_name]
    return getattr(importlib.import_module(module_name), function_name)


def _standardize(calculated: pd.DataFrame, market: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    if factor_name not in calculated.columns:
        raise ValueError(f"Formula output does not contain {factor_name}; columns={list(calculated.columns)}")
    raw = calculated[["trade_date", "ts_code", factor_name]].copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.strftime("%Y%m%d")
    raw = raw.rename(columns={factor_name: "raw_factor"})
    metadata = market[["trade_date", "ts_code", "industry_name", "suspend_type", "circ_mv"]].drop_duplicates(
        ["trade_date", "ts_code"]
    )
    out = raw.merge(metadata, on=["trade_date", "ts_code"], how="left")
    out["raw_factor"] = pd.to_numeric(out["raw_factor"], errors="coerce")
    out[factor_name] = out["raw_factor"].replace([np.inf, -np.inf], np.nan)
    out = out.drop(columns=["raw_factor"])
    out = out.dropna(subset=[factor_name, "industry_name", "circ_mv"])
    out = remove_resume_window_data(out, window=10)
    out = winsorize_factor(out)
    out = neutralize_factor(out, target_factor=factor_name)
    return out.dropna(subset=[factor_name])


def _factor_rows(df: pd.DataFrame, factor_name: str, factor_version: str, data_version: str, run_id: str) -> pd.DataFrame:
    out = df[["trade_date", "ts_code", factor_name, "raw_factor"]].rename(
        columns={factor_name: "factor_value", "raw_factor": "raw_value"}
    )
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    out["factor_name"] = factor_name
    out["factor_version"] = factor_version
    out["data_version"] = data_version
    out["neutralized"] = True
    out["winsorized"] = True
    out["run_id"] = run_id
    out["created_at"] = pd.Timestamp.now()
    return out[["trade_date", "ts_code", "factor_name", "factor_value", "raw_value", "factor_version", "data_version", "neutralized", "winsorized", "run_id", "created_at"]]


def _delete_existing(session, db_path: str, factor_name: str, factor_version: str) -> None:
    session.run(f"t=loadTable('{db_path}', `factor_daily); delete from t where factor_name=`{factor_name}, factor_version=`{factor_version}")


def _delete_existing_window(
    session, db_path: str, factor_name: str, factor_version: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> None:
    start = start_date.strftime("%Y.%m.%d")
    end = end_date.strftime("%Y.%m.%d")
    session.run(
        f"t=loadTable('{db_path}', `factor_daily); delete from t where factor_name=`{factor_name}, factor_version=`{factor_version}, trade_date between {start}:{end}"
    )


def calculate_factor(session, args: argparse.Namespace, factor_name: str) -> int:
    formula = _formula(factor_name)
    run_id = f"{factor_name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    if args.replace:
        _delete_existing(session, args.db_path, factor_name, args.factor_version)
    total = 0
    for month in list_months(args.start_month, args.end_month):
        month_start = pd.Period(month, freq="M").start_time
        month_end = pd.Period(month, freq="M").end_time.normalize()
        warmup_days = max(args.warmup_days, MIN_WARMUP_DAYS.get(factor_name, 0))
        market = load_market_window(session, args.db_path, month_start - pd.Timedelta(days=warmup_days), month_end)
        if market.empty:
            continue
        calculated = formula(market.copy())
        standardized = _standardize(calculated, market, factor_name)
        in_month = standardized["trade_date"].between(month_start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d"))
        rows = _factor_rows(standardized.loc[in_month], factor_name, args.factor_version, args.data_version, run_id)
        if args.replace_window and not rows.empty:
            _delete_existing_window(session, args.db_path, factor_name, args.factor_version, month_start, month_end)
        total += _append(session, args.db_path, "factor_daily", rows)
        print(f"{factor_name} {month}: wrote {len(rows):,}; total {total:,}", flush=True)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factor", choices=[*FACTOR_SPECS, "all", "scheduled"])
    parser.add_argument("--start-month", default="2017-10")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--factor-version", default="v1")
    parser.add_argument("--data-version", default="dolphindb_core_v1")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--replace-window", action="store_true", help="Replace only the requested month windows after successful calculation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)
    session = connect(args.host, args.port, args.user, args.password)
    factors = FACTOR_SPECS if args.factor == "all" else (AUTO_UPDATE_FACTORS if args.factor == "scheduled" else [args.factor])
    completed, failed = {}, {}
    for factor_name in factors:
        try:
            completed[factor_name] = calculate_factor(session, args, factor_name)
        except Exception as exc:
            failed[factor_name] = str(exc)
            print(f"{factor_name} failed: {exc}", flush=True)
    print({"completed": completed, "failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())