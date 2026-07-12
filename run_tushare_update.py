"""Command line entry point for the Tushare end-of-day update job."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data_pipeline.tushare_update import TushareEodUpdater

PROJECT_ROOT = Path(__file__).resolve().parent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _factor_input_ready(updater: TushareEodUpdater, end_date: str, warmup_open_days: int = 120) -> tuple[bool, str]:
    expected = updater.client.open_dates(
        (pd.Timestamp(end_date) - pd.Timedelta(days=240)).strftime("%Y%m%d"),
        end_date,
    )
    if len(expected) < warmup_open_days:
        return False, f"only {len(expected)} expected open dates are available"
    observed = updater.session.run(
        f"t=loadTable('{updater.writer.db_path}', `core_market_bar_daily); select distinct trade_date from t where trade_date between "
        f"{(pd.Timestamp(end_date) - pd.Timedelta(days=240)).strftime('%Y.%m.%d')} : {pd.Timestamp(end_date).strftime('%Y.%m.%d')}"
    )
    observed_dates = set(pd.to_datetime(observed["trade_date"]).dt.strftime("%Y%m%d")) if not observed.empty else set()
    missing = [date for date in expected[-warmup_open_days:] if date not in observed_dates]
    return not missing, ("" if not missing else f"missing {len(missing)} market dates in warmup window, latest={missing[-1]}")


def _run_incremental_factors(updater: TushareEodUpdater, start_date: str, end_date: str) -> dict[str, object]:
    ready, reason = _factor_input_ready(updater, end_date)
    if not ready:
        return {"status": "skipped", "reason": reason}
    start_month = pd.Timestamp(start_date).strftime("%Y-%m")
    end_month = pd.Timestamp(end_date).strftime("%Y-%m")
    command = [
        sys.executable, "-B", "run_alpha_dolphindb.py", "scheduled",
        "--start-month", start_month, "--end-month", end_month,
        "--replace-window", "--data-version", "tushare_eod",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Incremental factor update failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
    return {"status": "success", "start_month": start_month, "end_month": end_month, "output": completed.stdout[-4000:]}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update QuantSystem raw and core data from Tushare.")
    parser.add_argument("command", choices=("market", "financial", "industry", "all"))
    parser.add_argument("--start-date", help="Inclusive YYYYMMDD start date for an explicit backfill window.")
    parser.add_argument("--end-date", help="Inclusive YYYYMMDD end date; defaults to the latest open date.")
    parser.add_argument("--lookback-trading-days", type=int, default=5, help="Revision lookback for normal market updates.")
    parser.add_argument("--max-trading-days", type=int, help="Optional cap for a controlled historical backfill batch.")
    parser.add_argument("--refresh-industry", action="store_true", help="Refresh SW2021 L1 memberships before materializing the requested dates.")
    parser.add_argument("--update-factors", action="store_true", help="After a successful market update, recalculate verified factors for the affected months.")
    parser.add_argument("--financial-periods", type=int, default=12, help="Completed report periods to refresh for financial raw tables.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="123456")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    updater = TushareEodUpdater(args.host, args.port, args.user, args.password)
    output: dict[str, object] = {}

    if args.command in {"market", "all"}:
        window = updater.resolve_window(
            end_date=args.end_date,
            start_date=args.start_date,
            lookback_trading_days=args.lookback_trading_days,
            max_trading_days=args.max_trading_days,
        )
        report = updater.update_market(window, refresh_industry=args.refresh_industry)
        output["market"] = {
            "run_id": report.run_id,
            "trade_dates": report.trade_dates,
            "raw_rows": report.raw_rows,
            "core_rows": report.core_rows,
            "quality_checks_passed": all(item["passed"] for item in report.quality_checks),
        }
        if args.update_factors:
            output["factors"] = _run_incremental_factors(updater, report.trade_dates[0], report.trade_dates[-1])

    if args.command == "industry":
        start_date = args.start_date or "20240101"
        end_date = args.end_date or updater.client.latest_open_date(pd.Timestamp.now().strftime("%Y%m%d"))
        output["industry"] = updater.refresh_industry_history(start_date, end_date)

    if args.command in {"financial", "all"}:
        output["financial_raw"] = updater.update_financial_raw(
            reference_date=args.end_date,
            periods=args.financial_periods,
        )

    print(json.dumps(output, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
