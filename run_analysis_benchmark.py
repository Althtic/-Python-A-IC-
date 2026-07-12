"""Compare the Python and DolphinDB engines on the same factor-analysis window."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from web_backend.api import factor_ic, factor_quantile


PROJECT_ROOT = Path(__file__).resolve().parent


def _run(engine: str, args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    ic = factor_ic(
        factor_name=args.factor_name,
        factor_version=args.factor_version,
        start_date=args.start_date,
        end_date=args.end_date,
        holding_period=args.holding_period,
        ic_ma_window=args.ic_ma_window,
        analysis_engine=engine,
    )
    ic_seconds = time.perf_counter() - started
    started = time.perf_counter()
    quantile = factor_quantile(
        factor_name=args.factor_name,
        factor_version=args.factor_version,
        start_date=args.start_date,
        end_date=args.end_date,
        layers=args.layers,
        holding_period=args.holding_period,
        industry_grouping=False,
        analysis_engine=engine,
    )
    quantile_seconds = time.perf_counter() - started
    return {"ic": ic, "quantile": quantile, "seconds": {"ic": ic_seconds, "quantile": quantile_seconds, "total": ic_seconds + quantile_seconds}}


def _series_difference(left: list[dict], right: list[dict], key: str) -> float | None:
    left_by_date = {row["trade_date"]: row.get(key) for row in left}
    right_by_date = {row["trade_date"]: row.get(key) for row in right}
    values = [abs(float(left_by_date[date]) - float(right_by_date[date])) for date in left_by_date.keys() & right_by_date.keys() if left_by_date[date] is not None and right_by_date[date] is not None]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-name", default="alpha_60")
    parser.add_argument("--factor-version", default="v1")
    parser.add_argument("--start-date", default="20190101")
    parser.add_argument("--end-date", default="20190331")
    parser.add_argument("--holding-period", type=int, default=5)
    parser.add_argument("--ic-ma-window", type=int, default=20)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "analysis_engine_benchmark.json")
    args = parser.parse_args()

    python = _run("python", args)
    dolphindb = _run("dolphindb", args)
    result = {
        "window": {"factor": args.factor_name, "version": args.factor_version, "start_date": args.start_date, "end_date": args.end_date, "holding_period": args.holding_period, "layers": args.layers},
        "python": {"seconds": python["seconds"], "ic_summary": python["ic"]["summary"], "quantile_summary": python["quantile"]["summary"]},
        "dolphindb": {"seconds": dolphindb["seconds"], "ic_summary": dolphindb["ic"]["summary"], "quantile_summary": dolphindb["quantile"]["summary"]},
        "difference": {
            "ic_rank_mean": dolphindb["ic"]["summary"]["rank_ic_mean"] - python["ic"]["summary"]["rank_ic_mean"],
            "ic_series_max_abs": _series_difference(python["ic"]["series"], dolphindb["ic"]["series"], "Rank_IC"),
            "quantile_ew_ls_final": dolphindb["quantile"]["summary"]["ls_ew_final"] - python["quantile"]["summary"]["ls_ew_final"],
            "quantile_vw_ls_final": dolphindb["quantile"]["summary"]["ls_vw_final"] - python["quantile"]["summary"]["ls_vw_final"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())