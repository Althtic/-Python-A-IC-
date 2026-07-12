from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
SCHEDULED_PID = 91308
SCHEDULED_OUT = LOG_DIR / "industry_factor_rebuild_20260711_135112.out.log"
SCHEDULED_ERR = LOG_DIR / "industry_factor_rebuild_20260711_135112.err.log"
FACTORS = ["alpha_07", "alpha_25", "alpha_28", "alpha_35", "alpha_37"]
DATA_VERSION = "repaired_alpha_20260711"
FACTOR_VERSION = "v1"
START_MONTH = "2017-10"
END_MONTH = "2025-12"

sys.path.insert(0, str(ROOT))
from data_pipeline.dolphindb_sync import DEFAULT_DB_PATH, connect  # noqa: E402

MONITOR_LOG = LOG_DIR / f"repair_after_91308_{datetime.now():%Y%m%d_%H%M%S}.monitor.log"


def log(message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pid_running(pid: int) -> bool:
    command = f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def wait_for_scheduled() -> None:
    log(f"waiting for PID {SCHEDULED_PID}")
    while pid_running(SCHEDULED_PID):
        time.sleep(60)
    log(f"PID {SCHEDULED_PID} exited")


def read_scheduled_summary() -> dict:
    lines = SCHEDULED_OUT.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{'completed':") or line.startswith('{"completed":'):
            return ast.literal_eval(line)
    raise RuntimeError(f"scheduled summary not found in {SCHEDULED_OUT}")


def assert_scheduled_ok() -> None:
    summary = read_scheduled_summary()
    failed = summary.get("failed") or {}
    completed = summary.get("completed") or {}
    log(f"scheduled completed factors={len(completed)} failed={failed}")
    if failed:
        raise RuntimeError(f"scheduled run has failed factors: {failed}")


def run_factor(factor: str) -> None:
    out_path = LOG_DIR / f"repair_{factor}_{datetime.now():%Y%m%d_%H%M%S}.out.log"
    err_path = LOG_DIR / f"repair_{factor}_{datetime.now():%Y%m%d_%H%M%S}.err.log"
    cmd = [
        sys.executable,
        "-B",
        "run_alpha_dolphindb.py",
        factor,
        "--start-month",
        START_MONTH,
        "--end-month",
        END_MONTH,
        "--replace-window",
        "--data-version",
        DATA_VERSION,
    ]
    log(f"running {' '.join(cmd)}")
    with out_path.open("w", encoding="utf-8") as out_fh, err_path.open("w", encoding="utf-8") as err_fh:
        result = subprocess.run(cmd, cwd=ROOT, stdout=out_fh, stderr=err_fh, text=True)
    log(f"{factor} returncode={result.returncode} stdout={out_path.name} stderr={err_path.name}")
    if result.returncode != 0:
        raise RuntimeError(f"{factor} repair failed, see {out_path} and {err_path}")


def _scalar(frame: pd.DataFrame, column: str, default=0):
    if frame.empty or column not in frame.columns:
        return default
    value = frame.iloc[0][column]
    if pd.isna(value):
        return default
    return value


def check_factor(factor: str) -> dict:
    session = connect("127.0.0.1", 8848, "admin", "123456")
    script = f"""
    t = loadTable('{DEFAULT_DB_PATH}', `factor_daily)
    base = select trade_date, ts_code from t where factor_name=`{factor}, factor_version=`{FACTOR_VERSION}
    summary = select count(*) as rows, min(trade_date) as start_date, max(trade_date) as end_date from base
    grouped = select count(*) as cnt from base group by trade_date, ts_code
    dup = select count(*) as duplicate_keys, sum(cnt - 1) as duplicate_rows from grouped where cnt > 1
    [summary, dup]
    """
    summary, dup = session.run(script)
    rows = int(_scalar(summary, "rows", 0))
    start_date = _scalar(summary, "start_date", None)
    end_date = _scalar(summary, "end_date", None)
    duplicate_keys = int(_scalar(dup, "duplicate_keys", 0))
    duplicate_rows = int(_scalar(dup, "duplicate_rows", 0))
    end_month = pd.to_datetime(end_date).strftime("%Y-%m") if end_date is not None else None
    result = {
        "factor": factor,
        "rows": rows,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "end_month": end_month,
        "duplicate_keys": duplicate_keys,
        "duplicate_rows": duplicate_rows,
    }
    log("check " + json.dumps(result, ensure_ascii=False))
    if rows <= 0:
        raise RuntimeError(f"{factor} has no rows in factor_daily")
    if end_month != "2025-12":
        raise RuntimeError(f"{factor} coverage does not reach 2025-12: {end_date}")
    if duplicate_keys or duplicate_rows:
        raise RuntimeError(f"{factor} has duplicate (trade_date, ts_code): keys={duplicate_keys}, rows={duplicate_rows}")
    return result


def main() -> int:
    try:
        wait_for_scheduled()
        log("alpha_03 was repaired separately after the scheduled lock failure; continuing the requested five-factor repair.")
        results = []
        for factor in FACTORS:
            run_factor(factor)
            results.append(check_factor(factor))
        log("all repaired factors completed " + json.dumps(results, ensure_ascii=False))
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
