from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
REPAIR_PID = 94204
START_DATE = "20171010"
END_DATE = "20251231"
RF_VERSION = "ff5_csv_rf_v2"
FACTOR_VERSION = "v2"
DATA_VERSION = "ff5_db_stream_v2"
LOG = LOG_DIR / "ff5_after_repairs_20260711.monitor.log"
REPORT = ROOT / "docs" / "ff5_db_vs_csv_v2_20171010_20251231.json"


def log(message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def pid_running(pid: int) -> bool:
    command = f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"
    return subprocess.run(["powershell", "-NoProfile", "-Command", command]).returncode == 0


def run(name: str, args: list[str]) -> None:
    out = LOG_DIR / f"ff5_after_repairs_{name}.out.log"
    err = LOG_DIR / f"ff5_after_repairs_{name}.err.log"
    command = [sys.executable, "-B", "run_ff5_dolphindb.py", *args]
    log("running " + " ".join(command))
    with out.open("w", encoding="utf-8") as out_handle, err.open("w", encoding="utf-8") as err_handle:
        result = subprocess.run(command, cwd=ROOT, stdout=out_handle, stderr=err_handle, text=True)
    log(f"{name} returncode={result.returncode} stdout={out.name} stderr={err.name}")
    if result.returncode:
        raise RuntimeError(f"{name} failed")


def validate_report() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    failures = {
        name: summary
        for name, summary in report["factors"].items()
        if summary["csv_only_dates"]
        or summary["db_only_dates"]
        or summary["over_tolerance_dates"]
    }
    if failures:
        raise RuntimeError(f"FF5 CSV/DB parity failed: {failures}")


def main() -> int:
    try:
        log(f"waiting for repair monitor PID {REPAIR_PID}")
        while pid_running(REPAIR_PID):
            time.sleep(60)
        repair_log = LOG_DIR / "repair_after_91308_restart.out.log"
        if "all repaired factors completed" not in repair_log.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError("five-factor repair monitor did not report completion")
        run("import_rf", [
            "import-rf-csv", "--start-date", START_DATE, "--end-date", END_DATE,
            "--data-version", RF_VERSION, "--replace-window",
        ])
        run("compare", [
            "compare-csv", "--path", "Factors/FF5_fixed_20260711.csv",
            "--start-date", START_DATE, "--end-date", END_DATE,
            "--chunk-months", "1", "--rf-data-version", RF_VERSION,
            "--output", str(REPORT.relative_to(ROOT)),
        ])
        validate_report()
        run("compute", [
            "compute", "--start-date", START_DATE, "--end-date", END_DATE,
            "--chunk-months", "1", "--rf-data-version", RF_VERSION,
            "--factor-version", FACTOR_VERSION, "--data-version", DATA_VERSION,
            "--replace-window",
        ])
        run("check", ["check", "--start-date", START_DATE, "--end-date", END_DATE, "--factor-version", FACTOR_VERSION])
        log("FF5 v2 streaming ingest completed")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())