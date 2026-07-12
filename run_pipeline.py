import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_DIR = PROJECT_ROOT / "功能模块"
FACTOR_DIR = PROJECT_ROOT / "Factor_Calculate"


STEPS = {
    "alpha60": FACTOR_DIR / "WQ_Alpha60.py",
    "validation": MODULE_DIR / "ValidationTest.py",
    "quantile": MODULE_DIR / "QuantileSpreadTest.py",
    "regression": MODULE_DIR / "RegressionAnalysis.py",
    "report": MODULE_DIR / "DeepSeekAnalyzer.py",
}


def run_script(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script}")
    subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QuantSystem research pipeline.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS),
        default=["alpha60", "validation", "quantile", "regression"],
        help="Pipeline steps to run. The report step requires DEEPSEEK_API_KEY.",
    )
    args = parser.parse_args()

    for step in args.steps:
        print(f"\n=== Running {step}: {STEPS[step]} ===")
        run_script(STEPS[step])


if __name__ == "__main__":
    main()

