from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FACTORS_DIR = PROJECT_ROOT / "Factors"
BACKTEST_DATA_DIR = PROJECT_ROOT / "回测数据集"
RESULTS_DIR = PROJECT_ROOT / "因子检验结果"
REPORTS_DIR = PROJECT_ROOT / "因子分析报告"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def factor_path(filename: str) -> Path:
    return FACTORS_DIR / filename


def backtest_data_path(filename: str) -> Path:
    return BACKTEST_DATA_DIR / filename


def result_dir(factor_name: str) -> Path:
    return ensure_dir(RESULTS_DIR / factor_name)


def report_path(filename: str) -> Path:
    return ensure_dir(REPORTS_DIR) / filename

