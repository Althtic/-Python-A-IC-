from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DATA_DIR = PROJECT_ROOT / "回测数据集"
FINANCIAL_INDICATOR_DIR = PROJECT_ROOT / "A股上市财务指标数据"
BALANCE_SHEET_DIR = PROJECT_ROOT / "A股上市资产负债表数据"
CASHFLOW_DIR = PROJECT_ROOT / "A股上市现金流量表数据"
OHLC_RAW_DIR = PROJECT_ROOT / "A股日度OHLC数据"


def backtest_data_path(filename: str) -> Path:
    return BACKTEST_DATA_DIR / filename


def financial_indicator_path(filename: str) -> Path:
    return FINANCIAL_INDICATOR_DIR / filename


def balance_sheet_path(filename: str) -> Path:
    return BALANCE_SHEET_DIR / filename


def cashflow_path(filename: str) -> Path:
    return CASHFLOW_DIR / filename


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

