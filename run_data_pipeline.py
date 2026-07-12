import argparse
import logging

from data_pipeline.financial import (
    load_and_clean_disclosure_files,
    run_cashflow_qoq_pipeline,
    run_disclosure_alignment,
)
from data_pipeline.market import run_market_pipeline
from data_pipeline.paths import backtest_data_path, balance_sheet_path, cashflow_path, financial_indicator_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _clean_roe(sample_rows):
    if sample_rows is not None:
        raise ValueError("clean-roe does not support --sample-rows because it scans disclosure source files.")
    return load_and_clean_disclosure_files(
        source_dir=financial_indicator_path(""),
        date_col="ann_date",
        value_col="roe",
        output_path=financial_indicator_path("roe_data_clean.csv"),
    )


def _clean_equity(sample_rows):
    if sample_rows is not None:
        raise ValueError("clean-equity does not support --sample-rows because it scans disclosure source files.")
    return load_and_clean_disclosure_files(
        source_dir=balance_sheet_path(""),
        date_col="f_ann_date",
        value_col="total_hldr_eqy_exc_min_int",
        output_path=balance_sheet_path("归母股东权益_clean.csv"),
    )


def _clean_cashflow(sample_rows):
    if sample_rows is not None:
        raise ValueError("clean-cashflow does not support --sample-rows because it scans disclosure source files.")
    return load_and_clean_disclosure_files(
        source_dir=cashflow_path(""),
        date_col="f_ann_date",
        value_col="c_pay_acq_const_fiolta",
        output_path=cashflow_path("构建固定资产等支付的现金_clean.csv"),
    )


STEPS = {
    "market": lambda sample_rows: run_market_pipeline(
        audit_path=None if sample_rows is not None else backtest_data_path("market_panel_audit.json"),
        sample_rows=sample_rows,
    ),
    "clean-roe": _clean_roe,
    "clean-equity": _clean_equity,
    "clean-cashflow": _clean_cashflow,
    "cashflow-qoq": lambda sample_rows: run_cashflow_qoq_pipeline(),
    "align-roe": lambda sample_rows: run_disclosure_alignment(
        cleaned_disclosure_path=financial_indicator_path("roe_data_clean.csv"),
        output_path=backtest_data_path("roe.csv"),
        announcement_col="ann_date",
        value_columns=["roe"],
        audit_path=None if sample_rows is not None else backtest_data_path("roe_audit.json"),
        sample_rows=sample_rows,
    ),
    "align-equity": lambda sample_rows: run_disclosure_alignment(
        cleaned_disclosure_path=balance_sheet_path("归母股东权益_clean.csv"),
        output_path=backtest_data_path("归母股东权益.csv"),
        announcement_col="f_ann_date",
        value_columns=["total_hldr_eqy_exc_min_int"],
        audit_path=None if sample_rows is not None else backtest_data_path("equity_audit.json"),
        sample_rows=sample_rows,
    ),
    "align-cashflow-qoq": lambda sample_rows: run_disclosure_alignment(
        cleaned_disclosure_path=cashflow_path("qoq.csv"),
        output_path=backtest_data_path("环比购买固定资产支出增长率(TTM).csv"),
        announcement_col="f_ann_date",
        value_columns=["qoq"],
        audit_path=None if sample_rows is not None else backtest_data_path("cashflow_qoq_audit.json"),
        sample_rows=sample_rows,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-factor raw data preprocessing steps.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS),
        default=["market"],
        help="Data preprocessing steps to run. Factor calculation is intentionally excluded.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only the first N rows for a fast smoke test. Output CSVs are not overwritten in sample mode.",
    )
    args = parser.parse_args()

    for step in args.steps:
        logging.info("Running data step: %s", step)
        result = STEPS[step](args.sample_rows)
        logging.info("Finished %s: %s rows x %s columns", step, len(result), len(result.columns))


if __name__ == "__main__":
    main()

