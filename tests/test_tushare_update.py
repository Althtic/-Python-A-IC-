import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.tushare_update import _normalise_financial_frame, _quarter_periods


class FinancialDateTests(unittest.TestCase):
    def test_only_completed_report_periods_are_selected(self) -> None:
        self.assertEqual(
            _quarter_periods("20260711", 4),
            ["20260331", "20251231", "20250930", "20250630"],
        )

    def test_missing_announcement_date_uses_conservative_deadline(self) -> None:
        source = pd.DataFrame(
            [{"ts_code": "000001.SZ", "ann_date": None, "end_date": "20251231", "roe": 0.1}]
        )

        result = _normalise_financial_frame(source, ["roe"], "test")

        self.assertEqual(result.loc[0, "ann_date"], pd.Timestamp("2026-04-30"))
        self.assertEqual(result.loc[0, "end_date"], pd.Timestamp("2025-12-31"))


if __name__ == "__main__":
    unittest.main()
