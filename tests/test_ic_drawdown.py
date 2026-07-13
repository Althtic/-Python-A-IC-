from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ic_drawdown import calculate_cumulative_ic_drawdown


def test_drawdown_uses_peak_before_trough() -> None:
    rank_ic = pd.Series(
        [0.10, 0.20, -0.10, -0.15],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    result = calculate_cumulative_ic_drawdown(rank_ic)

    assert result["max_drawdown"] == pytest.approx(0.25)
    assert result["max_drawdown_date"] == pd.Timestamp("2024-01-05")
    assert result["peak_date"] == pd.Timestamp("2024-01-03")
    assert result["max_drawdown_ratio"] == pytest.approx(0.25 / 0.30)


def test_drawdown_ratio_is_unavailable_without_positive_peak() -> None:
    rank_ic = pd.Series([-0.10, -0.20, 0.05], index=pd.date_range("2024-01-02", periods=3))

    result = calculate_cumulative_ic_drawdown(rank_ic)

    assert result["max_drawdown"] == pytest.approx(0.30)
    assert result["max_drawdown_date"] == pd.Timestamp("2024-01-03")
    assert np.isnan(result["max_drawdown_ratio"])


def test_monotonic_cumulative_ic_has_zero_drawdown() -> None:
    rank_ic = pd.Series([0.10, 0.20, 0.15], index=pd.date_range("2024-01-02", periods=3))

    result = calculate_cumulative_ic_drawdown(rank_ic)

    assert result["max_drawdown"] == 0.0
    assert result["max_drawdown_date"] == pd.Timestamp("2024-01-02")
    assert result["max_drawdown_ratio"] == 0.0


def test_recovery_period_counts_observations_after_trough() -> None:
    rank_ic = pd.Series([0.10, 0.20, -0.10, -0.15, 0.30], index=pd.date_range("2024-01-02", periods=5))
    result = calculate_cumulative_ic_drawdown(rank_ic)
    assert result["recovery_date"] == pd.Timestamp("2024-01-06")
    assert result["recovery_days"] == 1
    assert result["recovery_status"] == "recovered"


def test_open_drawdown_is_not_marked_recovered_at_window_end() -> None:
    rank_ic = pd.Series([0.10, 0.20, -0.10, -0.15], index=pd.date_range("2024-01-02", periods=4))
    result = calculate_cumulative_ic_drawdown(rank_ic)
    assert pd.isna(result["recovery_date"])
    assert pd.isna(result["recovery_days"])
    assert result["recovery_status"] == "unrecovered"
