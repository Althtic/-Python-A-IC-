"""Shared cumulative IC drawdown calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_cumulative_ic_drawdown(rank_ic: pd.Series) -> dict[str, Any]:
    """Return cumulative IC, maximum drawdown, and its recovery metadata."""
    values = pd.to_numeric(rank_ic, errors="coerce").astype(float)
    values = values[np.isfinite(values)]
    if values.empty:
        return {
            "cumulative_ic": values,
            "max_drawdown": np.nan,
            "max_drawdown_date": pd.NaT,
            "max_drawdown_ratio": np.nan,
            "peak_date": pd.NaT,
            "peak_value": np.nan,
            "recovery_date": pd.NaT,
            "recovery_days": np.nan,
            "recovery_status": "unrecovered",
        }

    cumulative_ic = values.cumsum()
    # Cumulative IC starts from zero, so zero is eligible as a peak.
    running_peak = cumulative_ic.cummax().clip(lower=0.0)
    drawdown = cumulative_ic - running_peak
    trough_position = int(np.argmin(drawdown.to_numpy()))
    trough_date = cumulative_ic.index[trough_position]
    max_drawdown = float(-drawdown.iloc[trough_position])
    peak_value = float(running_peak.iloc[trough_position])
    if peak_value > 0:
        peak_slice = cumulative_ic.iloc[:trough_position + 1]
        peak_date = peak_slice[peak_slice == peak_value].index[-1]
    else:
        # The cumulative series has an implicit zero baseline before its
        # first observation. Use the first date to make an open negative
        # drawdown visible in charts, while keeping its ratio undefined.
        peak_date = cumulative_ic.index[0]

    # Recovery is the first observation after the trough that reaches the
    # exact peak level used by this drawdown.
    recovery_date = pd.NaT
    recovery_days = np.nan
    recovery_status = "unrecovered"
    if max_drawdown <= 0:
        recovery_date = trough_date
        recovery_days = 0
        recovery_status = "recovered"
    else:
        recovered = np.flatnonzero(cumulative_ic.iloc[trough_position + 1:].to_numpy() >= peak_value)
        if recovered.size:
            recovery_position = trough_position + 1 + int(recovered[0])
            recovery_date = cumulative_ic.index[recovery_position]
            recovery_days = recovery_position - trough_position
            recovery_status = "recovered"

    return {
        "cumulative_ic": cumulative_ic,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": trough_date,
        "max_drawdown_ratio": max_drawdown / peak_value if peak_value > 0 else np.nan,
        "peak_date": peak_date,
        "peak_value": peak_value,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
        "recovery_status": recovery_status,
    }