from __future__ import annotations

import numpy as np
import pandas as pd

FF5_FACTORS = ("mkt", "smb", "hml", "rmw", "cma")


def _three_way(rank: pd.Series) -> pd.Series:
    bucket = pd.Series(pd.NA, index=rank.index, dtype="object")
    bucket.loc[rank <= 0.30] = "low"
    bucket.loc[(rank > 0.30) & (rank <= 0.70)] = "mid"
    bucket.loc[rank > 0.70] = "high"
    return bucket


def _complete_mean(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.Series:
    values = frame.loc[:, columns]
    return values.mean(axis=1).where(values.notna().all(axis=1))


def _portfolio_grid(frame: pd.DataFrame, bucket_name: str) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([["s", "b"], ["low", "mid", "high"]])
    return (
        frame.dropna(subset=[bucket_name])
        .groupby(["trade_date", "size", bucket_name], observed=True)["dret"]
        .mean()
        .unstack(["size", bucket_name])
        .reindex(columns=columns)
    )


def _spread(grid: pd.DataFrame, positive: list[tuple[str, str]], negative: list[tuple[str, str]]) -> pd.Series:
    return _complete_mean(grid, positive) - _complete_mean(grid, negative)


def _size_spread(grid: pd.DataFrame) -> pd.Series:
    small = _complete_mean(grid, [("s", "low"), ("s", "mid"), ("s", "high")])
    big = _complete_mean(grid, [("b", "low"), ("b", "mid"), ("b", "high")])
    return small - big

def calculate_ff5(panel: pd.DataFrame, require_rf: bool = True) -> pd.DataFrame:
    """Calculate the project's equal-weight 2x3 FF5 portfolios from one date chunk."""
    required = {"trade_date", "dret", "circ_mv", "equity", "roe", "inv"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"FF5 panel missing columns: {missing}")
    if require_rf and "rf" not in panel.columns:
        raise ValueError("FF5 panel is missing rf.")

    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    for column in ("dret", "circ_mv", "equity", "roe", "inv"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "rf" in frame.columns:
        frame["rf"] = pd.to_numeric(frame["rf"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"])
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", *FF5_FACTORS])

    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y%m%d")
    frame["bm"] = frame["equity"] / frame["circ_mv"]
    size_rank = frame.groupby("trade_date")["circ_mv"].rank(pct=True)
    frame["size"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[size_rank <= 0.50, "size"] = "s"
    frame.loc[size_rank > 0.50, "size"] = "b"
    frame["bm_bucket"] = _three_way(frame.groupby("trade_date")["bm"].rank(pct=True))
    frame["roe_bucket"] = _three_way(frame.groupby("trade_date")["roe"].rank(pct=True))
    frame["inv_bucket"] = _three_way(frame.groupby("trade_date")["inv"].rank(pct=True))

    market = frame.dropna(subset=["dret", "circ_mv"])[["trade_date", "dret", "circ_mv"]].copy()
    market["weight"] = market["circ_mv"] / market.groupby("trade_date")["circ_mv"].transform("sum")
    market = (market["dret"] * market["weight"]).groupby(market["trade_date"]).sum().rename("mkt_ret")
    if require_rf:
        rf = frame.groupby("trade_date")["rf"].last()
        missing_rf = rf[rf.isna()]
        if not missing_rf.empty:
            sample = missing_rf.index[:5].tolist()
            raise ValueError(f"Missing rf for dates like {sample}")
    else:
        rf = pd.Series(0.0, index=market.index)

    bm_grid = _portfolio_grid(frame, "bm_bucket")
    roe_grid = _portfolio_grid(frame, "roe_bucket")
    inv_grid = _portfolio_grid(frame, "inv_bucket")
    wide = pd.DataFrame({"mkt": market - rf})
    wide["hml"] = _spread(
        bm_grid,
        [("s", "high"), ("b", "high")],
        [("s", "low"), ("b", "low")],
    )
    wide["rmw"] = _spread(
        roe_grid,
        [("s", "high"), ("b", "high")],
        [("s", "low"), ("b", "low")],
    )
    wide["cma"] = _spread(
        inv_grid,
        [("s", "low"), ("b", "low")],
        [("s", "high"), ("b", "high")],
    )
    wide["smb"] = (
        _size_spread(bm_grid)
        + _size_spread(roe_grid)
        + _size_spread(inv_grid)
    ) / 3.0
    wide = wide.reindex(columns=FF5_FACTORS).round(7)
    return wide.rename_axis("trade_date").reset_index()