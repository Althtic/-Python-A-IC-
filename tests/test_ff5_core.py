import numpy as np
import pandas as pd
import pytest

from ff5_core import calculate_ff5


def _bucket(value):
    if value <= 6:
        return "low"
    if value <= 14:
        return "mid"
    return "high"


def _mean_return(frame, size, bucket_column, bucket):
    return frame.loc[
        (frame["size"] == size) & (frame[bucket_column] == bucket), "dret"
    ].mean()


def _size_spread(frame, bucket_column):
    small = np.mean([_mean_return(frame, "s", bucket_column, bucket) for bucket in ("low", "mid", "high")])
    big = np.mean([_mean_return(frame, "b", bucket_column, bucket) for bucket in ("low", "mid", "high")])
    return small - big


def test_calculate_ff5_matches_equal_weight_2x3_formula():
    bm = [1, 2, 3, 7, 8, 9, 10, 15, 16, 17, 4, 5, 6, 11, 12, 13, 14, 18, 19, 20]
    roe = [20, 19, 18, 14, 13, 12, 11, 6, 5, 4, 17, 16, 15, 10, 9, 8, 7, 3, 2, 1]
    inv = [15, 1, 7, 16, 2, 8, 17, 3, 9, 10, 18, 4, 11, 19, 5, 12, 20, 6, 13, 14]
    rows = []
    for index in range(20):
        rows.append(
            {
                "trade_date": "20200102",
                "dret": (index + 1) / 1000.0,
                "circ_mv": index + 1.0,
                "equity": bm[index] * (index + 1.0),
                "roe": roe[index],
                "inv": inv[index],
                "rf": 0.0001,
            }
        )
    panel = pd.DataFrame(rows)
    expected = panel.copy()
    expected["size"] = np.where(expected.index < 10, "s", "b")
    expected["bm_bucket"] = [_bucket(value) for value in bm]
    expected["roe_bucket"] = [_bucket(value) for value in roe]
    expected["inv_bucket"] = [_bucket(value) for value in inv]

    hml = np.mean([_mean_return(expected, size, "bm_bucket", "high") for size in ("s", "b")]) - np.mean([_mean_return(expected, size, "bm_bucket", "low") for size in ("s", "b")])
    rmw = np.mean([_mean_return(expected, size, "roe_bucket", "high") for size in ("s", "b")]) - np.mean([_mean_return(expected, size, "roe_bucket", "low") for size in ("s", "b")])
    cma = np.mean([_mean_return(expected, size, "inv_bucket", "low") for size in ("s", "b")]) - np.mean([_mean_return(expected, size, "inv_bucket", "high") for size in ("s", "b")])
    smb = np.mean([_size_spread(expected, column) for column in ("bm_bucket", "roe_bucket", "inv_bucket")])
    mkt = np.average(expected["dret"], weights=expected["circ_mv"]) - 0.0001

    actual = calculate_ff5(panel).iloc[0]
    assert actual["mkt"] == pytest.approx(round(mkt, 7))
    assert actual["smb"] == pytest.approx(round(smb, 7))
    assert actual["hml"] == pytest.approx(round(hml, 7))
    assert actual["rmw"] == pytest.approx(round(rmw, 7))
    assert actual["cma"] == pytest.approx(round(cma, 7))