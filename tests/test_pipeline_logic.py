import numpy as np
import pandas as pd
from src import config
from src.features import build_features, FEATURE_COLS, TARGET
from src.split import chronological_split


def make_clean(n=400):
    t = pd.date_range("2025-01-01", periods=n, freq="h")
    d = pd.DataFrame({"city": "karachi", "datetime_pkt": t, "pm25": np.arange(n, dtype=float)})
    for c in config.WEATHER_VARS:
        d[c] = 1.0
    return d


def test_target_and_lags_are_aligned():
    f = build_features(make_clean())
    row = f.iloc[10]
    i = int(row["pm25_now"])
    assert row[TARGET] == i + 24          # label is 24h in the future
    assert row["pm25_lag_1h"] == i - 1    # lag is the past
    assert row["pm25_lag_24h"] == i - 24


def test_feature_cols_unique_and_present():
    f = build_features(make_clean())
    assert len(set(FEATURE_COLS)) == len(FEATURE_COLS)
    assert set(FEATURE_COLS) <= set(f.columns)


def test_split_has_no_overlap_or_label_leak():
    f = build_features(make_clean(2000))
    tr, va, te, t1, t2 = chronological_split(f)
    gap = pd.Timedelta(hours=24)
    assert tr.datetime_pkt.max() + gap < va.datetime_pkt.min()
    assert va.datetime_pkt.max() + gap < te.datetime_pkt.min()
    assert len(tr) > len(va) and len(tr) > len(te)