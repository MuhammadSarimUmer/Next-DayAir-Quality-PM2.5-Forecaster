"""Lightweight prediction adjustments for known failure modes (e.g. Lahore smog recovery)."""
import numpy as np
import pandas as pd


def apply_clearing_blend(y_pred, X: pd.DataFrame) -> np.ndarray:
    """
    When rolling means stay high but PM is falling, persistence models over-forecast.
    Blend toward a short-horizon momentum extrapolation for those rows only.
    """
    y_pred = np.asarray(y_pred, dtype=float).copy()
    if not {"pm25_now", "pm25_change_24h", "pm25_roll_mean_24h", "pm25_change_6h"} <= set(X.columns):
        return y_pred

    lag_excess = (X["pm25_roll_mean_24h"] - X["pm25_now"]).to_numpy()
    change_6h = X["pm25_change_6h"].to_numpy()
    clearing = (lag_excess > 45) & (change_6h < -10)
    if "city" in X.columns:
        clearing &= (X["city"].to_numpy() == "lahore")
    if not clearing.any():
        return y_pred

    alt = (X["pm25_now"] + X["pm25_change_24h"]).to_numpy()
    y_pred[clearing] = 0.5 * y_pred[clearing] + 0.5 * alt[clearing]
    return y_pred
