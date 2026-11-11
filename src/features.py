"""Build features + 24h-ahead target.

Run:  python -m src.features
Out:  data/processed/features.csv  (ALL columns; FEATURE_COLS says which ones models use)

Every feature at row t uses information available at time t or earlier.
Target = PM2.5 at t + 24h (shift(-24) on a complete hourly index).
"""
import numpy as np
import pandas as pd
from src import config

WIND_DIR = "wind_direction_10m"
HAS_WIND_DIR = WIND_DIR in config.WEATHER_VARS
RAW_WEATHER = [c for c in config.WEATHER_VARS if c != WIND_DIR]

PM_BASE = (
    [f"pm25_lag_{h}h" for h in config.LAG_HOURS]
    + [f"pm25_roll_mean_{w}h" for w in config.ROLL_HOURS]
)
PM_EXTRA = ["pm25_diff_3h", "pm25_roll_std_24h", "pm25_roll_max_24h"]
PM_MOMENTUM = [
    "pm25_change_6h",       # 6h trend direction: rising or falling?
    "pm25_change_24h",      # day-over-day direction: same hour yesterday
    "pm25_rate_of_change",  # 3h velocity in µg/m³ per hour
    "pm25_7d_lag",          # same hour last week: weekly seasonality anchor
    "pm25_roll_min_24h",    # 24h floor: how low did it get recently (recovery signal)
]
DOMAIN_FEATURES = [
    "pm25_acceleration",     # 2nd derivative: is pollution surging or tapering?
    "stagnation_index",      # low wind + high humidity trapping index
    "smog_season_intensity", # Gaussian peak centered at peak crop-burning / winter inversion (Nov 10)
    "lahore_smog_risk",      # City-specific winter stagnation interaction for Lahore
]
# Regime shift: high 24h/72h means vs now (post-smog recovery); CAMS vs ground gap
REGIME_FEATURES = [
    "pm25_lag_excess_24h",
    "pm25_lag_excess_72h",
    "pm25_clearing_index",
    "pm25_cams_divergence",
    "cams_pm25_change_24h",
]
CAMS_FEATURE_COLS = list(config.CAMS_COLS)
WX_EXTRA = ["pressure_change_24h", "wind_speed_roll_mean_24h", "precip_sum_24h"]
WIND_COMPONENTS = ["wind_dir_sin", "wind_dir_cos"]
CAL_BASE = ["hour", "month", "weekday", "is_winter"]
CAL_EXTRA = ["hour_sin", "hour_cos"]
CATEGORICAL_FEATURES = ["city"]
TARGET = "target_pm25_t24h"

FEATURE_COLS = (
    (["pm25_now"] if config.INCLUDE_PM25_NOW else [])
    + PM_BASE + RAW_WEATHER + CAL_BASE + CATEGORICAL_FEATURES
)
if config.EXTRA_FEATURES:
    FEATURE_COLS += (
        PM_EXTRA + PM_MOMENTUM + DOMAIN_FEATURES + REGIME_FEATURES
        + CAMS_FEATURE_COLS + WX_EXTRA + CAL_EXTRA
    )
    if HAS_WIND_DIR:
        FEATURE_COLS += WIND_COMPONENTS

NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_FEATURES]


def _roll(s: pd.Series, w: int):
    """Rolling window that tolerates up to 25% missing hours."""
    return s.rolling(w, min_periods=int(w * 0.75))


def _contiguous_segments(g: pd.DataFrame, max_gap_hours: int = 48) -> list[pd.DataFrame]:
    """Split on long PM outages (e.g. Dec 2024–Jun 2025 gap between Kaggle and OpenAQ)."""
    g = g.sort_values("datetime_pkt").reset_index(drop=True)
    gaps = g["datetime_pkt"].diff()
    seg_id = (gaps > pd.Timedelta(hours=max_gap_hours)).cumsum()
    min_len = max(config.ROLL_HOURS) + config.HORIZON_HOURS + 24
    return [seg for _, seg in g.groupby(seg_id) if len(seg) >= min_len]


def build_city_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("datetime_pkt").reset_index(drop=True)
    gaps = g["datetime_pkt"].diff().dropna()
    assert (gaps == pd.Timedelta(hours=1)).all(), "index must be a complete hourly grid"

    pm = g["pm25"]
    out = pd.DataFrame(
        {"city": g["city"], "datetime_pkt": g["datetime_pkt"], "pm25_now": pm}
    )

    # --- PM2.5 history ---
    for h in config.LAG_HOURS:
        out[f"pm25_lag_{h}h"] = pm.shift(h)
    for w in config.ROLL_HOURS:
        out[f"pm25_roll_mean_{w}h"] = _roll(pm, w).mean()
    out["pm25_diff_3h"] = pm - pm.shift(3)              # short-term trend
    out["pm25_roll_std_24h"] = _roll(pm, 24).std()      # volatility
    out["pm25_roll_max_24h"] = _roll(pm, 24).max()      # recent peak

    # --- PM2.5 momentum (Option 1 improvement: direction + velocity) ---
    out["pm25_change_6h"] = pm - pm.shift(6)                    # 6h trend direction
    out["pm25_change_24h"] = pm - pm.shift(24)                  # day-over-day direction
    out["pm25_rate_of_change"] = (pm - pm.shift(3)) / 3.0      # µg/m³ per hour velocity
    out["pm25_7d_lag"] = pm.shift(168)                          # same hour last week
    out["pm25_roll_min_24h"] = _roll(pm, 24).min()              # recent floor / recovery signal

    # --- Domain physics & smog indicators (Option 2 accuracy boost) ---
    # 2nd derivative: detects whether pollution acceleration is surging into a smog spike
    out["pm25_acceleration"] = (pm - pm.shift(3)) - (pm.shift(3) - pm.shift(6))

    # Stagnation index: low wind + high humidity traps particulate matter
    wind_clipped = g["wind_speed_10m"].clip(lower=0.5)
    out["stagnation_index"] = (1.0 / wind_clipped) * (g["relative_humidity_2m"] / 100.0)

    # Calendar & smog season curve
    t = g["datetime_pkt"]
    doy = t.dt.day_of_year if hasattr(t.dt, "day_of_year") else t.dt.dayofyear
    # Smooth peak centered at peak Punjab crop residue burning (Nov 10 = Day ~314)
    out["smog_season_intensity"] = np.exp(-0.5 * ((doy - 314) / 25.0) ** 2)

    # City-specific winter smog risk: Lahore suffers severe winter inversion compared to coastal Karachi
    is_lahore = (g["city"] == "lahore").astype(float)
    out["lahore_smog_risk"] = is_lahore * out["smog_season_intensity"] * out["stagnation_index"]

    roll24 = out["pm25_roll_mean_24h"]
    roll72 = out["pm25_roll_mean_72h"]
    out["pm25_lag_excess_24h"] = roll24 - pm
    out["pm25_lag_excess_72h"] = roll72 - pm
    falling = np.maximum(-(pm - pm.shift(6)), 0.0)
    out["pm25_clearing_index"] = np.maximum(roll24 - pm, 0.0) * falling

    for c in config.CAMS_COLS:
        if c in g.columns:
            out[c] = g[c]
        else:
            out[c] = np.nan
    out["pm25_cams_divergence"] = pm - out["cams_pm2_5"]
    out["cams_pm25_change_24h"] = out["cams_pm2_5"] - out["cams_pm2_5"].shift(24)

    # --- weather (raw, at time t) ---
    for c in RAW_WEATHER:
        out[c] = g[c]
    if HAS_WIND_DIR:
        rad = np.deg2rad(g[WIND_DIR])                   # direction is circular -> sin/cos
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)

    # --- weather (engineered, past only) ---
    out["pressure_change_24h"] = g["surface_pressure"] - g["surface_pressure"].shift(24)
    out["wind_speed_roll_mean_24h"] = _roll(g["wind_speed_10m"], 24).mean()  # stagnation
    out["precip_sum_24h"] = _roll(g["precipitation"], 24).sum()              # rain washout

    # --- calendar ---
    t = g["datetime_pkt"]
    out["hour"] = t.dt.hour
    out["month"] = t.dt.month
    out["weekday"] = t.dt.day_of_week
    out["is_winter"] = t.dt.month.isin(config.WINTER_MONTHS).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)

    # --- target ---
    out[TARGET] = pm.shift(-config.HORIZON_HOURS)
    return out


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in clean.groupby("city"):
        for seg in _contiguous_segments(g):
            parts.append(build_city_features(seg))
    feats = pd.concat(parts, ignore_index=True)

    # PM2.5-derived columns and the target must exist. Weather NaNs are left for the
    # pipeline's imputer (fitted on train only).
    must_exist = ["pm25_now"] + PM_BASE + PM_EXTRA + PM_MOMENTUM + DOMAIN_FEATURES + [TARGET]
    n0 = len(feats)
    feats = feats.dropna(subset=must_exist)
    print(f"Dropped {n0 - len(feats)} of {n0} rows with missing PM2.5 inputs/target")
    return feats.sort_values(["datetime_pkt", "city"]).reset_index(drop=True)


if __name__ == "__main__":
    clean = pd.read_csv(config.CLEAN_FILE, parse_dates=["datetime_pkt"])
    feats = build_features(clean)
    feats.to_csv(config.FEATURES_FILE, index=False)
    print(feats.groupby("city").size())
    print(f"{len(FEATURE_COLS)} model features: {FEATURE_COLS}")
    print(f"Saved {len(feats)} rows -> {config.FEATURES_FILE}")