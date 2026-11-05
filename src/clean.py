"""Clean raw data and merge PM2.5 + weather into one hourly table per city (PKT).

Run:  python -m src.clean
Out:  data/processed/clean_hourly.csv, reports/gap_report.csv
"""
import pandas as pd
from src import config


def _to_pkt(series_utc: pd.Series) -> pd.Series:
    """UTC strings -> tz-naive Pakistan Standard Time (no DST in Pakistan, so this is safe)."""
    return (
        pd.to_datetime(series_utc, utc=True)
        .dt.tz_convert(config.PKT_TZ)
        .dt.tz_localize(None)
    )


def _mask_stuck(s: pd.Series, n: int) -> pd.Series:
    """Set runs of >= n identical consecutive values to NaN (stuck sensor)."""
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    return s.mask((run_len >= n) & s.notna())


def clean_sensor(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """One sensor's raw hourly rows -> cleaned, regular hourly Series (index = PKT)."""
    df = df.copy()
    df["datetime_pkt"] = _to_pkt(df["datetime_utc"])
    df = df.drop_duplicates("datetime_pkt").sort_values("datetime_pkt")
    stats = {"raw_rows": len(df)}

    # 1. low-coverage hours (coverage may be missing -> comparison is False -> kept)
    low_cov = df["coverage_pct"] < config.MIN_COVERAGE_PCT
    df.loc[low_cov, "pm25"] = float("nan")
    stats["low_coverage"] = int(low_cov.sum())

    # 2. impossible values
    bad = (df["pm25"] < config.PM25_MIN) | (df["pm25"] > config.PM25_MAX)
    df.loc[bad, "pm25"] = float("nan")
    stats["out_of_range"] = int(bad.sum())

    # 3. stuck values (needs a regular hourly index first)
    s = df.set_index("datetime_pkt")["pm25"]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    before = s.notna().sum()
    s = _mask_stuck(s, config.STUCK_HOURS)
    stats["stuck"] = int(before - s.notna().sum())
    return s, stats


def load_city_pm25(city: str) -> pd.Series:
    files = sorted(config.RAW_DIR.glob(f"openaq_{city}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw OpenAQ files for {city}. Run collect.py first.")
    series = {}
    for f in files:
        s, stats = clean_sensor(pd.read_csv(f))
        series[f.stem] = s
        print(f"  {f.name}: {stats}")
    # average across sensors per hour (NaN only if every sensor is missing)
    return pd.DataFrame(series).mean(axis=1).rename("pm25")


def load_city_weather(city: str) -> pd.DataFrame:
    w = pd.read_csv(config.RAW_DIR / f"weather_{city}.csv")
    w["datetime_pkt"] = _to_pkt(w["datetime_utc"])
    return w.drop(columns=["city", "datetime_utc"]).set_index("datetime_pkt")


def longest_gap(s: pd.Series) -> int:
    miss = s.isna()
    run_id = (miss != miss.shift()).cumsum()
    runs = miss.groupby(run_id).sum()
    return int(runs.max()) if len(runs) else 0


def clean_city(city: str) -> tuple[pd.DataFrame, dict]:
    print(f"[{city}]")
    pm = load_city_pm25(city)
    wx = load_city_weather(city)

    df = wx.join(pm, how="left")                    # weather index is the complete hourly grid
    first, last = pm.first_valid_index(), pm.last_valid_index()
    df = df.loc[first:last].copy()                  # trim empty edges

    raw_missing = df["pm25"].isna()
    df["pm25"] = df["pm25"].interpolate(limit=config.MAX_INTERP_HOURS, limit_area="inside")
    df["pm25_interpolated"] = raw_missing & df["pm25"].notna()
    df[config.WEATHER_VARS] = df[config.WEATHER_VARS].interpolate(
        limit=config.MAX_INTERP_HOURS, limit_area="inside"
    )

    report = {
        "city": city,
        "hours": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "pm25_missing_before_fill": int(raw_missing.sum()),
        "pm25_interpolated": int(df["pm25_interpolated"].sum()),
        "pm25_still_missing": int(df["pm25"].isna().sum()),
        "pm25_still_missing_pct": round(100 * df["pm25"].isna().mean(), 2),
        "longest_pm25_gap_hours": longest_gap(df["pm25"]),
        "weather_missing_cells": int(df[config.WEATHER_VARS].isna().sum().sum()),
    }
    df.insert(0, "city", city)
    return df.rename_axis("datetime_pkt").reset_index(), report


if __name__ == "__main__":
    frames, reports = [], []
    for city in config.CITIES:
        d, r = clean_city(city)
        frames.append(d)
        reports.append(r)

    clean = pd.concat(frames, ignore_index=True)
    clean.to_csv(config.CLEAN_FILE, index=False)
    rep = pd.DataFrame(reports)
    rep.to_csv(config.GAP_REPORT_FILE, index=False)
    print("\n", rep.T.to_string())
    print(f"\nSaved {len(clean)} rows -> {config.CLEAN_FILE}")