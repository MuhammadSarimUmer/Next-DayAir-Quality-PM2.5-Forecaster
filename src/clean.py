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


def load_city_openaq_pm25(city: str) -> pd.Series | None:
    files = sorted(config.RAW_DIR.glob(f"openaq_{city}_*.csv"))
    if not files:
        return None
    series = {}
    for f in files:
        s, stats = clean_sensor(pd.read_csv(f))
        series[f.stem] = s
        print(f"  {f.name}: {stats}")
    return pd.DataFrame(series).mean(axis=1).rename("pm25")


def load_city_kaggle_pm25(city: str) -> tuple[pd.Series | None, pd.DataFrame | None]:
    path = config.RAW_DIR / f"kaggle_{city}_hourly.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["datetime_pkt"])
    bad = (df["pm25"] < config.PM25_MIN) | (df["pm25"] > config.PM25_MAX)
    df.loc[bad, "pm25"] = float("nan")
    pm = df.set_index("datetime_pkt")["pm25"].sort_index()
    pm = pm[~pm.index.duplicated(keep="last")]
    wx = df.set_index("datetime_pkt")
    return pm, wx


def merge_city_pm25(city: str) -> tuple[pd.Series, pd.Series]:
    """OpenAQ ground average when available; otherwise Kaggle API (pre-2025 winters)."""
    pm_oa = load_city_openaq_pm25(city)
    pm_k, kaggle_wx = (None, None)
    if config.USE_KAGGLE_HISTORY:
        pm_k, kaggle_wx = load_city_kaggle_pm25(city)

    if pm_oa is None and pm_k is None:
        raise FileNotFoundError(
            f"No PM2.5 for {city}: run ingest_kaggle and/or OpenAQ collect."
        )

    if pm_k is None:
        return pm_oa, pd.Series("openaq", index=pm_oa.index, dtype=object)

    if pm_oa is None:
        return pm_k, pd.Series("kaggle_api", index=pm_k.index, dtype=object)

    idx = pm_k.index.union(pm_oa.index)
    pm = pm_k.reindex(idx)
    pm.loc[pm_oa.reindex(idx).notna()] = pm_oa.reindex(idx).dropna()
    source = pd.Series("kaggle_api", index=idx, dtype=object)
    source.loc[pm_oa.reindex(idx).notna()] = "openaq"
    print(f"  PM2.5 sources: openaq={int((source == 'openaq').sum())}h, "
          f"kaggle_api={int((source == 'kaggle_api').sum())}h")
    return pm, source


def _meteo_to_pkt(df: pd.DataFrame) -> pd.Series:
    """Open-Meteo rows: PKT-naive local if present, else UTC column converted to PKT."""
    if "datetime_pkt_local" in df.columns:
        return pd.to_datetime(df["datetime_pkt_local"])
    return _to_pkt(df["datetime_utc"])


def load_city_weather(city: str) -> pd.DataFrame:
    w = pd.read_csv(config.RAW_DIR / f"weather_{city}.csv")
    w["datetime_pkt"] = _meteo_to_pkt(w)
    drop_cols = [c for c in ("city", "datetime_utc", "datetime_pkt_local") if c in w.columns]
    w = w.drop(columns=drop_cols).set_index("datetime_pkt")

    cams_path = config.RAW_DIR / f"airquality_{city}.csv"
    if cams_path.exists():
        aq = pd.read_csv(cams_path)
        aq["datetime_pkt"] = _meteo_to_pkt(aq)
        drop_cols = [c for c in ("city", "datetime_utc", "datetime_pkt_local") if c in aq.columns]
        aq = aq.drop(columns=drop_cols).set_index("datetime_pkt")
        w = w.join(aq[config.CAMS_COLS], how="left")
    else:
        print(f"  [warn] {cams_path.name} missing; run: python -m src.collect --cams")
    return w


def longest_gap(s: pd.Series) -> int:
    miss = s.isna()
    run_id = (miss != miss.shift()).cumsum()
    runs = miss.groupby(run_id).sum()
    return int(runs.max()) if len(runs) else 0


def clean_city(city: str) -> tuple[pd.DataFrame, dict]:
    print(f"[{city}]")
    pm, pm_source = merge_city_pm25(city)
    wx = load_city_weather(city)

    df = wx.join(pm.rename("pm25"), how="left")
    df["pm25_source"] = pm_source.reindex(df.index)
    first, last = pm.first_valid_index(), pm.last_valid_index()
    df = df.loc[first:last].copy()

    _, kaggle_wx = load_city_kaggle_pm25(city) if config.USE_KAGGLE_HISTORY else (None, None)
    if kaggle_wx is not None:
        for c in config.WEATHER_VARS:
            if c in kaggle_wx.columns:
                fill = kaggle_wx[c].reindex(df.index)
                df[c] = df[c].fillna(fill)

    raw_missing = df["pm25"].isna()
    df["pm25"] = df["pm25"].interpolate(limit=config.MAX_INTERP_HOURS, limit_area="inside")
    df["pm25_interpolated"] = raw_missing & df["pm25"].notna()
    df[config.WEATHER_VARS] = df[config.WEATHER_VARS].interpolate(
        limit=config.MAX_INTERP_HOURS, limit_area="inside"
    )
    for c in config.CAMS_COLS:
        if c in df.columns:
            df[c] = df[c].interpolate(limit=config.MAX_INTERP_HOURS, limit_area="inside")
            df[c] = df[c].ffill().bfill()

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
    if config.USE_KAGGLE_HISTORY:
        missing = [
            c
            for c in config.CITIES
            if not (config.RAW_DIR / f"kaggle_{c}_hourly.csv").exists()
        ]
        if missing:
            from src import ingest_kaggle

            print("Building Kaggle hourly files:", ", ".join(missing))
            ingest_kaggle.main()

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