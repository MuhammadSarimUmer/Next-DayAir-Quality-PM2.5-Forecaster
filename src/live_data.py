"""Fetch the freshest inputs and build one model-ready row (Task 15).

Live inference needs the same feature definitions used in training:

1. Recent hourly PM2.5 from every selected OpenAQ sensor (last ~10 days).
2. Recent/current weather from Open-Meteo's forecast endpoint (past_days).
3. Recent/current CAMS regional PM2.5 and aerosol fields.

The three sources are aligned in Pakistan Standard Time and passed through
``src.features.build_city_features``. The latest complete row is returned.
If any network source or the OpenAQ key is unavailable, the function falls
back to the most recent row in ``data/processed/live_fallback.csv`` (or
``features.csv``) so the demo and tests remain usable offline.

Run:
    python -m src.live_data --city lahore
    python -m src.live_data --city karachi --offline
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from src import config
from src.clean import clean_sensor
from src.features import FEATURE_COLS, build_city_features

LIVE_LOOKBACK_DAYS = 10
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
FALLBACK_FILE = config.PROCESSED_DIR / "live_fallback.csv"


def _pkt_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=config.PKT_TZ).tz_localize(None)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_openaq_pm25(city: str, days: int = LIVE_LOOKBACK_DAYS) -> pd.Series:
    """City-average hourly PM2.5 for the last ``days`` days (PKT index)."""
    from src.openaq_api import get

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    series: dict[str, pd.Series] = {}

    for sensor_id in config.SELECTED_SENSORS.get(city, []):
        rows: list[dict[str, Any]] = []
        page, limit = 1, 1000
        while True:
            data = get(
                f"/sensors/{sensor_id}/hours",
                {
                    "datetime_from": _iso_utc(start),
                    "datetime_to": _iso_utc(now),
                    "limit": limit,
                    "page": page,
                },
            )
            results = data.get("results", [])
            for item in results:
                period = item.get("period") or {}
                rows.append(
                    {
                        "datetime_utc": (period.get("datetimeFrom") or {}).get("utc"),
                        "pm25": item.get("value"),
                        "coverage_pct": (item.get("coverage") or {}).get("percentCoverage"),
                    }
                )
            if len(results) < limit:
                break
            page += 1

        if not rows:
            continue
        try:
            cleaned, _ = clean_sensor(pd.DataFrame(rows))
        except (ValueError, TypeError):
            continue
        if not cleaned.dropna().empty:
            series[str(sensor_id)] = cleaned

    if not series:
        raise RuntimeError(f"No recent OpenAQ PM2.5 data for {city}")

    return pd.DataFrame(series).mean(axis=1).rename("pm25").sort_index()


def fetch_weather(city: str, days: int = LIVE_LOOKBACK_DAYS) -> pd.DataFrame:
    """Weather at hourly resolution, including recent past days (PKT naive)."""
    coords = config.CITIES[city]
    resp = requests.get(
        OPENMETEO_FORECAST,
        params={
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": ",".join(config.WEATHER_VARS),
            "past_days": days,
            "forecast_days": 2,
            "timezone": config.PKT_TZ,
        },
        timeout=60,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly:
        raise RuntimeError("Open-Meteo returned no hourly weather")
    df = pd.DataFrame(hourly).rename(columns={"time": "datetime_pkt"})
    df["datetime_pkt"] = pd.to_datetime(df["datetime_pkt"])
    return df.set_index("datetime_pkt").sort_index()


def fetch_cams(city: str, days: int = LIVE_LOOKBACK_DAYS) -> pd.DataFrame:
    """Regional CAMS PM2.5/dust/AOD, aligned to PKT naive hourly index."""
    coords = config.CITIES[city]
    resp = requests.get(
        config.OPENMETEO_AIR_QUALITY,
        params={
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": ",".join(config.CAMS_API_VARS),
            "past_days": days,
            "forecast_days": 2,
            "timezone": config.PKT_TZ,
        },
        timeout=60,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly:
        raise RuntimeError("Open-Meteo returned no CAMS air-quality data")
    df = pd.DataFrame(hourly).rename(
        columns={
            "time": "datetime_pkt",
            "pm2_5": "cams_pm2_5",
            "dust": "cams_dust",
            "aerosol_optical_depth": "cams_aod",
        }
    )
    df["datetime_pkt"] = pd.to_datetime(df["datetime_pkt"])
    return df.set_index("datetime_pkt").sort_index()


def _merge_sources(
    city: str, pm: pd.Series, weather: pd.DataFrame, cams: pd.DataFrame
) -> pd.DataFrame:
    df = weather.join(cams[config.CAMS_COLS], how="left").join(pm, how="left")
    df["pm25"] = df["pm25"].interpolate(
        limit=config.MAX_INTERP_HOURS, limit_area="inside"
    )
    for col in config.CAMS_COLS:
        df[col] = df[col].interpolate(limit=3, limit_area="inside").ffill().bfill()
    df.insert(0, "city", city)
    return df.rename_axis("datetime_pkt").reset_index()


def _latest_complete_row(feats: pd.DataFrame) -> pd.Series:
    usable = feats.dropna(subset=FEATURE_COLS)
    if usable.empty:
        raise RuntimeError("No row has all model features available")
    return usable.sort_values("datetime_pkt").iloc[-1]


def get_live_features(
    city: str,
    *,
    offline: bool = False,
    prefer_offline: bool = False,
    days: int = LIVE_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Return the freshest complete feature row for ``city``.

    Parameters
    ----------
    city : {"karachi", "lahore"}
    offline : bool
        Skip all network calls and use the saved fallback snapshot.
    prefer_offline : bool
        Try the snapshot first (useful for fast UI reloads).
    days : int
        How much PM2.5 history to request. Must cover the 7-day lag (>= 8).
    """
    city = city.lower().strip()
    if city not in config.CITIES:
        raise ValueError(f"city must be one of {sorted(config.CITIES)}")
    days = max(int(days), 9)

    if not offline and not prefer_offline:
        try:
            pm = fetch_openaq_pm25(city, days)
            weather = fetch_weather(city, days)
            cams = fetch_cams(city, days)
            merged = _merge_sources(city, pm, weather, cams)
            feats = build_city_features(merged)
            row = _latest_complete_row(feats)
            return _result(row, city, "live", pm)
        except Exception as exc:  # noqa: BLE001 - fallback is intentional for demos
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = "offline mode requested"

    try:
        row = _offline_row(city)
        result = _result(row, city, "offline", None)
        result["notes"] = f"Using saved snapshot ({fallback_reason})"
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Live fetch failed ({fallback_reason}); no offline snapshot available either."
        ) from exc


def _offline_row(city: str) -> pd.Series:
    path = FALLBACK_FILE if FALLBACK_FILE.exists() else config.FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing fallback snapshot: {path}")
    df = pd.read_csv(path, parse_dates=["datetime_pkt"])
    df = df[df["city"].str.lower() == city]
    if df.empty:
        raise ValueError(f"No rows for {city} in {path.name}")
    usable = df.dropna(subset=FEATURE_COLS)
    if usable.empty:
        raise ValueError(f"No complete feature rows for {city} in {path.name}")
    return usable.sort_values("datetime_pkt").iloc[-1]


def _result(row: pd.Series, city: str, source: str, pm: pd.Series | None) -> dict[str, Any]:
    ts = pd.Timestamp(row["datetime_pkt"])
    if pm is not None and not pm.dropna().empty:
        observed_at = pm.last_valid_index()
        observed_pm25 = float(pm.loc[observed_at])
    else:
        observed_at = ts
        observed_pm25 = float(row["pm25_now"])

    return {
        "city": city,
        "features": row[FEATURE_COLS].to_frame().T.reset_index(drop=True),
        "feature_timestamp": ts,
        "observed_at": pd.Timestamp(observed_at),
        "observed_pm25": observed_pm25,
        "source": source,
        "notes": "",
        "stale_hours": round((_pkt_now() - ts) / pd.Timedelta(hours=1), 1),
    }


def build_live_fallback_snapshot() -> pd.DataFrame:
    """Save the last complete row per city for offline/HF fallback."""
    df = pd.read_csv(config.FEATURES_FILE, parse_dates=["datetime_pkt"])
    rows = []
    for city in config.CITIES:
        part = df[(df["city"].str.lower() == city)].dropna(subset=FEATURE_COLS)
        if not part.empty:
            rows.append(part.sort_values("datetime_pkt").iloc[-1])
    if not rows:
        raise RuntimeError("No complete rows available for fallback snapshot")
    out = pd.DataFrame(rows)
    out.to_csv(FALLBACK_FILE, index=False)
    print(f"Saved {len(out)} fallback rows -> {FALLBACK_FILE}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch freshest PM2.5 model inputs")
    ap.add_argument("--city", choices=sorted(config.CITIES), default="lahore")
    ap.add_argument("--offline", action="store_true", help="Use saved snapshot")
    ap.add_argument("--prefer-offline", action="store_true")
    ap.add_argument("--refresh-fallback", action="store_true")
    args = ap.parse_args()

    if args.refresh_fallback:
        build_live_fallback_snapshot()

    result = get_live_features(
        args.city, offline=args.offline, prefer_offline=args.prefer_offline
    )
    row = result["features"].iloc[0]
    print(f"City:            {result['city']}")
    print(f"Source:          {result['source']}")
    print(f"Feature time:    {result['feature_timestamp']} PKT")
    print(f"Latest PM2.5:    {result['observed_pm25']:.1f} ug/m3")
    print(f"Staleness:       {result['stale_hours']:.1f} h")
    if result["notes"]:
        print(f"Notes:           {result['notes']}")
    print(f"pm25_now:        {float(row['pm25_now']):.1f}")
    print(f"pm25_roll_24h:   {float(row['pm25_roll_mean_24h']):.1f}")
    print(f"wind_speed_10m:  {float(row['wind_speed_10m']):.1f}")


if __name__ == "__main__":
    main()
