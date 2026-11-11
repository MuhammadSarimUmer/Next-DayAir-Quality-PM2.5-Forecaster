"""Download raw PM2.5 (OpenAQ) and weather (Open-Meteo) data. Raw files are saved unchanged.

Run:
    python -m src.collect              # openaq + weather + CAMS air quality
    python -m src.collect --weather    # weather only
    python -m src.collect --openaq     # pm2.5 only
    python -m src.collect --cams       # CAMS air quality only

All timestamps are stored in UTC here. Conversion to Pakistan Standard Time happens in cleaning.
"""
import argparse
import pandas as pd
import requests
from src import config
from src.openaq_api import get


def download_sensor_hours(city: str, sensor_id: int) -> pd.DataFrame:
    """Hourly PM2.5 for one sensor over the configured date range."""
    rows, page, limit = [], 1, 1000
    while True:
        data = get(
            f"/sensors/{sensor_id}/hours",
            {
                "datetime_from": f"{config.START_DATE}T00:00:00Z",
                "datetime_to": f"{config.END_DATE}T23:59:59Z",
                "limit": limit,
                "page": page,
            },
        )
        results = data.get("results", [])
        for r in results:
            period = r.get("period") or {}
            rows.append(
                {
                    "city": city,
                    "sensor_id": sensor_id,
                    "datetime_utc": ((period.get("datetimeFrom") or {}).get("utc")),
                    "pm25": r.get("value"),
                    "coverage_pct": (r.get("coverage") or {}).get("percentCoverage"),
                }
            )
        print(f"  sensor {sensor_id} page {page}: {len(results)} rows")
        if len(results) < limit:
            break
        page += 1
    return pd.DataFrame(rows)


def collect_openaq():
    for city, sensor_ids in config.SELECTED_SENSORS.items():
        if not sensor_ids:
            print(f"[{city}] no sensors selected in config.py, skipping.")
            continue
        for sid in sensor_ids:
            print(f"[{city}] downloading sensor {sid} ...")
            df = download_sensor_hours(city, sid)
            out = config.RAW_DIR / f"openaq_{city}_{sid}.csv"
            df.to_csv(out, index=False)
            print(f"  saved {len(df)} rows -> {out.name}")


def collect_weather():
    for city, c in config.CITIES.items():
        print(f"[{city}] downloading weather ...")
        resp = requests.get(
            config.OPENMETEO_ARCHIVE,
            params={
                "latitude": c["lat"],
                "longitude": c["lon"],
                "start_date": config.WEATHER_START_DATE,
                "end_date": config.END_DATE,
                "hourly": ",".join(config.WEATHER_VARS),
                "timezone": config.PKT_TZ,
            },
            timeout=300,
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        df = pd.DataFrame(hourly).rename(columns={"time": "datetime_pkt_local"})
        df.insert(0, "city", city)
        out = config.RAW_DIR / f"weather_{city}.csv"
        df.to_csv(out, index=False)
        print(f"  saved {len(df)} rows -> {out.name}")


def collect_cams():
    """Regional PM2.5, dust, and AOD from Open-Meteo CAMS (not derived from local sensors)."""
    rename = {"pm2_5": "cams_pm2_5", "dust": "cams_dust", "aerosol_optical_depth": "cams_aod"}
    for city, c in config.CITIES.items():
        print(f"[{city}] downloading CAMS air quality ...")
        resp = requests.get(
            config.OPENMETEO_AIR_QUALITY,
            params={
                "latitude": c["lat"],
                "longitude": c["lon"],
                "start_date": config.CAMS_START_DATE,
                "end_date": config.END_DATE,
                "hourly": ",".join(config.CAMS_API_VARS),
                "timezone": config.PKT_TZ,
            },
            timeout=120,
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        df = pd.DataFrame(hourly).rename(columns={"time": "datetime_pkt_local"})
        df = df.rename(columns=rename)
        df.insert(0, "city", city)
        out = config.RAW_DIR / f"airquality_{city}.csv"
        df.to_csv(out, index=False)
        print(f"  saved {len(df)} rows -> {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--openaq", action="store_true")
    ap.add_argument("--weather", action="store_true")
    ap.add_argument("--cams", action="store_true")
    args = ap.parse_args()

    run_all = not (args.openaq or args.weather or args.cams)
    if run_all or args.weather:
        collect_weather()
    if run_all or args.cams:
        collect_cams()
    if run_all or args.openaq:
        collect_openaq()