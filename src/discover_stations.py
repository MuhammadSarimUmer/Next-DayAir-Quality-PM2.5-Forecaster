"""Find candidate PM2.5 stations near each city so we can pick which ones to use.

Run:  python -m src.discover_stations
Output: data/raw/stations_candidates.csv  (look at it, then fill SELECTED_SENSORS in config.py)
"""
import pandas as pd
from src import config
from src.openaq_api import get

PM25_PARAMETER_ID = 2  # OpenAQ's id for pm25


def find_stations(city: str, lat: float, lon: float, radius_m: int = 25000) -> list[dict]:
    rows, page = [], 1
    while True:
        data = get(
            "/locations",
            {
                "coordinates": f"{lat},{lon}",
                "radius": radius_m,
                "parameters_id": PM25_PARAMETER_ID,
                "limit": 100,
                "page": page,
            },
        )
        results = data.get("results", [])
        for loc in results:
            pm25_sensors = [
                s for s in loc.get("sensors", [])
                if (s.get("parameter") or {}).get("name") == "pm25"
            ]
            for s in pm25_sensors:
                rows.append(
                    {
                        "city": city,
                        "location_id": loc.get("id"),
                        "location_name": loc.get("name"),
                        "provider": (loc.get("provider") or {}).get("name"),
                        "is_monitor": loc.get("isMonitor"),
                        "sensor_id": s.get("id"),
                        "first_utc": ((loc.get("datetimeFirst") or {}).get("utc")),
                        "last_utc": ((loc.get("datetimeLast") or {}).get("utc")),
                        "lat": (loc.get("coordinates") or {}).get("latitude"),
                        "lon": (loc.get("coordinates") or {}).get("longitude"),
                    }
                )
        if len(results) < 100:
            break
        page += 1
    return rows


if __name__ == "__main__":
    all_rows = []
    for city, c in config.CITIES.items():
        print(f"Searching {city} ...")
        all_rows += find_stations(city, c["lat"], c["lon"])

    df = pd.DataFrame(all_rows)
    out = config.RAW_DIR / "stations_candidates.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {len(df)} candidate sensors to {out}")