"""Normalize Kaggle 2021-24 hourly air-quality exports into pipeline-ready CSVs.

Run:  python -m src.ingest_kaggle
Out:  data/raw/kaggle_{city}_hourly.csv  (PKT-naive datetime_pkt, µg/m³ pm25)

Source: datasets/2021-24 (OpenWeather-style components.pm2_5 + bundled weather).
Not ground OpenAQ — used to extend Lahore/Karachi winters before Jun 2025.
"""
from pathlib import Path

import pandas as pd

from src import config

KAGGLE_ROOT = config.ROOT / "datasets" / "2021-24"
PM_COLS = ("components.pm2_5", "components_pm2_5")
WEATHER_MAP = {
    "temperature_2m": "temperature_2m",
    "relative_humidity_2m": "relative_humidity_2m",
    "surface_pressure": "surface_pressure",
    "wind_speed_10m": "wind_speed_10m",
    "precipitation": "precipitation",
    "wind_direction_10m": "wind_direction_10m",
}


def _pm_col(df: pd.DataFrame) -> str:
    for c in PM_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"No PM2.5 column in {list(df.columns)}")


def _read_city_frames(city: str) -> list[pd.DataFrame]:
    city = city.lower()
    frames: list[pd.DataFrame] = []
    train_xlsx = KAGGLE_ROOT / "Training" / f"{city}_complete_data.xlsx"
    test_csv = KAGGLE_ROOT / "Testing" / f"{city}_complete_data_july_to_dec_2024.csv"
    if train_xlsx.exists():
        frames.append(pd.read_excel(train_xlsx))
    if test_csv.exists():
        t = pd.read_csv(test_csv)
        t.columns = [c.replace(".", "_") for c in t.columns]
        frames.append(t)
    if not frames:
        return []
    return frames


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" not in out.columns:
        raise ValueError("Missing datetime column")
    if out["datetime"].dtype == object:
        out["datetime_pkt"] = pd.to_datetime(out["datetime"], format="mixed", dayfirst=True)
    else:
        out["datetime_pkt"] = pd.to_datetime(out["datetime"])
    return out


def build_city_kaggle(city: str) -> pd.DataFrame:
    raw_parts = [_parse_datetime(f) for f in _read_city_frames(city)]
    raw = pd.concat(raw_parts, ignore_index=True)
    pm_col = _pm_col(raw)
    raw = raw.sort_values("datetime_pkt").drop_duplicates("datetime_pkt", keep="last")

    out = pd.DataFrame(
        {
            "city": city,
            "datetime_pkt": raw["datetime_pkt"],
            "pm25": pd.to_numeric(raw[pm_col], errors="coerce"),
            "pm25_source": "kaggle_api",
        }
    )
    for src, dst in WEATHER_MAP.items():
        src_key = src.replace("_", ".") if f"{src.replace('_', '.')}" in raw.columns else src
        if src_key in raw.columns:
            out[dst] = raw[src_key]
        elif src in raw.columns:
            out[dst] = raw[src]
    return out.reset_index(drop=True)


def main():
    if not KAGGLE_ROOT.exists():
        raise FileNotFoundError(f"Missing {KAGGLE_ROOT}")

    for city in config.CITIES:
        df = build_city_kaggle(city)
        if df.empty:
            print(f"[{city}] no Kaggle files found, skipping.")
            continue
        out = config.RAW_DIR / f"kaggle_{city}_hourly.csv"
        df.to_csv(out, index=False)
        print(
            f"[{city}] {len(df)} hours "
            f"({df.datetime_pkt.min()} -> {df.datetime_pkt.max()}) -> {out.name}"
        )


if __name__ == "__main__":
    main()
