# Data Sources

This project forecasts the PM2.5 concentration 24 hours ahead for Karachi and
Lahore, Pakistan. The training data combines hourly PM2.5 observations from
OpenAQ with hourly weather variables from Open-Meteo.

## 1. Air quality: OpenAQ

- **Source:** [OpenAQ](https://openaq.org/)
- **API:** OpenAQ API v3, `GET /v3/sensors/{sensor_id}/hours`
- **Variable:** PM2.5 hourly mean, in micrograms per cubic metre (ug/m3)
- **Download code:** [`src/collect.py`](./src/collect.py)
- **API client:** [`src/openaq_api.py`](./src/openaq_api.py)
- **Raw data directory:** [`data/raw/`](./data/raw/)
- **Requested period:** 2025-06-23 through 2026-10-31 UTC, as configured in
  [`src/config.py`](./src/config.py)
- **Time handling:** OpenAQ timestamps are retained as UTC in the raw files and
  converted to Pakistan Standard Time (PKT, UTC+05:00) during cleaning.
- **Provider:** The selected sensors are listed by OpenAQ as AirGradient
  sensors.
- **Reference-grade status:** These are community/low-cost monitoring
  stations. This project does not claim that they are reference-grade
  instruments. The model is intended for forecasting and comparative
  analysis, not regulatory compliance measurements.

### Selected sensors

The following sensors are the sensors configured in
[`src/config.py`](./src/config.py). Location and availability metadata came
from the project snapshot [`data/raw/stations_candidates.csv`](./data/raw/stations_candidates.csv).
The `first_utc` and `last_utc` columns below describe the availability reported
in that snapshot; the actual modelling download is limited by the requested
period above.

| City | Location name | OpenAQ location ID | OpenAQ sensor ID | Provider | First data (UTC) | Last data in metadata snapshot (UTC) |
|---|---|---:|---:|---|---|---|
| Karachi | Urban Resource Center | 4791924 | 13368464 | AirGradient | 2025-06-19 08:00 | 2026-11-05 15:00 |
| Karachi | NED University City Campus | 4793473 | 13369654 | AirGradient | 2025-06-19 11:00 | 2026-10-27 13:00 |
| Karachi | Agha Khan University IED | 4793983 | 13370571 | AirGradient | 2025-06-19 13:00 | 2026-11-05 15:00 |
| Karachi | Karachi Metropolitan Corporation | 4803209 | 13377694 | AirGradient | 2025-06-20 07:00 | 2026-11-05 13:00 |
| Karachi | Aman Khan Kaka Khail | 4808325 | 13381251 | AirGradient | 2025-06-20 16:00 | 2026-11-05 14:00 |
| Karachi | NED Main Campus | 4837117 | 13406434 | AirGradient | 2025-06-23 11:00 | 2026-11-05 15:00 |
| Karachi | Urban Unit Karachi | 4838122 | 13407059 | AirGradient | 2025-06-23 13:00 | 2026-11-05 12:00 |
| Karachi | Asif Gulistan-e-Johar | 4838712 | 13408226 | AirGradient | 2025-06-23 16:00 | 2026-11-05 15:00 |
| Karachi | Engr Muzaffar Malir Cantonment, Askari 5 | 4840233 | 13409122 | AirGradient | 2025-06-23 18:00 | 2026-11-05 15:00 |
| Karachi | Orangi Pilot Project | 4847173 | 13414781 | AirGradient | 2025-06-24 07:00 | 2026-11-05 11:00 |
| Karachi | Climate Action Center | 4932783 | 13490916 | AirGradient | 2025-07-02 12:00 | 2026-11-05 15:00 |
| Karachi | G3 Engineering Consultants Karachi | 4985886 | 13533643 | AirGradient | 2025-07-07 05:00 | 2026-11-05 15:00 |
| Lahore | Barki, Lahore | 4515157 | 13137726 | AirGradient | 2025-05-24 17:00 | 2026-11-05 15:00 |
| Lahore | Civil Secretariat | 4527035 | 13153587 | AirGradient | 2025-05-26 07:00 | 2026-11-05 15:00 |
| Lahore | Learning Alliance Intl. DHA | 4527173 | 13154528 | AirGradient | 2025-05-26 09:00 | 2026-11-05 15:00 |
| Lahore | Foot & Mouth Disease Research Centre | 4527306 | 13155469 | AirGradient | 2025-05-26 11:00 | 2026-11-05 15:00 |
| Lahore | HBFC Society, DHA Phase 5 (Hussainy House) | 4527402 | 13156083 | AirGradient | 2025-05-26 13:00 | 2026-11-05 15:00 |
| Lahore | Sandha Road | 4554729 | 13182280 | AirGradient | 2025-05-29 08:00 | 2026-11-05 07:00 |
| Lahore | Ravi Road | 4555745 | 13183459 | AirGradient | 2025-05-29 11:00 | 2026-11-05 15:00 |
| Lahore | Shalimar Town | 4557686 | 13184347 | AirGradient | 2025-05-29 13:00 | 2026-11-05 15:00 |
| Lahore | IEEE, Punjab University, LHR | 4565933 | 13190998 | AirGradient | 2025-05-30 07:00 | 2026-11-05 15:00 |
| Lahore | Model Town | 4568423 | 13192746 | AirGradient | 2025-05-30 12:00 | 2026-11-05 15:00 |
| Lahore | College Road | 4605119 | 13223107 | AirGradient | 2025-06-02 18:00 | 2026-11-05 15:00 |
| Lahore | Kahna Hospital | 4609353 | 13230942 | AirGradient | 2025-06-03 14:00 | 2026-11-05 15:00 |
| Lahore | Gulberg III | 4618814 | 13238783 | AirGradient | 2025-06-04 12:00 | 2026-11-05 15:00 |
| Lahore | Cantonment | 4698865 | 13300980 | AirGradient | 2025-06-11 08:00 | 2026-11-05 15:00 |
| Lahore | Samanabad | 4700982 | 13302769 | AirGradient | 2025-06-11 14:00 | 2026-11-05 15:00 |
| Lahore | Forman Christian College & Chartered University | 4757305 | 13341452 | AirGradient | 2025-06-16 10:00 | 2026-11-05 15:00 |
| Lahore | Thokar Niaz Baig | 5371017 | 13817570 | AirGradient | 2025-08-09 11:00 | 2026-11-02 12:00 |
| Lahore | ARC, Lahore | 6125629 | 14599816 | AirGradient | 2025-11-10 12:00 | 2026-11-05 15:00 |

### Station selection rationale

The project uses multiple sensors in each city rather than treating one
station as representative of the whole city. The selected sensors provide
spatial coverage across Karachi and Lahore and overlap the common modelling
period. For each city and hour, the cleaning step averages the available
quality-controlled sensor values. This reduces dependence on one station and
preserves a city-level forecasting target.

The sensor list is a project selection, not a claim that every sensor is
equivalent or colocated. Sensor outages and differences in siting may still
introduce measurement uncertainty.

## 2. Weather: Open-Meteo Historical Weather API

- **Source:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Download code:** [`src/collect.py`](./src/collect.py)
- **Requested timezone:** GMT/UTC during collection so weather timestamps
  align with OpenAQ raw timestamps.
- **Coordinates:**
  - Karachi: latitude 24.8607, longitude 67.0011
  - Lahore: latitude 31.5204, longitude 74.3587
- **Hourly variables:**
  `temperature_2m`, `relative_humidity_2m`, `surface_pressure`,
  `wind_speed_10m`, `precipitation`, `cloud_cover`, and
  `wind_direction_10m`
- **Licence:** Open-Meteo states that its data is provided under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying
  weather data may have additional attribution requirements; attribution
  should be retained when distributing the application or results.

The historical weather data is reanalysis/archive data. It is suitable for
training, but it is not identical to the forecast weather data that would be
used by a future live-inference implementation. This archive/forecast
distribution difference is a modelling limitation.

## 3. Processing and quality control

Raw files in [`data/raw/`](./data/raw/) are downloaded inputs and are not
modified by the cleaning process.

[`src/clean.py`](./src/clean.py) performs the following operations:

1. Converts raw UTC timestamps to timezone-naive PKT timestamps after applying
   the `Asia/Karachi` timezone.
2. Builds a regular hourly index for each sensor.
3. Removes duplicate timestamps.
4. Marks PM2.5 values below 0 or above 1000 ug/m3 as invalid.
5. Marks hours with less than 50% OpenAQ coverage as missing.
6. Marks runs of eight or more identical readings as a stuck-sensor run.
7. Averages the remaining sensor readings within each city and hour.
8. Interpolates only short internal gaps, up to three hours.
9. Merges PM2.5 and weather on the hourly PKT timestamp.

The processed output is
[`data/processed/clean_hourly.csv`](./data/processed/clean_hourly.csv), and
the quality-control summary is
[`reports/gap_report.csv`](./reports/gap_report.csv).

For the current processed data:

| City | Hourly rows | Processed coverage | PM2.5 missing after cleaning | Longest remaining gap |
|---|---:|---|---:|---:|
| Karachi | 11,903 | 2025-06-23 05:00 to 2026-11-01 03:00 PKT | 78 (0.66%) | 67 hours |
| Lahore | 11,903 | 2025-06-23 05:00 to 2026-11-01 03:00 PKT | 11 (0.09%) | 5 hours |

Weather had zero missing cells in the current quality-control report.

## 4. Reproducibility

The collection period, city coordinates, selected sensor IDs, weather
variables, and quality-control thresholds are defined in
[`src/config.py`](./src/config.py). To reproduce the download and processing
steps after configuring `OPENAQ_API_KEY`:

```text
python -m src.collect
python -m src.clean
python -m src.features
python -m src.split
```

An OpenAQ API key is required for collection and must be stored in a local
`.env` file. Do not commit the key or any other credentials.
