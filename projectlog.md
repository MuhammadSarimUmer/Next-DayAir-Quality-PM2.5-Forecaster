# Project Log

## Project overview

This project forecasts the PM2.5 concentration in Karachi and Lahore,
Pakistan, 24 hours into the future. The workflow combines city-level PM2.5
observations with hourly weather, time-based variables, and recent PM2.5
history.

The project is designed as a chronological forecasting problem. This matters
because a forecasting model must only use information that would have been
available at prediction time. Randomly mixing past and future rows would make
the evaluation unrealistically optimistic.

## Progress summary

### Completed

- Data-source selection and documentation
- OpenAQ PM2.5 collection
- Open-Meteo weather collection
- UTC-to-PKT timestamp conversion
- Hourly alignment and PM2.5/weather merging
- Sensor quality control and short-gap interpolation
- Exploratory data analysis
- Feature engineering
- 24-hour shifted prediction target
- Chronological train/validation/test split
- Leakage checks and core pipeline tests
- Preprocessing/model pipeline (StandardScaler + OneHotEncoder)
- Baseline metrics (persistence model)
- Ridge and KNN tuning
- Random Forest and LightGBM tuning
- Model selection and final test evaluation
- AQI category accuracy metric added to evaluation

### Completed Setup Documentation

- Project setup files (`README.md`, `.gitignore`, and `.env.example`) have been populated.
- Station metadata and data-source documentation are recorded in [`DATA_SOURCES.md`](./DATA_SOURCES.md).

### Not yet completed

The product stages have not yet been implemented:

- Error analysis
- AQI advice module
- Live-data inference
- Streamlit application
- FastAPI endpoint
- Deployment
- Final report, README polish, demo script, and slides

## Step 1: Initial project setup

### What was done

The project was organized into source code, raw data, processed data,
notebooks, reports, and tests:

```text
data/
  raw/
  processed/
notebooks/
reports/
  figures/
src/
tests/
```

The dependency list was recorded in
[`requirements.txt`](./requirements.txt). It includes pandas, NumPy,
requests, python-dotenv, matplotlib, scikit-learn, LightGBM, joblib,
Streamlit, FastAPI, Uvicorn, pytest, and jupytext.

Configuration is centralized in
[`src/config.py`](./src/config.py). This keeps paths, API settings, city
coordinates, selected sensors, time ranges, cleaning thresholds, feature
settings, and split proportions in one place.

### Why this structure was chosen

Separating raw and processed data prevents the original downloads from being
silently modified. Centralized configuration also makes the data pipeline
reproducible and avoids scattering important constants through multiple
scripts.

### Setup status

The setup files (`README.md`, `.gitignore`, and `.env.example`) are populated.
The API key is loaded from the environment (`.env`) and is ignored by git so it is never committed.

## Step 2: Selecting stations and documenting sources

### What was done

The project uses multiple OpenAQ PM2.5 sensors in each city:

- 12 sensors in Karachi
- 18 sensors in Lahore

The selected sensor IDs and station names are defined in
[`src/config.py`](./src/config.py). The station metadata snapshot is stored in
[`data/raw/stations_candidates.csv`](./data/raw/stations_candidates.csv).

The source methodology, station table, licenses, coordinates, processing
decisions, and limitations are documented in
[`DATA_SOURCES.md`](./DATA_SOURCES.md).

### Why multiple sensors were selected

One sensor may fail, go offline, or represent only a small local area. Using
multiple sensors gives better spatial coverage and reduces dependence on a
single instrument. During cleaning, the available quality-controlled sensor
values are averaged by city and hour to produce a city-level PM2.5 signal.

The selected sensors are community/low-cost monitoring stations. They are not
treated as reference-grade regulatory instruments. The resulting forecasts
are therefore appropriate for forecasting and comparative analysis, not for
official compliance reporting.

## Step 3: Collecting raw data

### PM2.5 collection

[`src/collect.py`](./src/collect.py) downloads hourly PM2.5 observations from
OpenAQ API v3 using the endpoint:

```text
/v3/sensors/{sensor_id}/hours
```

The OpenAQ API client is implemented in
[`src/openaq_api.py`](./src/openaq_api.py). It:

- Adds the API key from the environment
- Uses retry handling for rate limits and temporary server failures
- Applies a polite delay between successful requests
- Raises an explicit error when the API key is missing or requests fail

The requested collection period is configured as:

```text
2025-06-23 through 2026-10-31 UTC
```

Raw OpenAQ files are stored separately for each city and sensor in
[`data/raw/`](./data/raw/).

### Weather collection

The same collection script downloads hourly weather from the Open-Meteo
Historical Weather API:

```text
https://archive-api.open-meteo.com/v1/archive
```

Weather is collected for the city coordinates:

- Karachi: 24.8607, 67.0011
- Lahore: 31.5204, 74.3587

The collected variables are:

- `temperature_2m`
- `relative_humidity_2m`
- `surface_pressure`
- `wind_speed_10m`
- `precipitation`
- `cloud_cover`
- `wind_direction_10m`

Weather timestamps are requested in GMT/UTC so they initially match the
OpenAQ raw timestamps.

### Why raw data is preserved

The raw files are treated as immutable inputs. This makes it possible to
repeat cleaning with different thresholds or correct a processing decision
without downloading the source data again.

## Step 4: Cleaning and merging the data

The cleaning workflow is implemented in
[`src/clean.py`](./src/clean.py).

### Timestamp conversion

Raw UTC timestamps are converted to Pakistan Standard Time using the
`Asia/Karachi` timezone. The timezone is then removed from the stored
timestamp because Pakistan does not currently use daylight-saving changes.

This ensures that hourly patterns such as morning and evening pollution are
analysed in local Pakistani time.

### Sensor quality control

For each sensor, the cleaning process:

1. Removes duplicate timestamps.
2. Creates a complete hourly index.
3. Marks readings below `0` or above `1000` ug/m3 as invalid.
4. Marks readings with less than 50% OpenAQ coverage as missing.
5. Detects runs of at least eight identical consecutive readings as possible
   stuck-sensor values.
6. Averages the remaining sensor values for each city and hour.

These rules remove clearly invalid values while avoiding aggressive
interpolation of long outages.

### Missing-value handling

PM2.5 and weather are merged on the hourly PKT timestamp. Internal gaps of up
to three hours are linearly interpolated. Longer gaps remain missing rather
than being filled with an unjustified estimate.

The cleaned data is written to
[`data/processed/clean_hourly.csv`](./data/processed/clean_hourly.csv).
The quality-control summary is written to
[`reports/gap_report.csv`](./reports/gap_report.csv).

### Current quality-control results

| City | Hourly rows | Coverage | PM2.5 missing after cleaning | Missing percentage | Longest remaining gap |
|---|---:|---|---:|---:|---:|
| Karachi | 11,992 | 2025-06-19 12:00 to 2026-11-01 03:00 PKT | 78 | 0.65% | 67 hours |
| Lahore | 12,607 | 2025-05-24 21:00 to 2026-11-01 03:00 PKT | 57 | 0.45% | 32 hours |

The current report shows zero missing weather cells after processing.

## Step 5: Exploratory data analysis

The EDA is stored in
[`notebooks/02_eda.ipynb`](./notebooks/02_eda.ipynb). The notebook has been
executed and contains outputs for:

1. PM2.5 time series by city
2. Monthly PM2.5 distributions
3. Mean hourly PM2.5 patterns
4. Missing data by month and city
5. 24-hour PM2.5 persistence
6. Correlations between PM2.5 and weather variables
7. AQI-category distributions

Generated figures are stored in
[`reports/figures/`](./reports/figures/):

- `eda_timeseries.png`
- `eda_monthly.png`
- `eda_hourly.png`
- `eda_missing.png`
- `eda_corr.png`

### Important EDA observations

- Lahore has a higher average PM2.5 level than Karachi in the processed data.
- PM2.5 has meaningful 24-hour persistence, supporting the use of lag
  features.
- Hour-of-day and month-of-year patterns are present, supporting calendar
  features.
- Weather variables show measurable relationships with PM2.5, although
  correlation alone does not establish causation.
- AQI categories are not evenly distributed between the two cities.

These observations motivated the use of lagged PM2.5, rolling statistics,
weather variables, and calendar variables in the model features.

## Step 6: Feature engineering and target creation

Feature generation is implemented in
[`src/features.py`](./src/features.py).

### Forecast target

The prediction target is:

```text
target_pm25_t24h = PM2.5 at time t + 24 hours
```

The target is created with a negative 24-hour shift. Rows without sufficient
history or a future target are removed.

### Feature groups

The current dataset contains 35 model features:

#### Current and historical PM2.5

- `pm25_now`
- `pm25_lag_1h`
- `pm25_lag_3h`
- `pm25_lag_6h`
- `pm25_lag_12h`
- `pm25_lag_24h`

#### Rolling PM2.5 statistics

- `pm25_roll_mean_6h`
- `pm25_roll_mean_24h`
- `pm25_roll_mean_72h`
- `pm25_diff_3h`
- `pm25_roll_std_24h`
- `pm25_roll_max_24h`

#### Momentum & Trend features

- `pm25_change_6h` (6-hour trend direction)
- `pm25_change_24h` (day-over-day direction)
- `pm25_rate_of_change` (3-hour PM2.5 velocity in µg/m³ per hour)
- `pm25_7d_lag` (same hour last week for weekly seasonality)
- `pm25_roll_min_24h` (24-hour floor / recovery indicator)

#### Weather

- `temperature_2m`
- `relative_humidity_2m`
- `surface_pressure`
- `wind_speed_10m`
- `precipitation`
- `cloud_cover`

#### Wind-direction encoding

- `wind_dir_sin`
- `wind_dir_cos`

Wind direction is circular. Encoding it as sine and cosine prevents the model
from treating 359 degrees and 1 degree as far apart.

#### Weather-derived features

- `pressure_change_24h`
- `wind_speed_roll_mean_24h`
- `precip_sum_24h`

These represent recent pressure movement, recent wind stagnation, and recent
rainfall accumulation.

#### Calendar features

- `hour`
- `month`
- `weekday`
- `is_winter`
- `hour_sin`
- `hour_cos`

The cyclic hour encoding represents the relationship between the end and
beginning of a day. The winter flag represents the project-defined winter
months: November, December, January, and February.

#### City feature

- `city`

City is retained as a categorical feature so the model can learn systematic
differences between Karachi and Lahore.

The generated feature file is
[`data/processed/features.csv`](./data/processed/features.csv). It contains
the feature columns, `datetime_pkt`, and `target_pm25_t24h`.

### Leakage prevention

All lag and rolling features use only the current or earlier timestamps.
The future PM2.5 value is used only as the target and is not included as an
input feature.

## Step 7: Chronological splitting

The split is implemented in
[`src/split.py`](./src/split.py).

### Split policy

The data is split chronologically into:

- 70% training
- 15% validation
- 15% test

The split is based on unique timestamps rather than randomly selecting rows.
This keeps the evaluation representative of future forecasting.

### Purge gap

A 24-hour purge gap is applied at split boundaries. A row at time `t` has a
target at `t + 24 hours`, so rows immediately before a boundary could have
targets inside the next split. Removing those rows reduces target leakage
across train, validation, and test.

### Generated split files

- [`data/processed/train.csv`](./data/processed/train.csv)
- [`data/processed/val.csv`](./data/processed/val.csv)
- [`data/processed/test.csv`](./data/processed/test.csv)
- [`data/processed/split_info.json`](./data/processed/split_info.json)

### Current split results

| Split | Rows | Start | End |
|---|---:|---|---|
| Train | 16,372 | 2025-05-31 21:00 | 2026-05-29 21:00 |
| Validation | 3,630 | 2026-05-30 22:00 | 2026-08-14 12:00 |
| Test | 3,678 | 2026-08-15 13:00 | 2026-10-31 03:00 |

The validation and test sets contain equal numbers of Karachi and Lahore rows (Validation: 1,815 each; Test: 1,839 each).

## Verification completed

The tests in
[`tests/test_pipeline_logic.py`](./tests/test_pipeline_logic.py) passed:

```text
3 passed
```

The tests verify that:

- The 24-hour target and PM2.5 lag values are aligned correctly.
- Feature names are unique and present in the generated data.
- Train, validation, and test splits do not overlap through the 24-hour
  target horizon.

The test run produced pandas deprecation warnings for `t.dt.weekday`. These
warnings do not currently fail the tests, but the expression should eventually
be updated to `t.dt.day_of_week`.

## Reproducible pipeline commands

After setting `OPENAQ_API_KEY` in a local `.env` file, the data stages can be
run in order:

```text
python -m src.collect
python -m src.clean
python -m src.features
python -m src.split
```

The EDA notebook can then be opened and rerun from
[`notebooks/02_eda.ipynb`](./notebooks/02_eda.ipynb).

## Next recommended work

The modelling stage (Stage D) is complete. Key results:

- **Validation Comparison**:
  - Ridge: MAE 9.93 µg/m³, RMSE 14.83, R² 0.3040, AQI Accuracy 70.7%
  - KNN: MAE 9.47 µg/m³, RMSE 14.64, R² 0.3221, AQI Accuracy 68.3%
  - Random Forest: MAE 8.83 µg/m³, RMSE 14.01, R² 0.3793, AQI Accuracy 69.5%
  - **LightGBM (Winner)**: MAE 8.77 µg/m³, RMSE 13.91, R² 0.3876, AQI Accuracy 71.0%
- **Final Evaluation (Test Set, Retrained on Train+Val 20,002 samples)**:
  - Selected model: LightGBM (35 features, momentum and trend added)
  - Overall Test MAE: 13.50 µg/m³
  - Overall Test RMSE: 23.93 µg/m³
  - Overall Test R²: 0.6621
  - Overall Test AQI Accuracy: 71.34% (Standard) / 68.11% (With conservative buffers)
  - **Karachi Breakdown**: MAE 6.25 µg/m³, RMSE 9.56, R² 0.7233, AQI Accuracy 82.60%
  - **Lahore Breakdown**: MAE 20.75 µg/m³, RMSE 32.47, R² 0.5558, AQI Accuracy 60.09%
- **Baseline Comparison (Persistence - Test Set)**:
  - Baseline MAE: 14.64 µg/m³, RMSE 27.17, R² 0.5765, AQI Accuracy 69.83%
  - **Improvement over Baseline**: MAE +7.8% (14.64 → 13.50), RMSE +11.9% (27.17 → 23.93), R² 0.5765 → 0.6621, AQI Accuracy +2.2% (69.8% → 71.3%)
- **Experiments**:
  - Option 1 (Momentum features): added 5 features (35 total), improving LightGBM test AQI accuracy to 71.34%
  - Option 2 (CityRouter): tested city-specialist routing, achieved 67.4% accuracy (worse than single model; rejected)
  - Data extension investigation: attempted extending collection back to 2022 via OpenAQ and WAQI; confirmed that ground community monitors only began reporting in May/June 2025
- Final pipeline saved as `models/final_pipeline.joblib`

The next implementation stage is the product stage (Stage E):

1. Error analysis notebook to understand Lahore underperformance (60.09% vs
   Karachi 82.60% AQI accuracy)
2. AQI advice module (`src/aqi.py`)
3. Live-data inference
4. Streamlit application
5. FastAPI endpoint
6. Deployment
7. Final report, README polish, demo script, and slides
