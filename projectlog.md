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
- Deployable final ensemble trained and saved
- Error analysis report, tables, and figures
- AQI breakpoint bug fixed and per-breakpoint tests added
- Live-data inference with offline fallback
- Streamlit application
- FastAPI endpoints
- Hugging Face Space deployment bundle
- Final report, limitations/ethics update, README polish
- Demo script and slides
- Full test suite (22 tests)

### Completed Setup Documentation

- Project setup files (`README.md`, `.gitignore`, and `.env.example`) have been populated.
- Station metadata and data-source documentation are recorded in [`DATA_SOURCES.md`](./DATA_SOURCES.md).

### Not yet completed

- Actual Hugging Face Space upload (requires account/token)
- Live OpenAQ end-to-end test with a real API key
- Scheduled retraining and drift monitoring

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
The quality-control summary is written to `reports/gap_report.csv` when
`python -m src.clean` is rerun. That generated file is gitignored, so the cleaned
data in `data/processed/clean_hourly.csv` is the committed source of truth.

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
| Train | 16,029 | 2025-06-30 05:00 | 2026-06-06 12:00 |
| Validation | 3,442 | 2026-06-07 13:00 | 2026-08-18 05:00 |
| Test | 3,492 | 2026-08-19 06:00 | 2026-10-30 23:00 |

The validation and test sets contain equal numbers of Karachi and Lahore rows
(Validation: 1,721 each; Test: 1,746 each). Train has 8,120 Lahore and 7,909
Karachi rows.

## Verification completed

The full suite passed:

```text
22 passed
```

`tests/test_pipeline_logic.py` contributes 3 tests that verify:

- The 24-hour target and PM2.5 lag values are aligned correctly.
- Feature names are unique and present in the generated data.
- Train, validation, and test splits do not overlap through the 24-hour
  target horizon.

`tests/test_aqi.py` contributes 19 tests covering every AQI breakpoint (both
sides), gap values, invalid input, monotonicity, colours, and advice structure.

The test run produces pandas deprecation warnings for `t.dt.weekday`. These
warnings do not fail the tests, but the expression should eventually be updated
to `t.dt.day_of_week`.

## Reproducible pipeline commands

After setting `OPENAQ_API_KEY` in a local `.env` file, the data stages can be
run in order:

```text
python -m src.collect
python -m src.clean
python -m src.features
python -m src.split
python -m src.baseline
python -m src.train
python -m src.train_final
python -m src.evaluate
python -m src.error_analysis
pytest tests/
```

The EDA notebook can then be opened and rerun from
[`notebooks/02_eda.ipynb`](./notebooks/02_eda.ipynb).

Product commands (no OpenAQ key needed for the offline path):

```text
python -m src.live_data --refresh-fallback --offline
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload --port 8000
```

## Stage D results (current 47-feature run)

Validation comparison:

| Model | MAE | RMSE | R² | AQI accuracy |
|---|---:|---:|---:|---:|
| Ridge | 10.23 | 15.29 | 0.2645 | 69.2% |
| KNN | 9.08 | 14.18 | 0.3679 | 71.5% |
| Random Forest | **8.43** | **13.67** | **0.4121** | 71.9% |
| LightGBM | 8.49 | 13.78 | 0.4023 | 71.9% |
| Ensemble | 8.45 | 13.70 | 0.4099 | **72.4%** |

The ensemble was selected using the pre-declared 0.2 µg/m³ stability rule.

Final test evaluation (ensemble retrained on train + validation, 19,471 rows):

| Scope | MAE | RMSE | R² | AQI accuracy |
|---|---:|---:|---:|---:|
| Overall | **13.31** | **23.65** | **0.6695** | **71.65%** |
| Karachi | 6.20 | 9.46 | 0.7282 | 82.02% |
| Lahore | 20.42 | 32.08 | 0.5591 | 61.28% |

Persistence baseline on the same test split: MAE 14.54, RMSE 26.98, R² 0.5700,
AQI accuracy 69.99%. The ensemble improves MAE by 8.5%, RMSE by 12.3%, R² by
0.0995, and AQI accuracy by 1.66 percentage points.

Artifacts: `models/final_pipeline.joblib` and `models/model_metadata.json`.

## Stage E results (product stage)

Completed:

- Error analysis in `src/error_analysis.py`, with report, CSVs, JSON, and seven
  figures in `reports/`.
- AQI module fixed (breakpoint gap bug) and covered by 19 per-breakpoint tests.
- Live-data inference with OpenAQ/Open-Meteo/CAMS and a labelled offline fallback
  in `src/live_data.py`.
- Streamlit app in `app/streamlit_app.py` (verified headless, health 200).
- FastAPI service in `app/api.py` (`/health`, `/predict`, `/aqi`, `/metadata`).
- Hugging Face Space bundle in `deploy/huggingface/` (actual push requires a
  Hugging Face account/token, so it was not uploaded from this environment).
- Final report, limitations/ethics update, README polish, demo script, and slides.
- Full test suite: 22 passed.

Key error-analysis finding: the ensemble is strong on typical hours but
underpredicts top-5% spikes (90.3% of spike hours) and overpredicts rapid clearing
days. Lahore October 2026 MAE is 30.79; the worst day is 29 October 2026 (MAE
108.82). Most AQI category errors are one band away.

Rejected post-processing, documented in `Limitations.md`:

- Clearing blend: MAE 13.51, AQI 71.33% (worse than raw ensemble).
- Conservative AQI buffers: MAE 13.35, AQI 69.22% (worse AQI accuracy).

## Stage F: Manual Testing and Documentation Completion (November 12, 2026)

### Completed

- **Application Deployment:** Both Streamlit frontend and FastAPI backend successfully deployed and tested in local environment.
  - FastAPI backend: Running on port 8001 with all endpoints operational (`/health`, `/predict`, `/aqi`, `/metadata`, `/docs`)
  - Streamlit frontend: Running on port 8502 with full UI functionality
  - Backend-frontend integration verified via environment variable `API_BASE_URL`

- **Manual Testing Performed:**
  - ✅ **Live Predictions Verified** (November 12, 2026, ~21:00 PKT):
    - **Lahore:** Observed 262.93 µg/m³ → Predicted 273.35 µg/m³ (Hazardous, AQI 336)
    - **Karachi:** Observed 47.83 µg/m³ → Predicted 55.29 µg/m³ (Unhealthy for Sensitive Groups, AQI 150)
    - Data staleness: 2.2 hours (acceptable for live mode)
  - ✅ **Offline Mode:** Successfully tested with saved snapshot (October 30, 2026 data)
  - ✅ **City Switching:** Predictions update correctly when switching between Lahore and Karachi
  - ✅ **Backend Modes:** Both "Local pipeline" and "FastAPI service" modes functional
  - ✅ **Charts and Visualizations:** 
    - Recent trend line chart displays historical PM2.5 correctly
    - Now vs Tomorrow bar chart renders comparison
    - Model accuracy panel shows all three error analysis figures
  - ✅ **Manual AQI Calculator:** Slider-based PM2.5 to AQI conversion working across all six EPA categories
  - ✅ **Health Advice:** Contextual recommendations display appropriately for each AQI severity level
  - ✅ **Data Freshness Indicators:** Staleness warnings appear correctly for offline/stale data
  - ✅ **API Endpoints:** All REST endpoints tested via Swagger UI and command-line requests
    - Health check returns status "ok" with model metrics
    - Predict endpoint handles valid/invalid cities, live/offline modes
    - Manual AQI endpoint validates input ranges (0-2000 µg/m³)
    - Metadata endpoint returns test metrics and feature count

- **Automated Testing:**
  - ✅ **31 out of 31 tests passing** (100% pass rate)
  - Test coverage:
    - 9 API endpoint tests (`test_api.py`): Health check, predictions, validation, error handling
    - 19 AQI calculation tests (`test_aqi.py`): All breakpoint boundaries, edge cases, monotonicity
    - 3 pipeline integrity tests (`test_pipeline_logic.py`): Target alignment, feature uniqueness, split non-overlap

- **Testing Documentation Created:**
  - **`reports/manual_testing_guide.md`** (Comprehensive, 400+ lines):
    - Step-by-step testing scenarios for all features
    - 50+ item testing checklist (API tests, UI tests, prediction tests, edge cases)
    - Expected results and sample outputs for each scenario
    - Bug reporting template with severity classification
    - Detailed troubleshooting guide for common issues
    - Performance expectations and accuracy benchmarks
  - **`QUICK_TEST_CHECKLIST.md`** (Quick reference, 15-minute testing track):
    - 10 priority tests covering core functionality
    - Visual verification guide for AQI color coding
    - Common issues quick-fix table
  - **`RUNNING_APPLICATION_STATUS.md`** (Service status document):
    - Real-time service status with URLs and process IDs
    - Verification test results summary
    - Quick start guide for immediate testing
    - Service management commands

- **Model Performance Confirmed:**
  - Voting Ensemble: Ridge (20%) + Random Forest (30%) + LightGBM (50%)
  - 47 engineered features (PM2.5 lags, rolling stats, momentum, weather, CAMS aerosol, calendar, city)
  - Training: 19,471 rows (train + validation combined for final model)
  - Held-out test metrics:
    - MAE: 13.31 µg/m³ (8.5% improvement over persistence baseline)
    - RMSE: 23.65 µg/m³ (12.3% improvement)
    - R²: 0.6695
    - AQI Category Accuracy: 71.65%
  - Per-city performance:
    - Karachi: MAE 6.20 µg/m³, AQI accuracy 82.0%
    - Lahore: MAE 20.42 µg/m³, AQI accuracy 61.3%

### Features Validated Through Manual Testing

| Feature Category | Status | Notes |
|-----------------|--------|-------|
| Live API Data Fetching | ✅ Working | OpenAQ + Open-Meteo + CAMS integration functional |
| Offline Snapshot Fallback | ✅ Working | Graceful degradation when APIs unavailable |
| City-Specific Predictions | ✅ Working | Lahore and Karachi predictions differ appropriately |
| AQI Category Color Coding | ✅ Working | All 6 EPA categories display correct colors (Green→Maroon) |
| Health Advice Generation | ✅ Working | Contextual recommendations match severity levels |
| Trend Visualization | ✅ Working | Historical PM2.5 line chart with 24h rolling mean |
| Comparison Charts | ✅ Working | Now vs Tomorrow bar chart |
| Model Accuracy Display | ✅ Working | Error scatter, monthly MAE, confusion matrix figures |
| Manual AQI Simulator | ✅ Working | Real-time slider-based PM2.5→AQI conversion |
| Cache Refresh | ✅ Working | "Refresh inputs" button clears cached predictions |
| Input Validation | ✅ Working | API rejects invalid cities, out-of-range PM2.5 values |
| Error Handling | ✅ Working | User-friendly error messages for backend failures |
| Backend Switching | ✅ Working | Local pipeline and FastAPI service modes both functional |

### Testing Environment

- **Date:** November 12, 2026
- **Services:** FastAPI (port 8001), Streamlit (port 8502)
- **Data Currency:** Live data with 2.2-hour staleness
- **Browser Compatibility:** Verified in modern browsers
- **Platform:** Windows environment with Python 3.14.7

### Limitations Documented

Updated `Limitations.md` with deployment status clarifications:
- Historical chart data ends at 2026-10-30 23:00 PKT (latest training data)
- Offline snapshot can be stale (312+ hours in offline mode)
- Community sensor data quality constraints
- Lahore spike/clearing prediction challenges
- AQI advice is general awareness, not medical/regulatory guidance
- Hugging Face deployment bundle prepared but **actual Space upload pending** (requires HF account/token)

### Key Findings from Manual Testing

1. **Live Mode Performance:** Application successfully fetches and processes real-time data with acceptable latency (2-10 seconds for predictions)
2. **Prediction Quality:** Live predictions align with expected AQI categories based on current conditions
3. **User Experience:** Interface is intuitive with clear visual hierarchy and responsive controls
4. **Error Resilience:** Application handles backend failures gracefully without crashing
5. **Data Transparency:** Staleness indicators and source labels provide clear data provenance
6. **Cross-City Comparison:** Lahore consistently shows higher PM2.5 than Karachi, matching known patterns

### What Works Well

- Both deployment modes (local pipeline and API service) are stable
- Live data integration with automatic fallback to offline snapshot
- Comprehensive error handling and user-friendly messaging
- Visual clarity of AQI categories through color coding
- Real-time interactive features (sliders, dropdowns, expanders)
- API documentation through Swagger UI is complete and functional

### Known Issues from Manual Testing

- **Port Conflict:** Initial port 8000 was unavailable; resolved by using port 8001
- **Module Import:** Streamlit required `PYTHONPATH` environment variable set to project root
- **Historical Charts:** Limited to training period ending October 30, 2026 (by design)
- **Stale Data Warning:** Triggers appropriately but could be more prominent for >24h staleness

### Testing Artifacts Created

All testing documentation is preserved in the repository:
- Manual testing guide with 50+ test scenarios
- Quick checklist for rapid verification
- Application status document with live URLs
- Screenshots and expected outputs documented

## Remaining work

1. Create and push the Hugging Face Space, then add `OPENAQ_API_KEY` as a secret (bundle prepared in `deploy/huggingface/`, actual upload requires HF account).
2. Run one live end-to-end collection/inference test with a real API key (partially completed with manual testing of live predictions).
3. Add scheduled retraining and input-drift monitoring.
4. Add more Lahore winter data and external crop-burning/fire indicators.
5. Export `reports/slides.md` to PDF/PPTX if the course requires that format.
