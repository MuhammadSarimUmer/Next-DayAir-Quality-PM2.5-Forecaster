# Next-Day Air Quality (PM2.5) Forecaster for Karachi and Lahore

> CT-378 Machine Learning, Semester Project
> **Team Khubsoorat Shklen**

| Member | Roll No. |
|---|---|
| Muhammad Sarim Umer | CT-24235 |
| Muhammad Feras Malik | CT-24247 |
| Muhammad Ahmed Khan | CT-24245 |

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [Why are we building it?](#2-why-are-we-building-it)
3. [What exactly are we predicting?](#3-what-exactly-are-we-predicting)
4. [Data](#4-data)
5. [Features](#5-features)
6. [How we built it](#6-how-we-built-it)
7. [How we judge success](#7-how-we-judge-success)
8. [Deployment](#8-deployment)
9. [Repository structure](#9-repository-structure)
10. [Getting started](#10-getting-started)
11. [Project timeline](#11-project-timeline)
12. [Limitations and ethics](#12-limitations-and-ethics)
13. [Licence and data attribution](#13-licence-and-data-attribution)

---

## 1. What is this project?

A machine learning system that predicts the **PM2.5 concentration 24 hours into the future** for **Karachi** and **Lahore**, using recent pollution readings and weather data. The prediction is then translated into an **AQI health category** (Good, Moderate, Unhealthy, and so on) with a one-line piece of advice, so that a non-technical person can act on it.

This is a **supervised regression** problem on **time-series tabular data**.

## 2. Why are we building it?

Air pollution in Karachi and Lahore regularly reaches unhealthy levels, especially during the winter smog season. Existing apps and websites mostly show the **current** reading. That is useful, but it does not help someone decide ahead of time:

- Should I exercise outdoors tomorrow morning?
- Should my children go to school tomorrow?
- Should I carry a mask?

A **forecast** answers these questions in time to act. PM2.5 (particles smaller than 2.5 micrometres) is the pollutant most strongly linked to respiratory and cardiovascular harm, which is why it is our target.

**Why machine learning rather than a simple rule?** PM2.5 depends on many interacting factors: how polluted the air already is, wind speed (which disperses smog), humidity and temperature (which trap or form particles), the time of day (traffic cycles), and the season (winter inversions). A model can learn these interactions from data. We prove this value by comparing against a naive baseline (see [Section 7](#7-how-we-will-judge-success)).

## 3. What exactly are we predicting?

| Item | Definition |
|---|---|
| **Task type** | Regression |
| **Target variable** | PM2.5 concentration (µg/m³) at time **t + 24 hours** |
| **Inputs** | Past PM2.5 values and averages, weather, calendar fields, city |
| **Output shown to user** | Predicted PM2.5 value, AQI category, one-line health advice |

Weather variables are **inputs only**. We are not predicting the weather; the target is pollution.

### AQI category mapping

The predicted number is mapped to a health category using the **2024 US EPA PM2.5 breakpoints** (24-hour average, µg/m³). This table is used consistently in `src/aqi.py`, `src/metrics.py`, and the final report.

| PM2.5 (µg/m³) | Category | Example advice |
|---|---|---|
| 0.0 – 9.0 | Good | Air is clean. Enjoy outdoor activities. |
| 9.1 – 35.4 | Moderate | Unusually sensitive people should reduce prolonged exertion. |
| 35.5 – 55.4 | Unhealthy for Sensitive Groups | Children, elderly and people with asthma should limit time outside. |
| 55.5 – 125.4 | Unhealthy | Avoid outdoor exercise. Wear an N95 mask outside. |
| 125.5 – 225.4 | Very Unhealthy | Stay indoors where possible. Keep windows closed. |
| 225.5+ | Hazardous | Avoid going outside. Use an air purifier if available. |

## 4. Data

We combine three free, public sources. None contains personal data.

| Source | What we take | Licence |
|---|---|---|
| **[OpenAQ](https://openaq.org)** | Hourly PM2.5 readings from community stations in Karachi and Lahore. We keep stations with long, continuous histories and reject low-coverage or stuck sensors during cleaning. | Open data (per-provider terms recorded in `DATA_SOURCES.md`) |
| **[Open-Meteo Historical Weather API](https://open-meteo.com)** | Hourly temperature, humidity, pressure, wind speed, precipitation, cloud cover | CC BY 4.0 |
| **Open-Meteo CAMS** | Regional PM2.5, dust, and aerosol optical depth | CC BY 4.0 |

**Volume:** 11,899 cleaned city-hours per city (23,798 total), from 2025-06-30 to
2026-10-30 after feature construction and splitting. Karachi mean PM2.5 is
44.6 µg/m³ and Lahore mean is 103.9 µg/m³.

**Why these sources?** They are free, reproducible, hourly, and cover the same time
span. OpenAQ ground sensors provide local measurements, weather explains dispersion,
and CAMS provides an independent regional smoke/dust signal. These are community
sensors rather than reference-grade monitors, which is a documented limitation.

**Documentation:** The exact stations chosen, source, collection method, licence and collection date are recorded in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## 5. Features

The model uses **47 engineered features** constructed strictly without forward leakage:

| Group | Features | Count | Description |
|---|---|---:|---|
| **Current + past PM2.5** | `pm25_now`, lags 1/3/6/12/24 h, rolling means 6/24/72 h | 9 | Autocorrelation, memory, multi-day accumulation |
| **Rolling extras** | `pm25_diff_3h`, `pm25_roll_std_24h`, `pm25_roll_max_24h` | 3 | Short trend, volatility, recent peak |
| **Momentum & trend** | `pm25_change_6h`, `pm25_change_24h`, `pm25_rate_of_change`, `pm25_7d_lag`, `pm25_roll_min_24h` | 5 | Trajectory, velocity, weekly anchor, daily floor |
| **Domain physics** | `pm25_acceleration`, `stagnation_index`, `smog_season_intensity`, `lahore_smog_risk` | 4 | Surges, low-wind/high-humidity trapping, crop-burning season, Lahore winter risk |
| **Regime shifts** | `pm25_lag_excess_24h`, `pm25_lag_excess_72h`, `pm25_clearing_index`, `pm25_cams_divergence`, `cams_pm25_change_24h` | 5 | Recovery/clearing and ground-vs-regional disagreement |
| **CAMS aerosol** | `cams_pm2_5`, `cams_dust`, `cams_aod` | 3 | Independent regional smoke/dust signal |
| **Weather (direct)** | `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `precipitation`, `cloud_cover` | 6 | Conditions governing dispersion and particle formation |
| **Wind vector** | `wind_dir_sin`, `wind_dir_cos` | 2 | Circular decomposition avoiding the 359°–1° discontinuity |
| **Weather (derived)** | `pressure_change_24h`, `wind_speed_roll_mean_24h`, `precip_sum_24h` | 3 | Pressure trend, stagnation, rain washout |
| **Calendar** | `hour`, `month`, `weekday`, `is_winter`, `hour_sin`, `hour_cos` | 6 | Diurnal, weekly, winter-smog, and cyclic-hour effects |
| **Location** | `city` | 1 | One-hot encoded (Karachi / Lahore) |
| **Total** | | **47** | |

---

## 6. How we built it

### Step 1: Data collection
Pulled hourly PM2.5 from OpenAQ, hourly weather from Open-Meteo, and CAMS aerosol data for both cities. Aligned them on a common hourly timestamp. Raw files are preserved unchanged in `data/raw/`.

### Step 2: Cleaning and EDA
Handled missing hours, sensor outliers (impossible or stuck values) and timezone alignment (everything in Pakistan Standard Time). Explored trends, seasonality, and correlations in `notebooks/`.

### Step 3: Feature engineering
Built the 47 features. The target is created by shifting PM2.5 **backwards by 24 hours**, so each row's inputs are known at time *t* and its label is the value at *t + 24h*.

### Step 4: Chronological split
Split **70% train / 15% validation / 15% test, in time order**, with a 24-hour purge gap between splits.

> **Why not a random split?** Time series have autocorrelation. A random split would leak future information into training (a row at 3 pm in the test set could have its neighbours at 2 pm and 4 pm in training), producing misleadingly good scores. A chronological split mimics real use: train on the past, predict the future.

### Step 5: Preprocessing Pipeline
We used a scikit-learn `Pipeline` / `ColumnTransformer` handling:

- **Imputation** of remaining missing values
- **Scaling** of numeric features (needed for Ridge and KNN)
- **Encoding** of categorical features (city)

The pipeline is **fitted on training data only** and then applied to validation and test data. This prevents data leakage.

### Step 6: Model comparison

| Model | Why we include it |
|---|---|
| **Baseline** ("tomorrow = now") | The bar any useful model must clear. |
| **Ridge Regression** | Simple, interpretable linear model with regularisation. Tells us how far linear relationships go. |
| **KNN Regressor** | Instance-based, no assumptions about the data shape. Shows whether "similar past days" are informative. |
| **Random Forest** | Captures non-linear interactions and is robust to noise and outliers. |
| **LightGBM** | Gradient boosting, typically the strongest on tabular data, and fast to train. |

Hyperparameters are tuned with **5-fold `TimeSeriesSplit`** cross-validation. Each fold trains only on data that comes *before* its validation window, so the model never sees the future.

### Step 7: Final evaluation and saving
Pick the best model using validation results, evaluate **once** on the untouched test set, and save the full pipeline (preprocessing + model) with `joblib`.

## 7. How we judge success & Experimental Results

### Model Comparison (Validation Set)

All models tuned with 5-fold `TimeSeriesSplit` cross-validation. Current run (47
features, 16,029 training rows, 3,442 validation rows):

| Model | Val MAE (µg/m³) | Val RMSE (µg/m³) | Val R² | Val AQI Category Acc |
|---|---:|---:|---:|---:|
| Ridge Regression (α=10.0) | 10.23 | 15.29 | 0.2645 | 69.2% |
| KNN Regressor (k=15, uniform) | 9.08 | 14.18 | 0.3679 | 71.5% |
| Random Forest (d=12, n=120) | **8.43** | **13.67** | **0.4121** | 71.9% |
| LightGBM (lr=0.03, n=300, leaves=31) | 8.49 | 13.78 | 0.4023 | 71.9% |
| **Blended Ensemble (selected)** | 8.45 | 13.70 | 0.4099 | **72.4%** |

The ensemble is selected by the documented rule: use it when it is within
0.2 µg/m³ of the best validation MAE, because it is more stable across cities.

### Final Evaluation on Held-Out Test Set (Run Once)

The winning model (weighted ensemble: Ridge 0.2, Random Forest 0.3, LightGBM 0.5)
was retrained on the combined train + validation set (19,471 samples) and evaluated
on the untouched held-out test set (3,492 samples, 1,746 per city):

| Metric | Baseline (Persistence) | Ensemble Final | Improvement |
|---|---:|---:|---:|
| **MAE** (µg/m³) | 14.54 | **13.31** | **+8.5%** |
| **RMSE** (µg/m³) | 26.98 | **23.65** | **+12.3%** |
| **R²** | 0.5700 | **0.6695** | **+0.0995** |
| **AQI Category Accuracy** | 69.99% | **71.65%** | **+1.66 pp** |

#### Per-City Breakdown on Test Set

| City | Test MAE (µg/m³) | Test RMSE (µg/m³) | Test R² | Test AQI Accuracy | Baseline AQI |
|---|---:|---:|---:|---:|---:|
| **Karachi** | 6.20 | 9.46 | 0.7282 | **82.0%** | 82.1% |
| **Lahore** | 20.42 | 32.08 | 0.5591 | **61.3%** | 57.9% |

#### Error Analysis and Rejected Variants

Full results are in [`reports/error_analysis.md`](reports/error_analysis.md). The
model under-predicts the top 5% of pollution spikes (MAE 41.43) and over-predicts
rapid clearing events (MAE 91.95). October 2026 Lahore is the hardest period
(MAE 30.79). Two optional post-processing heuristics were tested and **rejected**
because they reduced held-out performance: the clearing blend produced MAE 13.51 /
AQI 71.3%, and conservative AQI buffers produced MAE 13.35 / AQI 69.2%. Raw
ensemble output is used by default.

## 8. Deployment

The saved pipeline is served in three ways:

| Platform | Purpose | How to run |
|---|---|---|
| **Streamlit** | Interactive web app: city forecast, AQI category and advice, recent-PM2.5 trend chart, now-vs-tomorrow comparison, and model error figures. Runs on the local pipeline by default or against the FastAPI service. | `streamlit run app/streamlit_app.py` |
| **FastAPI** | REST endpoints: `/health`, `/predict`, `/aqi`, `/metadata`, Swagger docs at `/docs`. | `uvicorn app.api:app --reload --port 8000` |
| **Hugging Face Spaces** | Free public hosting. A Streamlit-Space bundle with the model, history-chart data, and pre-rendered error figures is prepared in `deploy/huggingface/`. | `python deploy/huggingface/prepare_space.py ../pm25-space`, then push the folder to a Space |

The apps load `models/final_pipeline.joblib`, fetch fresh inputs with
`src/live_data.py`, and fall back to `data/processed/live_fallback.csv` when the
network or `OPENAQ_API_KEY` is unavailable. The fallback is always labelled as an
offline snapshot in the UI/API.

**Testing the frontend:** see [`TESTING_FRONTEND.md`](TESTING_FRONTEND.md) for a
step-by-step checklist covering local-pipeline mode and the FastAPI-connected
mode (`$env:API_BASE_URL = "http://127.0.0.1:8000"`).

**Hugging Face note:** creating the Space and pushing require a Hugging Face
account/token. The repository includes everything needed (`deploy/huggingface/README.md`
with the Space YAML header, `requirements.txt`, and the bundler script), but the
actual upload cannot be performed from this environment. The bundle includes
`data/processed/features.csv` for the trend chart and `reports/figures/*.png` for
the accuracy panel, so both charts work on the Space. Rebuilding cleans stale files
from a previous bundle while preserving its Git metadata (including `.git`); a
non-empty folder that is not a previous Space bundle is refused rather than deleted.

## 9. Repository structure

```
.
├── README.md
├── ahmed.md                  # implementation log + rejected alternatives
├── LICENSE                   # MIT
├── Limitations.md            # detailed limitation log
├── DATA_SOURCES.md           # stations, source, method, licence, collection date
├── projectlog.md             # stage-by-stage build log
├── TESTING_FRONTEND.md       # frontend/backend test checklist
├── requirements.txt
├── data/
│   ├── raw/                  # untouched downloads
│   └── processed/            # cleaned, features, splits, live fallback
├── notebooks/
│   └── 02_eda.ipynb
├── src/
│   ├── collect.py            # OpenAQ + Open-Meteo + CAMS downloads
│   ├── clean.py              # QC, alignment, gap handling
│   ├── features.py           # 47 leakage-safe features + target
│   ├── split.py              # chronological split + purge gap
│   ├── pipeline.py           # sklearn preprocessing pipeline
│   ├── baseline.py           # persistence baseline
│   ├── train.py              # candidate tuning
│   ├── train_final.py        # deployable ensemble + metadata
│   ├── evaluate.py           # held-out evaluation
│   ├── metrics.py            # MAE/RMSE/R2/AQI accuracy
│   ├── error_analysis.py     # monthly, regime, spike, confusion analysis
│   ├── postprocess.py        # experimental clearing blend
│   ├── aqi.py                # PM2.5 -> AQI category + advice
│   ├── live_data.py          # freshest inputs / offline fallback
│   └── predict.py            # shared inference helper
├── models/
│   ├── final_pipeline.joblib # deployed 47-feature ensemble
│   └── model_metadata.json   # metrics + feature list
├── app/
│   ├── streamlit_app.py
│   └── api.py                # FastAPI app
├── deploy/
│   └── huggingface/          # Space YAML, requirements, bundler
├── reports/
│   ├── final_report.md
│   ├── error_analysis.md
│   ├── demo_script.md
│   ├── slides.md
│   └── figures/
└── tests/
    ├── test_aqi.py             # every EPA breakpoint + edge cases
    ├── test_api.py             # FastAPI contract and validation
    └── test_pipeline_logic.py
```

## 10. Getting started

```bash
# 1. Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Pipeline execution
python -m src.collect
python -m src.clean
python -m src.features
python -m src.split

# 4. Models and evaluation
python -m src.baseline
python -m src.train
python -m src.train_final
python -m src.evaluate
python -m src.error_analysis

# 5. Tests
pytest tests/

# 6. Product
python -m src.live_data --refresh-fallback --offline
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload --port 8000
```

`OPENAQ_API_KEY` is required only for live collection/inference. Copy
`.env.example` to `.env` and set it. The `.env` file is gitignored and must never
be committed or copied into a deployment bundle. Without the key, the product
uses the saved offline snapshot.

## 11. Project timeline

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Proposal | Done |
| 2 | Data collection + `DATA_SOURCES.md` | Done |
| 3 | Cleaning + EDA | Done |
| 4 | Feature engineering (47 features) + chronological split | Done |
| 5 | Sklearn Pipeline + model comparison (Ridge, KNN, RF, LightGBM) | Done |
| 6 | Evaluation + model saving (`final_pipeline.joblib`) | Done |
| 7 | Error analysis + AQI module + live data + tests | Done (28 tests passing) |
| 8 | Streamlit, FastAPI, Hugging Face deployment bundle | Done locally; Space push needs HF account |
| 9 | Final report, limitations/ethics, README, demo script, slides | Done |

## 12. Limitations and ethics

- **Forecast, not a guarantee.** Sudden events (crop burning, industrial incidents, dust storms) are hard to predict from past data alone.
- **Community sensors.** OpenAQ stations are not reference-grade monitors; spatial coverage and measurement uncertainty remain.
- **Smog transitions.** The model under-predicts top-5% spikes (90.3% of spike hours) and over-predicts rapid clearing days. Lahore is much harder than Karachi (MAE 20.42 vs 6.20).
- **Limited history.** About 16 months of data means only one full winter cycle; more Lahore winters are the main next data improvement.
- **AQI thresholds.** We use 2024 US EPA breakpoints because no equivalent Pakistani national table is configured in the project.
- **Static model.** There is no automatic retraining or drift monitor yet; the saved model should be refreshed as new data arrives.
- **Not medical advice.** The health tips are general guidance based on AQI categories.
- **Privacy.** We use only public environmental data. No personal data is collected or stored.

## 13. Licence and data attribution

- Code: MIT (see `LICENSE`)
- Weather data: [Open-Meteo](https://open-meteo.com), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Air quality data: [OpenAQ](https://openaq.org) and the original data providers (see `DATA_SOURCES.md`)

---

*Built for CT-378 Machine Learning by Team Khubsoorat Shklen.*
