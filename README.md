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
6. [How we will build it](#6-how-we-will-build-it)
7. [How we will judge success](#7-how-we-will-judge-success)
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

The predicted number is mapped to a health category using the US EPA PM2.5 breakpoints (24-hour average, µg/m³). *Note: the team should confirm which breakpoint table it adopts (EPA revised the PM2.5 breakpoints in 2024) and keep it consistent in `src/aqi.py`.*

| PM2.5 (µg/m³) | Category | Example advice |
|---|---|---|
| 0.0 – 9.0 | Good | Air is clean. Enjoy outdoor activities. |
| 9.1 – 35.4 | Moderate | Unusually sensitive people should reduce prolonged exertion. |
| 35.5 – 55.4 | Unhealthy for Sensitive Groups | Children, elderly and people with asthma should limit time outside. |
| 55.5 – 125.4 | Unhealthy | Avoid outdoor exercise. Wear an N95 mask outside. |
| 125.5 – 225.4 | Very Unhealthy | Stay indoors where possible. Keep windows closed. |
| 225.5+ | Hazardous | Avoid going outside. Use an air purifier if available. |

## 4. Data

We combine two free, public sources. Neither contains personal data.

| Source | What we take | Licence |
|---|---|---|
| **[OpenAQ](https://openaq.org)** | Hourly PM2.5 readings from stations in Karachi and Lahore. We prefer the US Embassy/Consulate monitors where available, plus other stations with long, continuous histories. | Open data (per-provider terms recorded in `DATA_SOURCES.md`) |
| **[Open-Meteo Historical Weather API](https://open-meteo.com)** | Hourly temperature, humidity, pressure, wind speed, precipitation, cloud cover | CC BY 4.0 |

**Volume:** 1 to 2 years of hourly data per city, depending on station availability. This gives well over the 5,000-row minimum for the course.

**Why these sources?** They are free, reproducible, hourly, and cover the same time span. Reference-grade monitors (such as the US Embassy ones) are more trustworthy than low-cost sensors, which is why we prefer them.

**Documentation:** The exact stations chosen, source, collection method, licence and collection date are recorded in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## 5. Features

The model uses **35 engineered features** constructed strictly without forward leakage:

| Group | Features | Count | Description |
|---|---|---:|---|
| **Past PM2.5** | `pm25_now`, `pm25_lag_1h`, `pm25_lag_3h`, `pm25_lag_6h`, `pm25_lag_12h`, `pm25_lag_24h` | 6 | Captures immediate autocorrelation and memory |
| **Rolling statistics** | `pm25_roll_mean_6h`, `pm25_roll_mean_24h`, `pm25_roll_mean_72h`, `pm25_diff_3h`, `pm25_roll_std_24h`, `pm25_roll_max_24h` | 6 | Smooths noise and detects multi-day smog accumulation & variance |
| **Momentum & Trend** | `pm25_change_6h`, `pm25_change_24h`, `pm25_rate_of_change`, `pm25_7d_lag`, `pm25_roll_min_24h` | 5 | 6h/24h trajectory, velocity (µg/m³/h), weekly anchor, and daily floor |
| **Weather (direct)** | `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `precipitation`, `cloud_cover` | 6 | Atmospheric conditions governing dispersion |
| **Wind vector** | `wind_dir_sin`, `wind_dir_cos` | 2 | Circular decomposition avoiding 359°–1° boundary discontinuity |
| **Weather (derived)** | `pressure_change_24h`, `wind_speed_roll_mean_24h`, `precip_sum_24h` | 3 | Barometric pressure trend, atmospheric stagnation, 24h rain washout |
| **Calendar** | `hour`, `month`, `weekday`, `is_winter`, `hour_sin`, `hour_cos` | 6 | Diurnal traffic cycles, weekly cycles, winter smog flag, cyclic hour |
| **Location** | `city` | 1 | One-hot encoded (Karachi / Lahore) |
| **Total** | | **35** | |

---

## 6. How we will build it

### Step 1: Data collection
Pull hourly PM2.5 from OpenAQ and hourly weather from Open-Meteo for both cities. Align them on a common hourly timestamp. Save raw files unchanged in `data/raw/`.

### Step 2: Cleaning and EDA
Handle missing hours, sensor outliers (impossible or stuck values) and timezone alignment (everything in Pakistan Standard Time). Explore trends, seasonality, and correlations in `notebooks/`.

### Step 3: Feature engineering
Build the 19 features. The target is created by shifting PM2.5 **backwards by 24 hours**, so each row's inputs are known at time *t* and its label is the value at *t + 24h*.

### Step 4: Chronological split
Split **70% train / 15% validation / 15% test, in time order**.

> **Why not a random split?** Time series have autocorrelation. A random split would leak future information into training (a row at 3 pm in the test set could have its neighbours at 2 pm and 4 pm in training), producing misleadingly good scores. A chronological split mimics real use: train on the past, predict the future.

### Step 5: Preprocessing Pipeline
A scikit-learn `Pipeline` / `ColumnTransformer` handling:

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

All models tuned with 5-fold `TimeSeriesSplit` cross-validation:

| Model | Val MAE (µg/m³) | Val RMSE (µg/m³) | Val R² | Val AQI Category Acc |
|---|---:|---:|---:|---:|
| Ridge Regression (α=10.0) | 9.93 | 14.83 | 0.3040 | 70.7% |
| KNN Regressor (k=15, uniform) | 9.47 | 14.64 | 0.3221 | 68.3% |
| Random Forest (d=20, n=100) | 8.83 | 14.01 | 0.3793 | 69.5% |
| **LightGBM (Winner)** | **8.77** | **13.91** | **0.3876** | **71.0%** |

### Final Evaluation on Held-Out Test Set (Run Once)

The winning model (LightGBM) was retrained on the combined Train + Validation set (20,002 samples) and evaluated on the untouched held-out test set (3,678 samples):

| Metric | Baseline (Persistence) | LightGBM Final | Improvement |
|---|---:|---:|---:|
| **MAE** (µg/m³) | 14.64 | **13.50** | **+7.8%** |
| **RMSE** (µg/m³) | 27.17 | **23.93** | **+11.9%** |
| **R²** | 0.5765 | **0.6621** | **+0.0856** |
| **AQI Category Accuracy** | 69.8% | **71.3%** (71.34%) | **+2.2%** |

#### Per-City Breakdown on Test Set

| City | Test MAE (µg/m³) | Test RMSE (µg/m³) | Test R² | Test AQI Accuracy |
|---|---:|---:|---:|---:|
| **Karachi** | 6.25 | 9.56 | 0.7233 | **82.60%** |
| **Lahore** | 20.75 | 32.47 | 0.5558 | **60.09%** |

*Note: The model clears the baseline on all metrics. For conservative health warnings, applying boundary buffer zones yields 68.11% accuracy while erring safely on the side of caution.*

## 8. Deployment

The saved pipeline is served in three ways:

| Platform | Purpose |
|---|---|
| **Streamlit** | Interactive web app: choose a city, see the forecast, AQI category and advice. |
| **Hugging Face Spaces** | Free public hosting so anyone can use the app without installing anything. |
| **FastAPI (local)** | A REST endpoint (`/predict`) showing how the model could be used by other applications. |

## 9. Repository structure

```
.
├── README.md
├── DATA_SOURCES.md          # stations, source, method, licence, collection date
├── requirements.txt
├── data/
│   ├── raw/                 # untouched downloads
│   └── processed/           # cleaned + feature-engineered data
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_features.ipynb
│   └── 04_modelling.ipynb
├── src/
│   ├── collect.py           # OpenAQ + Open-Meteo download scripts
│   ├── features.py          # feature engineering
│   ├── pipeline.py          # sklearn Pipeline definition
│   ├── train.py             # training + tuning + evaluation
│   └── aqi.py               # PM2.5 -> AQI category + advice
├── models/
│   └── pm25_pipeline.joblib
├── app/
│   ├── streamlit_app.py
│   └── api.py               # FastAPI app
└── reports/
    └── figures/
```

*(This is the planned layout and may change as the project grows.)*

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
python -m src.train
python -m src.evaluate
pytest tests/

# 4. Run the apps (Deployment)
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload
```

## 11. Project timeline

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Proposal | Done |
| 2 | Data collection + `DATA_SOURCES.md` | Done |
| 3 | Cleaning + EDA | Done |
| 4 | Feature engineering (35 features) + chronological split | Done |
| 5 | Sklearn Pipeline + model comparison (Ridge, KNN, RF, LightGBM) | Done |
| 6 | Evaluation + model saving (`final_pipeline.joblib`) | Done |
| 7 | Streamlit, FastAPI, Hugging Face deployment | In Progress |
| 8 | Final report + demo | In Progress |

## 12. Limitations and ethics

- **Forecast, not a guarantee.** Sudden events (crop burning, industrial incidents, dust storms) are hard to predict from past data alone.
- **Station coverage.** Readings from one or a few stations do not represent every neighbourhood in a large city.
- **Data gaps.** Public monitors have outages; imputation reduces but does not remove this uncertainty.
- **Limited history.** With 1 to 2 years of data, the model sees only a couple of winter seasons.
- **Not medical advice.** The health tips are general guidance based on AQI categories.
- **Privacy.** We use only public environmental data. No personal data is collected or stored.

## 13. Licence and data attribution

- Code: *(choose a licence, e.g. MIT, and add a `LICENSE` file)*
- Weather data: [Open-Meteo](https://open-meteo.com), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Air quality data: [OpenAQ](https://openaq.org) and the original data providers (see `DATA_SOURCES.md`)

---

*Built for CT-378 Machine Learning by Team Khubsoorat Shklen.*
