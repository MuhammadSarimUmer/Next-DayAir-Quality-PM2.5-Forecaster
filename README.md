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

We build **19 features** (final list may be adjusted during EDA):

| Group | Features | Count |
|---|---|---|
| **Past PM2.5** | Lagged values (1h, 3h, 6h, 12h, 24h ago) | 5 |
| **Rolling averages** | Mean PM2.5 over the last 6h, 24h, 72h | 3 |
| **Weather** | Temperature, humidity, pressure, wind speed, precipitation, cloud cover | 6 |
| **Calendar** | Hour, month, weekday, winter flag | 4 |
| **Location** | City (Karachi / Lahore) | 1 |
| **Total** | | **19** |

**Reasoning:**

- **Lags** capture momentum: pollution tomorrow strongly depends on pollution now.
- **Rolling averages** smooth out sensor noise and capture multi-day smog episodes.
- **Weather** explains *why* pollution rises or falls (low wind and high humidity trap pollutants).
- **Calendar fields** capture daily traffic cycles and seasonal patterns. The winter flag isolates smog season.
- **City** lets one model serve both cities while learning that their baselines differ.

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

## 7. How we will judge success

We report on the held-out test set:

| Metric | What it tells us |
|---|---|
| **MAE** (µg/m³) | Average size of the error, easy to interpret. |
| **RMSE** (µg/m³) | Like MAE but penalises large misses, which matter during smog spikes. |
| **R²** | Share of variance explained by the model. |

**Success criterion:** the final model must **beat the baseline "PM2.5 tomorrow equals PM2.5 now" on both MAE and RMSE**. If it cannot, a machine learning model adds no value over simply repeating today's reading.

We will also report per-city results and, as an extra, how often the predicted AQI *category* matches the true one, since that is what users actually see.

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

# 3. Collect data
python src/collect.py

# 4. Train
python src/train.py

# 5. Run the apps
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload
```

*Commands will be finalised as each script is written.*

## 11. Project timeline

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Proposal | Done |
| 2 | Data collection + `DATA_SOURCES.md` | Planned |
| 3 | Cleaning + EDA | Planned |
| 4 | Feature engineering + split | Planned |
| 5 | Pipeline + model comparison | Planned |
| 6 | Evaluation + model saving | Planned |
| 7 | Streamlit, FastAPI, Hugging Face deployment | Planned |
| 8 | Final report + demo | Planned |

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
