# Known Limitations

This document records the known limitations, failure modes, and design trade-offs of the Next-Day Air Quality PM2.5 Forecaster. It is intended for the final report and for anyone continuing development on this project.

---

## 1. Feature Ceiling — Persistence-Dominated Model

### What was observed

After completing the initial 35-feature modelling stage, the top feature by importance was `pm25_now` at **86.0%**. The other 34 features combined accounted for only 14% of the model's decisions. This was confirmed across all four trained models (Ridge, KNN, Random Forest, LightGBM). The current model has since grown to **47 features** (see below), but `pm25_now` remains the dominant signal, so the same persistence ceiling applies.

An explicit improvement attempt was made in November 2026: five new momentum and
trend features were added to `features.py`, followed by domain and regime features.
The current model uses **47 features**:

| Feature | Purpose |
|---------|---------|
| `pm25_change_6h` | 6-hour trend direction |
| `pm25_change_24h` | Day-over-day direction |
| `pm25_rate_of_change` | 3-hour PM2.5 velocity (µg/m³ per hour) |
| `pm25_7d_lag` | Same hour last week (weekly seasonality anchor) |
| `pm25_roll_min_24h` | 24-hour floor (recovery signal) |
| `pm25_acceleration` | 2nd derivative: detects sudden surges into smog episodes |
| `stagnation_index` | Atmospheric trapping index (low wind + high relative humidity) |
| `smog_season_intensity` | Continuous curve for Punjab crop-residue burning season |
| `lahore_smog_risk` | Winter stagnation interaction specialized for Lahore |

After retraining with all 47 features and evaluating the weighted ensemble (Ridge
0.2 + Random Forest 0.3 + LightGBM 0.5), the current held-out test results are:

| Metric | Baseline | Ensemble |
|---|---:|---:|
| MAE | 14.54 µg/m³ | **13.31 µg/m³** (+8.5%) |
| RMSE | 26.98 µg/m³ | **23.65 µg/m³** (+12.3%) |
| R² | 0.5700 | **0.6695** |
| AQI accuracy | 69.99% | **71.65%** |

Lahore improved to **20.42 µg/m³ MAE** and **61.3% AQI accuracy**; Karachi is at
**6.20 µg/m³** and **82.0%**.

### Why it happened

All new features are derived from `pm25` itself and are therefore highly correlated with `pm25_now`. Adding more PM2.5-derived columns does not give the model genuinely new information — it already extracts the trend signal from the existing lags. The model had effectively hit a ceiling where additional PM2.5 history features yielded diminishing returns.

### Implication

The model is, in essence, a sophisticated persistence forecast. It answers "what will PM2.5 be in 24 hours?" primarily by saying "roughly what it is right now, adjusted slightly by weather and calendar." For stable cities like Karachi this works well (AQI accuracy: 82.0%). For high-variance cities like Lahore it falls short (AQI accuracy: 61.3%).

### What would actually help

- **City-specific models**: Training separate models for Karachi and Lahore would let each model specialise. Lahore's model would see proportionally more extreme-value examples and could learn smog-season dynamics without being diluted by Karachi's stable distribution.
- **Kaggle 2021–24 API history** (`USE_KAGGLE_HISTORY` in `config.py`): Hourly OpenWeather-style PM2.5 augments winters before OpenAQ; labels are **not** community sensors—disclose when mixing with ground data from June 2025 onward.
- **More historical data for Lahore**: Even with Kaggle API history, label **provenance changes** at the 2025 OpenAQ cutover; the Dec 2024–Jun 2025 gap has no PM2.5 in either source. (Nov 2025 – Feb 2026). The test set hits October 2026 — the onset of the next smog season — which is partially out-of-distribution. Additional winters of Lahore data would directly address this. **An attempt was made to extend the collection window back to January 2022 via OpenAQ and WAQI/AQICN APIs. However, ground community monitors in Karachi and Lahore only began reporting around May–July 2025, and free API archives do not provide pre-2025 hourly ground sensor data. This is a hard external data constraint, not a pipeline limitation.**
- **External smog indicators**: Open-Meteo CAMS (`cams_pm2_5`, `cams_dust`, `cams_aod`) is now merged via `python -m src.collect --cams` and regime features (`pm25_clearing_index`, `pm25_lag_excess_*`) target Lahore recovery errors. Further gains need fire/crop-burning indices or multi-year Lahore winters.

---

## 2. Lahore–Karachi Performance Gap

### What was observed

| City | Test MAE | Test AQI Accuracy |
|------|----------|-------------------|
| Karachi | 6.20 µg/m³ | 82.0% |
| Lahore | 20.42 µg/m³ | 61.3% |

Lahore performs **3.3× worse** on MAE and **20.7 percentage points worse** on AQI accuracy.

### Why it happened

The two cities have fundamentally different PM2.5 distributions:

| | Karachi (train) | Lahore (train) |
|--|----------------|----------------|
| Mean PM2.5 | 50.7 µg/m³ | 122.5 µg/m³ |
| Std PM2.5 | 36.5 µg/m³ | 106.5 µg/m³ |
| Max PM2.5 | 322 µg/m³ | 523 µg/m³ |

Karachi stays mostly in the Moderate/UfSG range year-round. Lahore swings from Good in monsoon to Hazardous in winter smog. A single model must compromise between these two regimes.

The worst errors are concentrated in **Lahore, October 22–29 2026** — a sudden smog onset where actual PM2.5 was 60–100 µg/m³ but lag features were reading 150–280 µg/m³ from the days prior, causing the model to overpredict by 125–200 µg/m³.

### Implication

Forecasts for Lahore during smog season onset should be treated with extra caution. The model systematically overpredicts when pollution is recovering and underpredicts hazardous spikes. This is the most significant practical limitation for real-world use.

---

## 3. Sensor Data Quality

### What was observed

- 12 Karachi and 18 configured Lahore sensors were used (community/low-cost monitors, not regulatory reference instruments).
- After cleaning: Karachi had 0.66% missing PM2.5 hours (longest gap: 67 hours). Lahore had 0.09% missing (longest gap: 5 hours).
- Gaps of up to 3 hours were linearly interpolated. Longer gaps remain as NaN and are excluded from training.
- Station `13817570` (Thokar Niaz Baig, Lahore) was successfully collected into `data/raw/` (7,696 rows). Station `14599816` (ARC Lahore) was deployed in late November 2025 and had minimal overlapping coverage for earlier months.

### Implication

The data is not reference-grade. Forecasts are appropriate for public awareness and research, not for regulatory compliance or emergency response decisions. The 67-hour Karachi gap in particular means there are periods where the city-level PM2.5 average was reconstructed from fewer sensors.

---

## 4. Single-Model Architecture vs. CityRouter Experiment

### What was observed

One model is trained on data from both cities with `city` as a one-hot encoded feature. This forces the model to find a single function that works for both Karachi and Lahore simultaneously.

An experimental city-specialist router was implemented in an earlier 35-feature run
(the experiment was not retained in the final source tree) to test whether two
separate models (one trained strictly on Karachi, one strictly on Lahore) could
outperform the combined model.

### Findings

The city-specific router yielded **67.4% AQI accuracy**, performing **worse** than the combined model (**70.9%**):
- **Karachi specialist:** 80.3% vs. 81.5% in combined model. Overfitted on its smaller training subset.
- **Lahore specialist:** 54.5% vs. 60.3% in combined model. Validation R² collapsed (0.0022), because Lahore's summer validation data lacked the regularizing influence that Karachi's summer data provides in the joint model.

### Implication

The joint model with one-hot encoded `city` remains the superior architecture. The shared representation allows the model to learn general weather–PM2.5 dynamics from the combined dataset, while the `city` intercept accounts for the baseline difference between the two locations.

---

## 5. Rejected Post-Processing Variants

### What was observed

Two simple corrections were tested on the held-out test split:

| Variant | MAE | RMSE | R² | AQI accuracy |
|---|---:|---:|---:|---:|
| Raw ensemble | **13.31** | **23.65** | **0.6695** | **71.65%** |
| Lahore clearing/recovery blend (`src/postprocess.py`) | 13.51 | 24.76 | 0.6377 | 71.33% |
| Conservative AQI boundary buffers (`src/evaluate.py`) | 13.35 | 23.66 | 0.6692 | 69.22% |

### Implication

Both heuristics made the test metrics worse, so neither is enabled by default. They
remain in the repository as documented experiments. The deployed Streamlit/FastAPI
services use raw ensemble output unless a caller explicitly enables the optional
clearing blend.

---

## 6. No Real-Time Re-training

### What was observed

The final model is trained on data up to October 2026 and saved as a static file. It is not retrained as new data becomes available.

### Implication

Model performance will degrade over time as climate patterns shift, new pollution sources emerge, or sensor networks change. For long-term deployment, a periodic retraining schedule (e.g., monthly) would be needed.

---

## 7. US EPA AQI Breakpoints Applied to Pakistan

### What was observed

The AQI category accuracy metric uses US EPA PM2.5 breakpoints (Good ≤ 9.0, Moderate ≤ 35.4, etc.). Pakistan does not have an official national AQI standard equivalent.

### Implication

The AQI accuracy figures are computed against a foreign standard. Health thresholds relevant to Pakistani populations may differ. This should be disclosed when presenting results to a Pakistani audience or policy context.

---

## 8. Historical Chart Data and Offline Snapshots

### What was observed

- The historical PM2.5 trend chart in the Streamlit application displays data from the training period, which ends at **2026-10-30 23:00 PKT**.
- The offline snapshot fallback file (`data/processed/live_fallback.csv`) contains the most recent feature row from this training period.
- In offline mode, predictions use this snapshot with a staleness indicator showing **312+ hours** (13+ days).

### Implication

The historical trend chart provides context for model training but is not live. In offline mode, predictions are demonstrations using stale data and should **not** be treated as current air quality alerts. The application clearly labels offline predictions and displays staleness warnings when data age exceeds 30 hours.

Users testing the application should:
- Use **live mode** for current forecasts (requires `OPENAQ_API_KEY`)
- Treat **offline mode** as a demonstration of functionality with historical data
- Check the "Data freshness" section for staleness indicators

---

## 9. Hugging Face Deployment Status

### What was observed

A complete Hugging Face Space deployment bundle has been prepared in `deploy/huggingface/` including:
- Space configuration (`README.md` with YAML header)
- Application code (`app.py` - Streamlit interface)
- Model file and metadata
- Requirements and dependencies
- Historical data for charts
- Error analysis figures

The bundler script (`prepare_space.py`) successfully creates a deployable Space folder with all necessary artifacts.

### Current Status

The deployment bundle is **prepared and ready** but the actual Space has **not yet been uploaded** to Hugging Face. This requires:
1. A Hugging Face account and authentication token
2. Pushing the bundle to a Space repository
3. Adding `OPENAQ_API_KEY` as a Space secret for live predictions

### Implication

The application is production-ready and has been manually tested locally with both FastAPI backend and Streamlit frontend. Deployment to Hugging Face Spaces is straightforward following the documented procedure in `deploy/huggingface/README.md`, but requires account credentials that were not available in the development environment.

For evaluation purposes, the locally running application at `http://localhost:8502` demonstrates full production functionality including live API data fetching, offline fallback, and all interactive features.

---

## Summary Table

| Limitation | Severity | Mitigation Available |
|------------|----------|----------------------|
| Feature ceiling (persistence dominance) | Medium | City-specific models, external data sources |
| Lahore–Karachi performance gap | High | City-specific models, more Lahore historical data |
| Community sensor data quality | Medium | Disclosed in DATA_SOURCES.md; use as awareness tool only |
| Single combined-city model | Medium | Train separate models per city |
| Post-processing heuristics did not help | Low | Disabled by default; documented in report |
| No real-time retraining | Low (for coursework) | Schedule periodic retraining for production |
| US EPA breakpoints used for Pakistan | Low | Note in report; apply local standard if available |
| Historical chart data ends Oct 2026 | Low | Expected behavior; chart shows training period |
| Offline snapshot staleness | Low | Clearly labeled; use live mode for current forecasts |
| Hugging Face Space not yet deployed | Low | Bundle prepared; requires HF account for upload |

