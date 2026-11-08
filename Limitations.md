# Known Limitations

This document records the known limitations, failure modes, and design trade-offs of the Next-Day Air Quality PM2.5 Forecaster. It is intended for the final report and for anyone continuing development on this project.

---

## 1. Feature Ceiling — Persistence-Dominated Model

### What was observed

After completing the modelling stage, the top feature by importance was `pm25_now` at **86.0%**. All remaining 34 features combined accounted for only 14% of the model's decisions. This was confirmed across all four trained models (Ridge, KNN, Random Forest, LightGBM).

An explicit improvement attempt was made in November 2026 (Option 1): five new momentum and trend features were added to `features.py`:

| Feature | Purpose |
|---------|---------|
| `pm25_change_6h` | 6-hour trend direction |
| `pm25_change_24h` | Day-over-day direction |
| `pm25_rate_of_change` | 3-hour PM2.5 velocity (µg/m³ per hour) |
| `pm25_7d_lag` | Same hour last week (weekly seasonality anchor) |
| `pm25_roll_min_24h` | 24-hour floor (recovery signal) |

After retraining all four models with these 35 features (up from 30), the overall AQI category accuracy reached **71.34%** (test set, up from the 69.83% baseline).

### Why it happened

All new features are derived from `pm25` itself and are therefore highly correlated with `pm25_now`. Adding more PM2.5-derived columns does not give the model genuinely new information — it already extracts the trend signal from the existing lags. The model had effectively hit a ceiling where additional PM2.5 history features yielded diminishing returns.

### Implication

The model is, in essence, a sophisticated persistence forecast. It answers "what will PM2.5 be in 24 hours?" primarily by saying "roughly what it is right now, adjusted slightly by weather and calendar." For stable cities like Karachi this works well (AQI accuracy: 82.6%). For high-variance cities like Lahore it falls short (AQI accuracy: 60.1%).

### What would actually help

- **City-specific models**: Training separate models for Karachi and Lahore would let each model specialise. Lahore's model would see proportionally more extreme-value examples and could learn smog-season dynamics without being diluted by Karachi's stable distribution.
- **More historical data for Lahore**: The training window (June 2025 – June 2026) contains only one full Lahore smog season (Nov 2025 – Feb 2026). The test set hits October 2026 — the onset of the next smog season — which is partially out-of-distribution. Additional winters of Lahore data would directly address this. **An attempt was made to extend the collection window back to January 2022 via OpenAQ and WAQI/AQICN APIs. However, ground community monitors in Karachi and Lahore only began reporting around May–July 2025, and free API archives do not provide pre-2025 hourly ground sensor data. This is a hard external data constraint, not a pipeline limitation.**
- **External smog indicators**: Crop-burning season dates, Lahore traffic indices, or satellite aerosol optical depth (AOD) data would provide signals that are genuinely independent of the current PM2.5 reading.

---

## 2. Lahore–Karachi Performance Gap

### What was observed

| City | Test MAE | Test AQI Accuracy |
|------|----------|-------------------|
| Karachi | 6.25 µg/m³ | 82.60% |
| Lahore | 20.75 µg/m³ | 60.09% |

Lahore performs **3.3× worse** on MAE and **22.5 percentage points worse** on AQI accuracy.

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

- 12 Karachi sensors and 16 Lahore sensors were used (community/low-cost monitors, not regulatory reference instruments).
- After cleaning: Karachi had 0.66% missing PM2.5 hours (longest gap: 67 hours). Lahore had 0.09% missing (longest gap: 5 hours).
- Gaps of up to 3 hours were linearly interpolated. Longer gaps remain as NaN and are excluded from training.
- Station `13817570` (Thokar Niaz Baig, Lahore) was successfully collected into `data/raw/` (7,696 rows). Station `14599816` (ARC Lahore) was deployed in late November 2025 and had minimal overlapping coverage for earlier months.

### Implication

The data is not reference-grade. Forecasts are appropriate for public awareness and research, not for regulatory compliance or emergency response decisions. The 67-hour Karachi gap in particular means there are periods where the city-level PM2.5 average was reconstructed from fewer sensors.

---

## 4. Single-Model Architecture vs. CityRouter Experiment

### What was observed

One model is trained on data from both cities with `city` as a one-hot encoded feature. This forces the model to find a single function that works for both Karachi and Lahore simultaneously.

An experimental city-specialist router (`CityRouter` in `src/train_city_router.py`) was implemented to test whether two separate models (one trained strictly on Karachi, one strictly on Lahore) could outperform the combined model.

### Findings

The city-specific router yielded **67.4% AQI accuracy**, performing **worse** than the combined model (**70.9%**):
- **Karachi specialist:** 80.3% vs. 81.5% in combined model. Overfitted on its smaller training subset.
- **Lahore specialist:** 54.5% vs. 60.3% in combined model. Validation R² collapsed (0.0022), because Lahore's summer validation data lacked the regularizing influence that Karachi's summer data provides in the joint model.

### Implication

The joint model with one-hot encoded `city` remains the superior architecture. The shared representation allows the model to learn general weather–PM2.5 dynamics from the combined dataset, while the `city` intercept accounts for the baseline difference between the two locations.

---

## 5. No Real-Time Re-training

### What was observed

The final model is trained on data up to October 2026 and saved as a static file. It is not retrained as new data becomes available.

### Implication

Model performance will degrade over time as climate patterns shift, new pollution sources emerge, or sensor networks change. For long-term deployment, a periodic retraining schedule (e.g., monthly) would be needed.

---

## 6. US EPA AQI Breakpoints Applied to Pakistan

### What was observed

The AQI category accuracy metric uses US EPA PM2.5 breakpoints (Good ≤ 9.0, Moderate ≤ 35.4, etc.). Pakistan does not have an official national AQI standard equivalent.

### Implication

The AQI accuracy figures are computed against a foreign standard. Health thresholds relevant to Pakistani populations may differ. This should be disclosed when presenting results to a Pakistani audience or policy context.

---

## Summary Table

| Limitation | Severity | Mitigation Available |
|------------|----------|----------------------|
| Feature ceiling (persistence dominance) | Medium | City-specific models, external data sources |
| Lahore–Karachi performance gap | High | City-specific models, more Lahore historical data |
| Community sensor data quality | Medium | Disclosed in DATA_SOURCES.md; use as awareness tool only |
| Single combined-city model | Medium | Train separate models per city |
| No real-time retraining | Low (for coursework) | Schedule periodic retraining for production |
| US EPA breakpoints used for Pakistan | Low | Note in report; apply local standard if available |

