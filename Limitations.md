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

After retraining all four models with these 35 features (up from 30), the overall AQI category accuracy moved from **70.57% to 70.90%** — an improvement of only 0.33 percentage points.

### Why it happened

All new features are derived from `pm25` itself and are therefore highly correlated with `pm25_now`. Adding more PM2.5-derived columns does not give the model genuinely new information — it already extracts the trend signal from the existing lags. The model had effectively hit a ceiling where additional PM2.5 history features yielded diminishing returns.

### Implication

The model is, in essence, a sophisticated persistence forecast. It answers "what will PM2.5 be in 24 hours?" primarily by saying "roughly what it is right now, adjusted slightly by weather and calendar." For stable cities like Karachi this works well (AQI accuracy: 81.5%). For high-variance cities like Lahore it falls short (AQI accuracy: 60.3%).

### What would actually help

- **City-specific models**: Training separate models for Karachi and Lahore would let each model specialise. Lahore's model would see proportionally more extreme-value examples and could learn smog-season dynamics without being diluted by Karachi's stable distribution.
- **More historical data for Lahore**: The training window (June 2025 – June 2026) contains only one full Lahore smog season (Nov 2025 – Feb 2026). The test set hits October 2026 — the onset of the next smog season — which is partially out-of-distribution. Additional winters of Lahore data would directly address this.
- **External smog indicators**: Crop-burning season dates, Lahore traffic indices, or satellite aerosol optical depth (AOD) data would provide signals that are genuinely independent of the current PM2.5 reading.

---

## 2. Lahore–Karachi Performance Gap

### What was observed

| City | Test MAE | Test AQI Accuracy |
|------|----------|-------------------|
| Karachi | 6.44 µg/m³ | 81.5% |
| Lahore | 21.29 µg/m³ | 60.3% |

Lahore performs **3.3× worse** on MAE and **21 percentage points worse** on AQI accuracy.

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
- Two Lahore sensor IDs in `config.py` (`13817570`, `14599816`) have no corresponding raw CSV files, suggesting they were added to config but never collected.

### Implication

The data is not reference-grade. Forecasts are appropriate for public awareness and research, not for regulatory compliance or emergency response decisions. The 67-hour Karachi gap in particular means there are periods where the city-level PM2.5 average was reconstructed from fewer sensors.

---

## 4. Single-Model Architecture (Combined Cities)

### What was observed

One model is trained on data from both cities with `city` as a one-hot encoded feature. This forces the model to find a single function that works for both Karachi and Lahore simultaneously.

### Implication

The city encoding gives the model a constant offset per city but cannot give it city-specific feature weights. For example, the relationship between temperature and PM2.5 is different in coastal Karachi vs. landlocked Lahore — but the model uses the same coefficient for both. Separate models per city would remove this constraint.

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

