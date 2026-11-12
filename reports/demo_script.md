# Demo Script (6-8 minutes)

## Before the demo

```powershell
# From the project root
python -m src.live_data --refresh-fallback --offline
streamlit run app/streamlit_app.py
# In a second terminal:
uvicorn app.api:app --reload --port 8000
```

Open:
- Streamlit: http://localhost:8501
- FastAPI docs: http://127.0.0.1:8000/docs

Set `OPENAQ_API_KEY` in the project-root `.env` for live mode. If the venue Wi-Fi
or the key is unavailable, select **Saved snapshot only** in the Streamlit sidebar.
The app will label the inputs as offline.

## 1. Problem (45 seconds)

"Karachi and Lahore both have severe air-quality episodes, but most apps show only
the current reading. We forecast PM2.5 24 hours ahead so a person can decide whether
to exercise, send children to school, or carry a mask."

## 2. Show the result first (1 minute)

1. Open the Streamlit app.
2. Pick **Lahore**.
3. Point to the big card: predicted PM2.5, AQI number, category, and colour.
4. Read one sentence of general advice and one sensitive-group action.
5. Point to the **Data freshness** panel: feature hour, latest sensor hour, and
   staleness. Say: "If inputs are stale or the API is down, the app says so instead
   of pretending the data is live."
6. Scroll to **Recent trend and tomorrow's forecast**: the 7-day PM2.5 line chart
   and the now-versus-tomorrow bar chart. Open **How accurate is the model?** to
   show the test-set scatter, monthly error, and confusion figures.

## 3. Explain the pipeline (1.5 minutes)

Show the README feature table and say:

- 24-hour target created with a negative shift.
- 47 past-only features: lags, rolling statistics, momentum, weather, wind vector,
  CAMS aerosol, stagnation, smog season, calendar, city.
- Chronological 70/15/15 split with a 24-hour purge gap.
- Preprocessing (median imputation + scaling + one-hot city) is fitted on train only.
- Models: Ridge, KNN, Random Forest, LightGBM, weighted ensemble.
- Tuning uses `TimeSeriesSplit`, never random CV.

## 4. Show the numbers (1 minute)

Open `reports/final_report.md` section 6 or the README results table.

- Test MAE 13.31 µg/m³ versus persistence 14.54 (8.5% better).
- RMSE 23.65 versus 26.98 (12.3% better).
- R² 0.6695 versus 0.5700.
- AQI category accuracy 71.65% versus 69.99%.
- Karachi 6.20 MAE / 82.0% AQI; Lahore 20.42 MAE / 61.3% AQI.

## 5. Show where it fails (1 minute)

Open `reports/error_analysis.md` and the figure
`reports/figures/error_lahore_smog.png`.

- Top-5% spikes: MAE 41.43, 90.3% underpredicted.
- Rapid clearing: MAE 91.95, overpredicted.
- Worst day: Lahore 29 October 2026, MAE 108.82.
- Most category errors are one band away, not several.

Say: "This is the honest part. The model is a strong persistence model with weather
and regime adjustments; it is not an emergency warning system."

## 6. Show the API (45 seconds)

In the Swagger UI:

1. `GET /health` returns model status and test metrics.
2. `POST /predict` body:

```json
{"city": "lahore", "offline": true}
```

3. `POST /aqi` body:

```json
{"pm25": 85}
```

Point out that `/predict` returns the same AQI and advice structure used by the app.

## 7. Close (45 seconds)

"Everything is reproducible: one command per stage, saved model metadata, per-breakpoint
AQI tests, error-analysis outputs, and a Hugging Face Space bundle. The main remaining
risk is Lahore smog transitions; the next step would be more Lahore winters or
city-specific models with external fire data."

## Likely questions

| Question | Short answer |
|---|---|
| Why not a random split? | Autocorrelation would leak neighbouring test hours into training; we split chronologically with a 24-hour purge. |
| Why not deep learning? | Only ~16 months of hourly data; gradient boosting and forests are stronger and faster on tabular data of this size. |
| Why is Lahore worse? | It has a heavy-tailed distribution and rapid smog onset/clearing; Karachi is smoother and close to persistence. |
| Did the ensemble always win? | Random Forest had the best validation MAE by 0.02; we selected the ensemble by a pre-declared 0.2 µg/m³ stability rule. |
| Are the AQI thresholds Pakistani? | No; they are 2024 US EPA breakpoints because no equivalent national table is configured. This is stated in the report. |
| Is the data reference-grade? | No; OpenAQ stations are community/low-cost monitors, suitable for awareness and research only. |
| What happens without an API key? | `live_data.py` falls back to a saved snapshot and labels it as offline. |
| Did you test live data? | Yes: on 2026-11-12 Lahore returned 290.1 µg/m³ and Karachi 45.5 µg/m³ at 1.8 h staleness; weather and CAMS each returned 288 rows. |
| How is the API validated? | Pydantic models reject unknown cities and incomplete feature rows; `tests/test_api.py` covers 200/422 responses. |
| How many tests pass? | 28: every AQI breakpoint, pipeline logic, and the FastAPI contract. |
