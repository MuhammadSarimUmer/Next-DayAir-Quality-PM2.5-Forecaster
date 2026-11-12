# Testing the Streamlit frontend

The Streamlit app (`app/streamlit_app.py`) supports two backends:

- **Local pipeline** (default): runs inference inside the Streamlit process.
- **FastAPI service**: sends HTTP requests to `app/api.py` (`/health`,
  `/metadata`, `/predict`, and `/aqi`).

Both modes use the same saved model, saved history chart, and pre-rendered
accuracy figures. The FastAPI mode is the one that demonstrates a separate
frontend talking to a backend over HTTP.

## 1. Install dependencies

From the project root:

```powershell
python -m pip install -r requirements.txt
```

## 2. Quick test: local pipeline only

This does not require a second process.

```powershell
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>). In the sidebar,
leave **Backend** on **Local pipeline**.

## 3. Full test: Streamlit frontend connected to FastAPI

### Terminal 1 - start the backend

```powershell
uvicorn app.api:app --reload --port 8000
```

Verify it is up:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
```

Expected: `status = ok`, `n_features = 47`, and the test metrics block. You can
also open <http://127.0.0.1:8000/docs> for Swagger UI.

### Terminal 2 - start the frontend in API mode

```powershell
$env:API_BASE_URL = "http://127.0.0.1:8000"
streamlit run app/streamlit_app.py
```

When `API_BASE_URL` is set, the sidebar **Backend** selector defaults to
**FastAPI service**. You can also select it manually and type the URL.

## 4. Manual checklist

| # | Action | Expected result |
|---|---|---|
| 1 | Open the app | Forecast card, AQI category, advice, 2 charts, and 3 accuracy images load without a traceback. |
| 2 | Change **City** to Karachi | Forecast and observed value change; the card names Karachi. |
| 3 | Set **Input source** to **Saved snapshot only** | Data source becomes `Offline`; stale-hours warning appears when the snapshot is old. |
| 4 | Set **Input source** to **Live APIs** | Source is `Live` when the key/network work; otherwise the app labels the offline fallback instead of crashing. |
| 5 | Set **Backend** to **FastAPI service** | Sidebar metrics and the forecast still load, now from the API. The input-snapshot table is replaced by an API note; freshness values still appear. |
| 6 | Move the **Manual AQI** slider | AQI category and advice update. In API mode this calls `POST /aqi`. |
| 7 | Press **Refresh inputs** | Cached API values are cleared and the page reloads. |
| 8 | Stop the `uvicorn` process and reload the page | The app shows a clear "FastAPI backend error" message and stops, with no Python traceback. |
| 9 | Switch back to **Local pipeline** | The app works without the backend again. |

The history chart is deliberately historical: it ends on the last date in
`data/processed/features.csv` and is labelled as such. It is not a live series.

## 5. Direct API smoke tests

With the backend running:

```powershell
# Health and model metadata
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5

# Deterministic offline forecast
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"city":"lahore","offline":true}' | ConvertTo-Json -Depth 5

# Manual AQI
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/aqi `
  -ContentType "application/json" `
  -Body '{"pm25":35.5}' | ConvertTo-Json -Depth 5
```

Expected: the forecast response contains `predicted_pm25`, `aqi`, `category`,
`source`, `feature_timestamp`, and `stale_hours`. The AQI response contains
`aqi = 101` and `category = Unhealthy for Sensitive Groups` for `35.5`.

## 6. Automated tests

```powershell
python -m pytest tests -q
python -m compileall -q src app deploy tests
```

Expected: **31 passed** and `compileall` exits silently with code 0. A
non-blocking Starlette/httpx deprecation warning may appear.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Model artifact is missing` | Run `python -m src.train_final` from the project root. |
| `FastAPI backend error` / connection refused | Start `uvicorn` in another terminal; confirm the port and `API_BASE_URL` match. |
| Live mode falls back to `Offline` | Add `OPENAQ_API_KEY` to the project-root `.env`, or use **Saved snapshot only**. |
| `Address already in use` | Change the backend port, e.g. `--port 8010`, and set `API_BASE_URL` to match. |
| History chart or accuracy figures missing | Ensure `data/processed/features.csv` and `reports/figures/*.png` exist. The Hugging Face bundler copies both. |
| 422 from `/predict` | Caller-supplied `features` must contain all 47 columns with finite numeric values; the Streamlit UI does not send malformed features. |

## 8. What to capture for the submission

- A screenshot with **Backend = Local pipeline**.
- A screenshot with **Backend = FastAPI service**, with `uvicorn` visible in a
  separate terminal.
- The output of `Invoke-RestMethod http://127.0.0.1:8000/health`.
- The `python -m pytest tests -q` summary.

Do not commit `.env`, and do not push or deploy while testing unless you intend
to.
