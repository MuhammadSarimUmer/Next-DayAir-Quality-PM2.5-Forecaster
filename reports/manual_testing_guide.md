# Manual Testing Guide: Next-Day PM2.5 Forecaster

**Version:** 1.0  
**Date:** November 12, 2026  
**Project:** Next-Day Air Quality (PM2.5) Forecaster for Karachi and Lahore

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [How to Start the Project](#how-to-start-the-project)
4. [Application URLs](#application-urls)
5. [Testing Checklist](#testing-checklist)
6. [Detailed Test Scenarios](#detailed-test-scenarios)
7. [Expected Results](#expected-results)
8. [How to Report Bugs](#how-to-report-bugs)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides step-by-step instructions for manually testing the PM2.5 Air Quality Forecaster application. The system consists of:

- **FastAPI Backend** - REST API service on port 8001
- **Streamlit Frontend** - Interactive web application on port 8502

The application predicts PM2.5 concentrations 24 hours ahead for Karachi and Lahore, translates predictions into EPA AQI health categories, and provides actionable health advice.

---

## Prerequisites

Before starting, ensure:

1. Python 3.8+ is installed
2. All dependencies are installed: `pip install -r requirements.txt`
3. `.env` file exists with valid `OPENAQ_API_KEY` (for live mode)
4. Model file exists: `models/final_pipeline.joblib`
5. Ports 8001 and 8502 are available

---

## How to Start the Project

### Step 1: Start the FastAPI Backend

Open a terminal/command prompt and run:

```bash
cd g:\Desktop\ml\Next-DayAir-Quality-PM2.5-Forecaster
uvicorn app.api:app --reload --port 8001
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started server process [XXXXX]
INFO:     Application startup complete.
```

### Step 2: Start the Streamlit Frontend

Open a **new** terminal/command prompt and run:

**Windows:**
```cmd
cd g:\Desktop\ml\Next-DayAir-Quality-PM2.5-Forecaster
set PYTHONPATH=%CD%
set API_BASE_URL=http://127.0.0.1:8001
streamlit run app/streamlit_app.py --server.port 8502
```

**Linux/Mac:**
```bash
cd /path/to/Next-DayAir-Quality-PM2.5-Forecaster
export PYTHONPATH=$PWD
export API_BASE_URL=http://127.0.0.1:8001
streamlit run app/streamlit_app.py --server.port 8502
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8502
```

### Step 3: Open the Application

The Streamlit app should automatically open in your default browser. If not, manually navigate to:

**http://localhost:8502**

---

## Application URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit App** | http://localhost:8502 | Main interactive web interface |
| **FastAPI Docs** | http://127.0.0.1:8001/docs | Interactive API documentation (Swagger UI) |
| **API Health** | http://127.0.0.1:8001/health | Backend health check |
| **API Root** | http://127.0.0.1:8001/ | API service information |

---

## Testing Checklist

Use this checklist to track your testing progress. Mark each test as PASS ✓, FAIL ✗, or N/A.

### A. Backend API Tests

| # | Test | Status | Notes |
|---|------|--------|-------|
| A1 | Health endpoint returns status "ok" | ☐ | |
| A2 | Health endpoint shows model loaded | ☐ | |
| A3 | Metadata endpoint returns test metrics | ☐ | |
| A4 | Predict endpoint works for Lahore (live mode) | ☐ | |
| A5 | Predict endpoint works for Karachi (live mode) | ☐ | |
| A6 | Predict endpoint works for Lahore (offline mode) | ☐ | |
| A7 | Predict endpoint works for Karachi (offline mode) | ☐ | |
| A8 | Manual AQI endpoint calculates correctly | ☐ | |
| A9 | Invalid city name returns error | ☐ | |
| A10 | Invalid PM2.5 value returns validation error | ☐ | |

### B. Frontend UI Tests

| # | Test | Status | Notes |
|---|------|--------|-------|
| B1 | Page loads without errors | ☐ | |
| B2 | City selection dropdown works (Lahore/Karachi) | ☐ | |
| B3 | Input source toggle works (Live/Snapshot) | ☐ | |
| B4 | Backend toggle works (Local/FastAPI) | ☐ | |
| B5 | Forecast card displays predicted PM2.5 | ☐ | |
| B6 | Forecast card displays AQI category with color | ☐ | |
| B7 | Health advice section displays correctly | ☐ | |
| B8 | Action recommendations list displays | ☐ | |
| B9 | Input snapshot table shows feature values | ☐ | |
| B10 | Data freshness information displays | ☐ | |
| B11 | Recent trend chart renders | ☐ | |
| B12 | Now vs Tomorrow bar chart renders | ☐ | |
| B13 | Model accuracy expander works | ☐ | |
| B14 | Error analysis figures display (3 images) | ☐ | |
| B15 | Manual AQI slider works | ☐ | |
| B16 | Refresh button clears cache | ☐ | |

### C. Prediction Tests

| # | Test | Status | Notes |
|---|------|--------|-------|
| C1 | Lahore prediction returns valid PM2.5 value | ☐ | |
| C2 | Karachi prediction returns valid PM2.5 value | ☐ | |
| C3 | Predictions are positive numbers | ☐ | |
| C4 | AQI category matches PM2.5 value | ☐ | |
| C5 | Health advice is appropriate for category | ☐ | |
| C6 | Observed PM2.5 "Now" is displayed | ☐ | |
| C7 | Feature timestamp is recent (live mode) | ☐ | |
| C8 | Staleness hours is reasonable (<24h in live) | ☐ | |
| C9 | Offline mode shows older timestamp | ☐ | |
| C10 | Offline mode shows staleness warning | ☐ | |

### D. Data Mode Tests

| # | Test | Status | Notes |
|---|------|--------|-------|
| D1 | Live mode fetches recent data | ☐ | |
| D2 | Live mode shows "live" as source | ☐ | |
| D3 | Offline mode uses snapshot data | ☐ | |
| D4 | Offline mode shows "offline" as source | ☐ | |
| D5 | Local pipeline mode works | ☐ | |
| D6 | FastAPI service mode works | ☐ | |
| D7 | Switching modes updates prediction | ☐ | |
| D8 | Warning shown for stale data (>30h) | ☐ | |

### E. Edge Cases & Error Handling

| # | Test | Status | Notes |
|---|------|--------|-------|
| E1 | App handles missing API key gracefully | ☐ | |
| E2 | App handles network errors (API down) | ☐ | |
| E3 | App handles invalid API responses | ☐ | |
| E4 | Manual AQI accepts values 0-400 | ☐ | |
| E5 | Manual AQI rejects negative values | ☐ | |
| E6 | App handles missing model file | ☐ | |
| E7 | Error messages are user-friendly | ☐ | |

---

## Detailed Test Scenarios

### Test Scenario 1: Basic Prediction Flow (Lahore)

**Objective:** Verify the complete prediction workflow for Lahore

**Steps:**
1. Open http://localhost:8502
2. In the sidebar, select **"lahore"** from City dropdown
3. Keep Input source as **"Live APIs (fallback to snapshot)"**
4. Keep Backend as **"FastAPI service"**
5. Observe the main forecast card

**Expected Results:**
- Forecast card displays:
  - City name: "Tomorrow's forecast for Lahore"
  - Predicted PM2.5 value (e.g., "273.4 ug/m3 PM2.5")
  - AQI number and category (e.g., "AQI 336 - Hazardous")
  - Background color matches AQI category (Hazardous = dark red/maroon)
- Below the card:
  - "Observed now" shows current PM2.5
  - "Input hour" shows timestamp in format "MM-DD HH:MM"
  - "Data source" shows "Live"
- Health advice section displays:
  - Summary text explaining the health risk
  - General public guidance
  - Sensitive groups guidance
  - 4-6 actionable recommendations as bullet points

**Sample Expected Output (November 2026):**
```
Tomorrow's forecast for Lahore
273.4 ug/m3 PM2.5
AQI 336 - Hazardous

Observed now: 262.9 ug/m3
Input hour: 11-12 21:00
Data source: Live

What this means:
Health warning of emergency conditions: Everyone is likely to be seriously affected.

General public: Avoid all physical activity outdoors. Remain indoors in clean air spaces.
Sensitive groups: Remain strictly indoors and avoid any physical exertion.

Recommended actions:
- Stay indoors with doors and windows tightly closed.
- Run air purifiers on high setting.
- Wear an N95 mask even for short trips outdoors.
- Seek medical attention if experiencing breathing distress or chest tightness.
```

### Test Scenario 2: Basic Prediction Flow (Karachi)

**Objective:** Verify the complete prediction workflow for Karachi

**Steps:**
1. In the sidebar, change City to **"karachi"**
2. Keep other settings the same
3. Observe the changes in the forecast

**Expected Results:**
- Forecast card updates with Karachi data:
  - City name: "Tomorrow's forecast for Karachi"
  - Different PM2.5 and AQI values (typically lower than Lahore)
  - Appropriate AQI category (e.g., "Unhealthy for Sensitive Groups")
  - Background color matches the category (orange for Unhealthy for Sensitive)
- Observed value reflects Karachi's current reading
- Health advice is appropriate for the AQI level
- Data source shows "Live"

**Sample Expected Output:**
```
Tomorrow's forecast for Karachi
55.3 ug/m3 PM2.5
AQI 150 - Unhealthy for Sensitive Groups

Observed now: 47.8 ug/m3
```

### Test Scenario 3: Offline Mode

**Objective:** Verify the application works with saved snapshot data

**Steps:**
1. In the sidebar, select City: **"lahore"**
2. Change Input source to **"Saved snapshot only"**
3. Wait for the prediction to update

**Expected Results:**
- Forecast card displays prediction based on snapshot data
- Data source shows **"Offline"**
- Feature hour shows an older timestamp (e.g., "10-30 23:00")
- Staleness hours is high (e.g., "312.2 hours")
- Warning message appears: "Using saved snapshot (offline mode requested)"
- Yellow/warning banner shows: "Inputs are more than 30 hours old. Treat this forecast as a demo, not a live alert."
- Prediction values may differ from live mode

**Sample Expected Output:**
```
Feature hour: 10-30 23:00 PKT
Latest sensor hour: 10-30 23:00 PKT
Staleness: 312.2 hours
⚠️ Using saved snapshot (offline mode requested)
⚠️ Inputs are more than 30 hours old. Treat this forecast as a demo, not a live alert.
```

### Test Scenario 4: Local Pipeline Mode

**Objective:** Verify the application works without FastAPI backend

**Steps:**
1. In the sidebar, change Backend to **"Local pipeline"**
2. Select City: **"karachi"**
3. Input source: **"Live APIs (fallback to snapshot)"**
4. Observe the prediction

**Expected Results:**
- Prediction is computed locally within Streamlit
- Results are identical to FastAPI mode for the same inputs
- "Input snapshot" table displays 9 feature values:
  - pm25_now
  - pm25_lag_24h
  - pm25_roll_mean_24h
  - pm25_roll_mean_72h
  - temperature_2m
  - relative_humidity_2m
  - wind_speed_10m
  - cams_pm2_5
  - stagnation_index
- Freshness information displays correctly
- No FastAPI connection errors

### Test Scenario 5: Recent Trend Chart

**Objective:** Verify the historical PM2.5 trend visualization

**Steps:**
1. Scroll down to "Recent trend and tomorrow's forecast" section
2. Observe the line chart on the left
3. Switch between Lahore and Karachi

**Expected Results:**
- Line chart displays two lines:
  - Blue line: "Observed PM2.5" (hourly values)
  - Orange line: "24h rolling mean" (smoothed trend)
- X-axis shows dates and times
- Y-axis shows PM2.5 concentration (ug/m3)
- Chart caption shows the ending timestamp of historical data
- Chart updates when switching cities
- Lahore shows higher values than Karachi

### Test Scenario 6: Now vs Tomorrow Comparison

**Objective:** Verify the bar chart comparing current and predicted values

**Steps:**
1. Look at the bar chart on the right side of the trend section
2. Compare the two bars

**Expected Results:**
- Two blue bars labeled:
  - "Now" - current observed PM2.5
  - "Tomorrow (+24h)" - predicted PM2.5
- Y-axis shows PM2.5 (ug/m3)
- Bar heights accurately represent the values
- Caption explains the comparison
- Values match those shown in the forecast card

### Test Scenario 7: Model Accuracy Panel

**Objective:** Verify the model performance metrics and visualizations

**Steps:**
1. Scroll to the bottom section
2. Click on "How accurate is the model? (held-out test set)" expander
3. Review the metrics and figures

**Expected Results:**
- Four metrics displayed:
  - **MAE:** ~13.31 ug/m3
  - **RMSE:** ~23.65 ug/m3
  - **R2:** ~0.6695
  - **AQI accuracy:** ~71.65%
- Three error analysis figures display:
  1. **error_scatter.png**: Scatter plot of actual vs predicted PM2.5
  2. **error_by_month.png**: Bar chart showing MAE by month
  3. **error_confusion.png**: Confusion matrix of AQI categories
- Each figure has a descriptive caption
- Figures load without errors

### Test Scenario 8: Manual AQI Calculator

**Objective:** Verify the manual PM2.5 to AQI conversion tool

**Steps:**
1. In the right sidebar section, expand "Manual AQI / advice simulator"
2. Move the slider to different PM2.5 values
3. Test the following values:
   - 5.0 ug/m3 (Good)
   - 20.0 ug/m3 (Moderate)
   - 50.0 ug/m3 (Unhealthy for Sensitive Groups)
   - 85.0 ug/m3 (Unhealthy)
   - 150.0 ug/m3 (Very Unhealthy)
   - 300.0 ug/m3 (Hazardous)

**Expected Results for Each Value:**

| PM2.5 | Expected AQI | Expected Category |
|-------|--------------|-------------------|
| 5.0 | 21 | Good |
| 20.0 | 68 | Moderate |
| 50.0 | 137 | Unhealthy for Sensitive Groups |
| 85.0 | 172 | Unhealthy |
| 150.0 | 225 | Very Unhealthy |
| 300.0 | 364 | Hazardous |

- AQI category and number update instantly
- Summary text changes appropriately
- Advice matches the severity level
- Colors correspond to categories:
  - Good: Green (#00E400)
  - Moderate: Yellow (#FFFF00)
  - Unhealthy for Sensitive: Orange (#FF7E00)
  - Unhealthy: Red (#FF0000)
  - Very Unhealthy: Purple (#8F3F97)
  - Hazardous: Maroon (#7E0023)

### Test Scenario 9: Refresh Functionality

**Objective:** Verify the refresh button clears cached data

**Steps:**
1. Note the current predicted PM2.5 value
2. Click the **"Refresh inputs"** button in the sidebar
3. Wait for the app to reload
4. Check if new data is fetched

**Expected Results:**
- Button click triggers cache clearing
- Spinner appears briefly: "Fetching the latest air-quality and weather inputs..."
- Prediction may update if new data is available
- Timestamp updates to the latest available hour
- No errors occur

### Test Scenario 10: API Health Check

**Objective:** Verify the FastAPI backend health endpoint

**Steps:**
1. Open a web browser
2. Navigate to: **http://127.0.0.1:8001/health**
3. Review the JSON response

**Expected Results:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "G:\\Desktop\\ml\\Next-DayAir-Quality-PM2.5-Forecaster\\models\\final_pipeline.joblib",
  "n_features": 47,
  "test_metrics": {
    "mae": 13.30916635258083,
    "rmse": 23.649916380310223,
    "r2": 0.669518393348154,
    "aqi_accuracy": 0.7165411949685535
  }
}
```

- Status is "ok"
- model_loaded is true
- n_features is 47
- test_metrics contains all four metrics

### Test Scenario 11: API Interactive Documentation

**Objective:** Test API endpoints using Swagger UI

**Steps:**
1. Open: **http://127.0.0.1:8001/docs**
2. Expand **POST /predict** endpoint
3. Click "Try it out"
4. Enter the following JSON:
```json
{
  "city": "lahore",
  "offline": false
}
```
5. Click "Execute"
6. Review the response

**Expected Results:**
- Swagger UI displays all available endpoints:
  - GET /
  - GET /health
  - POST /predict
  - POST /aqi
  - GET /metadata
- Response code: 200
- Response body contains:
  - city: "Lahore"
  - source: "live"
  - predicted_pm25: (number)
  - aqi: (number)
  - category: (string)
  - actions: (array)
  - All required fields present

### Test Scenario 12: Invalid Input Handling (API)

**Objective:** Verify API validates inputs correctly

**Test 12A: Invalid City Name**

**Steps:**
1. Go to http://127.0.0.1:8001/docs
2. Expand POST /predict
3. Enter:
```json
{
  "city": "islamabad",
  "offline": false
}
```
4. Click Execute

**Expected Results:**
- Response code: 422 (Unprocessable Entity)
- Error detail mentions: "city must be one of ['karachi', 'lahore']"

**Test 12B: Invalid PM2.5 Value (Manual AQI)**

**Steps:**
1. Expand POST /aqi
2. Enter:
```json
{
  "pm25": -10.0
}
```
3. Click Execute

**Expected Results:**
- Response code: 422
- Validation error indicating pm25 must be >= 0

**Test 12C: Out of Range PM2.5**

**Steps:**
1. Expand POST /aqi
2. Enter:
```json
{
  "pm25": 3000.0
}
```
3. Click Execute

**Expected Results:**
- Response code: 422
- Validation error indicating pm25 must be <= 2000

### Test Scenario 13: Backend Down Handling

**Objective:** Verify Streamlit handles backend failure gracefully

**Steps:**
1. Stop the FastAPI backend (Ctrl+C in terminal)
2. In Streamlit app, ensure Backend is set to **"FastAPI service"**
3. Try to get a prediction

**Expected Results:**
- Red error box appears with message like:
  - "FastAPI backend error at `http://127.0.0.1:8001`: [Connection error details]"
- Blue info box appears:
  - "Start it with `uvicorn app.api:app --reload --port 8000`, then reload this page or press Refresh inputs."
- App remains functional (doesn't crash)
- User can switch to "Local pipeline" mode to continue using the app

**Recovery Steps:**
1. Restart FastAPI backend
2. Click "Refresh inputs" in Streamlit
3. Prediction should work again

### Test Scenario 14: Missing API Key Handling

**Objective:** Verify graceful degradation when API key is missing

**Steps:**
1. Rename `.env` file temporarily to `.env.backup`
2. Set Input source to **"Live APIs (fallback to snapshot)"**
3. Try to get a prediction with Local pipeline mode

**Expected Results:**
- App attempts live data fetch
- Falls back to snapshot data automatically
- Warning message shows the issue
- Prediction still works with offline data
- OR error message suggests checking .env file

**Recovery Steps:**
1. Rename `.env.backup` back to `.env`
2. Click "Refresh inputs"

---

## Expected Results

### AQI Category Color Coding

Each AQI category should display with the correct background color:

| Category | PM2.5 Range (ug/m3) | AQI Range | Hex Color | Visual Color |
|----------|---------------------|-----------|-----------|--------------|
| Good | 0.0 - 9.0 | 0 - 50 | #00E400 | Bright Green |
| Moderate | 9.1 - 35.4 | 51 - 100 | #FFFF00 | Yellow |
| Unhealthy for Sensitive Groups | 35.5 - 55.4 | 101 - 150 | #FF7E00 | Orange |
| Unhealthy | 55.5 - 125.4 | 151 - 200 | #FF0000 | Red |
| Very Unhealthy | 125.5 - 225.4 | 201 - 300 | #8F3F97 | Purple |
| Hazardous | 225.5+ | 301+ | #7E0023 | Maroon |

### Health Advice Examples

**Good (PM2.5 < 9):**
- Summary: "Air quality is satisfactory, and air pollution poses little or no risk."
- General public: "Enjoy your usual outdoor activities."
- Actions: None or minimal

**Moderate (PM2.5 9-35):**
- Summary: "Air quality is acceptable. However, there may be a risk for some people."
- General public: "OK to be active outside."
- Sensitive groups: "Consider reducing prolonged or heavy exertion."

**Unhealthy for Sensitive Groups (PM2.5 35-55):**
- Summary: "Members of sensitive groups may experience health effects."
- Actions include: wearing masks outdoors, closing windows during peak hours

**Unhealthy (PM2.5 55-125):**
- Summary: "Everyone may begin to experience health effects."
- Actions include: avoiding outdoor exercise, wearing N95 masks

**Very Unhealthy (PM2.5 125-225):**
- Summary: "Health alert: The risk of adverse health effects is increased for everyone."
- Actions include: staying indoors, using air purifiers

**Hazardous (PM2.5 > 225):**
- Summary: "Health warning of emergency conditions: Everyone is likely to be seriously affected."
- Actions include: avoiding all outdoor activity, seeking medical attention if needed

---

## How to Report Bugs

If you encounter any issues during testing, please document them using this format:

### Bug Report Template

```
**Bug ID:** [Sequential number, e.g., BUG-001]
**Severity:** [Critical / High / Medium / Low]
**Test Scenario:** [Which test scenario from this guide]
**Component:** [Frontend / Backend / Both]

**Description:**
[Clear description of what went wrong]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [...]

**Expected Result:**
[What should have happened]

**Actual Result:**
[What actually happened]

**Screenshots/Error Messages:**
[Paste any error messages or attach screenshots]

**Environment:**
- OS: [Windows / Mac / Linux]
- Python Version: [e.g., 3.10.5]
- Browser: [e.g., Chrome 118]
- Date/Time: [When the issue occurred]

**Workaround:**
[If you found a way to work around the issue]
```

### Example Bug Report

```
**Bug ID:** BUG-001
**Severity:** Medium
**Test Scenario:** Test Scenario 5 (Recent Trend Chart)
**Component:** Frontend

**Description:**
The trend chart does not display for Karachi, showing "Historical PM2.5 series is not available" message, but Lahore chart works fine.

**Steps to Reproduce:**
1. Open Streamlit app
2. Select City: "karachi"
3. Scroll to "Recent trend and tomorrow's forecast"
4. Observe the chart area

**Expected Result:**
Line chart showing Karachi's PM2.5 history with observed values and 24h rolling mean.

**Actual Result:**
Info message: "Historical PM2.5 series is not available in the saved dataset."

**Screenshots/Error Messages:**
[Screenshot of the empty chart area]

**Environment:**
- OS: Windows 11
- Python Version: 3.10.0
- Browser: Chrome 118
- Date/Time: 2026-11-12 23:15

**Workaround:**
None. Lahore chart works correctly.
```

### Where to Submit Bug Reports

- **GitHub Issues:** Create an issue in the project repository
- **Team Communication:** Report to the development team via [your communication channel]
- **Bug Tracking System:** Log in [your bug tracking tool]

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'src'"

**Cause:** Python cannot find the src module

**Solution:**
```bash
# Windows
set PYTHONPATH=%CD%

# Linux/Mac
export PYTHONPATH=$PWD
```

Then restart Streamlit.

#### Issue 2: "Port 8001 is already in use"

**Cause:** Another process is using port 8001

**Solutions:**
- Use a different port: `uvicorn app.api:app --reload --port 8002`
- Kill the process using the port:
  ```bash
  # Windows
  netstat -ano | findstr :8001
  taskkill /PID [PID] /F
  
  # Linux/Mac
  lsof -ti:8001 | xargs kill -9
  ```

#### Issue 3: "Model artifact missing"

**Cause:** The trained model file doesn't exist

**Solution:**
```bash
python -m src.train_final
```

Wait for training to complete, then restart the applications.

#### Issue 4: "OPENAQ_API_KEY not found"

**Cause:** Missing or invalid .env file

**Solution:**
1. Copy `.env.example` to `.env`
2. Add your OpenAQ API key
3. Or use offline mode: set Input source to "Saved snapshot only"

#### Issue 5: Predictions seem outdated

**Cause:** Cached data or stale snapshot

**Solutions:**
- Click "Refresh inputs" button
- Ensure live mode is selected
- Check staleness hours - if >24h, data is old
- Refresh the snapshot: `python -m src.live_data --refresh-fallback`

#### Issue 6: Charts not displaying

**Cause:** Missing data files or figures

**Solutions:**
- Check `data/processed/features.csv` exists
- Check `reports/figures/*.png` exist
- Run: `python -m src.error_analysis` to regenerate figures
- Check browser console for errors (F12)

#### Issue 7: FastAPI returns 502 errors

**Cause:** Backend cannot fetch live data

**Solutions:**
- Check internet connection
- Verify OPENAQ_API_KEY in backend environment
- Switch to offline mode temporarily
- Check backend logs for specific errors

#### Issue 8: Streamlit shows blank page

**Cause:** JavaScript or rendering error

**Solutions:**
- Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Clear browser cache
- Try a different browser
- Check browser console (F12) for errors
- Restart Streamlit

#### Issue 9: Predictions are identical for both cities

**Cause:** City parameter not being passed correctly

**Solutions:**
- Verify city selection in dropdown
- Check API request logs
- Test API directly: `http://127.0.0.1:8001/docs`
- Clear Streamlit cache

#### Issue 10: "Saved snapshot only" mode fails

**Cause:** Missing fallback file

**Solution:**
```bash
python -m src.live_data --refresh-fallback --offline
```

This generates `data/processed/live_fallback.csv`.

---

## Testing Best Practices

1. **Test Systematically:** Go through each test scenario in order
2. **Document Everything:** Note all observations, even if tests pass
3. **Take Screenshots:** Capture both successful and failed states
4. **Test Edge Cases:** Try unusual inputs and scenarios
5. **Test Both Cities:** Verify functionality for Lahore and Karachi
6. **Test Both Modes:** Check live and offline modes
7. **Check Consistency:** Predictions should be consistent for same inputs
8. **Monitor Logs:** Keep an eye on backend logs for errors
9. **Clear Cache:** Use Refresh button between major test changes
10. **Restart Services:** If something seems wrong, restart both services

---

## Expected Performance

### Response Times

| Operation | Expected Time | Acceptable Range |
|-----------|---------------|------------------|
| API Health Check | <100ms | <500ms |
| API Prediction (offline) | <200ms | <1s |
| API Prediction (live) | 2-10s | <30s |
| Streamlit page load | 1-3s | <10s |
| Chart rendering | <500ms | <2s |
| Cache refresh | 2-10s | <30s |

### Accuracy Expectations

Based on the test set evaluation:

| Metric | Expected Value | Interpretation |
|--------|---------------|----------------|
| MAE | ~13.31 ug/m3 | Average prediction error |
| RMSE | ~23.65 ug/m3 | Penalizes large errors more |
| R² | ~0.67 | Explains 67% of variance |
| AQI Accuracy | ~71.65% | Correct category ~72% of time |

### Per-City Performance

| City | MAE (ug/m3) | AQI Accuracy |
|------|-------------|--------------|
| Karachi | ~6.20 | ~82% |
| Lahore | ~20.42 | ~61% |

*Note: Lahore is harder to predict due to extreme smog events*

---

## Conclusion

This manual testing guide covers all major functionality of the PM2.5 Forecaster application. Complete the testing checklist, document any issues found, and provide feedback to the development team.

**Good luck with your testing!**

---

**Document Version:** 1.0  
**Last Updated:** November 12, 2026  
**Created By:** Kiro AI Assistant  
**Project:** Next-Day Air Quality (PM2.5) Forecaster

