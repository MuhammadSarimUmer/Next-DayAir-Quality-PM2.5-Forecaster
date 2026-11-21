"""
FastAPI REST API Backend (Task 16).

Provides an operational /predict endpoint with Pydantic input validation,
error handling, and serves the demo HTML frontend.

Run with:
    uvicorn src.api:app --reload --port 8000
"""

import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator

from src import config
from src.aqi import get_health_advice, calculate_aqi, get_aqi_category_info

# Initialize FastAPI app
app = FastAPI(
    title="Next-Day Air Quality (PM2.5) Forecaster API",
    description="24-hour forward PM2.5 and AQI advisory service for Karachi & Lahore",
    version="1.0.0"
)

# Enable CORS for local demo and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline at startup
MODEL_PATH = config.ROOT / "models" / "final_pipeline.joblib"
pipeline = None

if MODEL_PATH.exists():
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"✓ Model pipeline loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")


# -------------------------------------------------------------
# Input Validation Models
# -------------------------------------------------------------
class ForecastRequest(BaseModel):
    city: str = Field(
        ...,
        description="City name ('karachi' or 'lahore')",
        example="lahore"
    )
    pm25: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Current PM2.5 concentration in µg/m³ (0.0 to 1000.0)",
        example=85.0
    )
    temperature: float = Field(
        25.0,
        ge=-10.0,
        le=60.0,
        description="Temperature in °C (-10.0 to 60.0)",
        example=24.5
    )
    relative_humidity: float = Field(
        60.0,
        ge=0.0,
        le=100.0,
        description="Relative humidity in % (0.0 to 100.0)",
        example=68.0
    )
    surface_pressure: float = Field(
        1012.0,
        ge=900.0,
        le=1100.0,
        description="Surface pressure in hPa (900.0 to 1100.0)",
        example=1014.0
    )
    wind_speed: float = Field(
        3.0,
        ge=0.0,
        le=50.0,
        description="Wind speed in m/s (0.0 to 50.0)",
        example=2.1
    )
    wind_direction: float = Field(
        180.0,
        ge=0.0,
        le=360.0,
        description="Wind direction in degrees (0.0 to 360.0)",
        example=270.0
    )
    precipitation: float = Field(
        0.0,
        ge=0.0,
        le=300.0,
        description="Precipitation in mm (0.0 to 300.0)",
        example=0.0
    )
    cloud_cover: float = Field(
        20.0,
        ge=0.0,
        le=100.0,
        description="Cloud cover percentage (0.0 to 100.0)",
        example=25.0
    )
    month: Optional[int] = Field(
        None,
        ge=1,
        le=12,
        description="Month of year (1 to 12). Defaults to current month.",
        example=11
    )
    hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Hour of day (0 to 23). Defaults to current hour.",
        example=14
    )

    @validator("city")
    def validate_city(cls, v):
        v_clean = v.strip().lower()
        if v_clean not in ["karachi", "lahore"]:
            raise ValueError("City must be either 'karachi' or 'lahore'")
        return v_clean


class ForecastResponse(BaseModel):
    status: str
    city: str
    target_time: str
    predicted_pm25: float
    confidence_interval_95: List[float]
    aqi_index: int
    aqi_category: str
    hex_color: str
    category_probabilities: Dict[str, float]
    health_advice: Dict[str, Any]
    drivers: Dict[str, Any]


# -------------------------------------------------------------
# Helper: Feature Construction
# -------------------------------------------------------------
def build_feature_dataframe(req: ForecastRequest) -> pd.DataFrame:
    now = datetime.datetime.now()
    month = req.month if req.month is not None else now.month
    hour = req.hour if req.hour is not None else now.hour
    weekday = now.weekday()
    dt = datetime.datetime(2026, month, 15, hour, 0)
    doy = dt.timetuple().tm_yday
    is_winter = 1 if month in [11, 12, 1, 2] else 0

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    wind_rad = np.deg2rad(req.wind_direction)
    wind_dir_sin = np.sin(wind_rad)
    wind_dir_cos = np.cos(wind_rad)

    smog_season_intensity = float(np.exp(-0.5 * ((doy - 314) / 25.0) ** 2))
    stagnation_index = float((1.0 / max(0.5, req.wind_speed)) * (req.relative_humidity / 100.0))
    lahore_smog_risk = float((1.0 if req.city == "lahore" else 0.0) * smog_season_intensity * stagnation_index)

    pm = req.pm25
    pm25_lag_1h = pm * 0.98
    pm25_lag_3h = pm * 0.95
    pm25_lag_6h = pm * 0.92
    pm25_lag_12h = pm * 0.90
    pm25_lag_24h = pm * 0.95
    pm25_7d_lag = pm * 0.90

    pm25_roll_mean_6h = (pm + pm25_lag_1h + pm25_lag_3h + pm25_lag_6h) / 4.0
    pm25_roll_mean_24h = (pm + pm25_lag_1h + pm25_lag_6h + pm25_lag_12h + pm25_lag_24h) / 5.0
    pm25_roll_mean_72h = pm25_roll_mean_24h * 0.98
    pm25_diff_3h = pm - pm25_lag_3h
    pm25_roll_std_24h = max(2.0, pm * 0.15)
    pm25_roll_max_24h = pm * 1.15
    pm25_roll_min_24h = pm * 0.85

    pm25_change_6h = pm - pm25_lag_6h
    pm25_change_24h = pm - pm25_lag_24h
    pm25_rate_of_change = pm25_diff_3h / 3.0
    pm25_acceleration = pm25_diff_3h - (pm25_lag_3h - pm25_lag_6h)

    # Regime shift & CAMS features
    roll24 = pm25_roll_mean_24h
    roll72 = pm25_roll_mean_72h
    pm25_lag_excess_24h = roll24 - pm
    pm25_lag_excess_72h = roll72 - pm
    falling = max(0.0, -(pm - pm25_lag_6h))
    pm25_clearing_index = max(0.0, roll24 - pm) * falling

    cams_pm2_5 = max(5.0, pm * 0.75)
    cams_dust = 15.0 if req.city == "karachi" else 8.0
    cams_aod = float(0.5 + 0.5 * (pm / 150.0))
    pm25_cams_divergence = pm - cams_pm2_5
    cams_pm25_change_24h = 0.0

    pressure_change_24h = 0.5
    wind_speed_roll_mean_24h = max(1.0, req.wind_speed * 1.05)
    precip_sum_24h = req.precipitation * 2.0

    features = {
        "pm25_now": pm,
        "pm25_lag_1h": pm25_lag_1h,
        "pm25_lag_3h": pm25_lag_3h,
        "pm25_lag_6h": pm25_lag_6h,
        "pm25_lag_12h": pm25_lag_12h,
        "pm25_lag_24h": pm25_lag_24h,
        "pm25_roll_mean_6h": pm25_roll_mean_6h,
        "pm25_roll_mean_24h": pm25_roll_mean_24h,
        "pm25_roll_mean_72h": pm25_roll_mean_72h,
        "temperature_2m": req.temperature,
        "relative_humidity_2m": req.relative_humidity,
        "surface_pressure": req.surface_pressure,
        "wind_speed_10m": req.wind_speed,
        "precipitation": req.precipitation,
        "cloud_cover": req.cloud_cover,
        "hour": hour,
        "month": month,
        "weekday": weekday,
        "is_winter": is_winter,
        "city": req.city,
        "pm25_diff_3h": pm25_diff_3h,
        "pm25_roll_std_24h": pm25_roll_std_24h,
        "pm25_roll_max_24h": pm25_roll_max_24h,
        "pm25_change_6h": pm25_change_6h,
        "pm25_change_24h": pm25_change_24h,
        "pm25_rate_of_change": pm25_rate_of_change,
        "pm25_7d_lag": pm25_7d_lag,
        "pm25_roll_min_24h": pm25_roll_min_24h,
        "pm25_acceleration": pm25_acceleration,
        "stagnation_index": stagnation_index,
        "smog_season_intensity": smog_season_intensity,
        "lahore_smog_risk": lahore_smog_risk,
        "pm25_lag_excess_24h": pm25_lag_excess_24h,
        "pm25_lag_excess_72h": pm25_lag_excess_72h,
        "pm25_clearing_index": pm25_clearing_index,
        "pm25_cams_divergence": pm25_cams_divergence,
        "cams_pm25_change_24h": cams_pm25_change_24h,
        "cams_pm2_5": cams_pm2_5,
        "cams_dust": cams_dust,
        "cams_aod": cams_aod,
        "pressure_change_24h": pressure_change_24h,
        "wind_speed_roll_mean_24h": wind_speed_roll_mean_24h,
        "precip_sum_24h": precip_sum_24h,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
    }

    return pd.DataFrame([features])


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "version": "1.0.0",
        "target": "PM2.5 24 hours ahead (µg/m³)",
        "cities": ["karachi", "lahore"]
    }


@app.post("/predict", response_model=ForecastResponse)
def predict_next_day(req: ForecastRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded on server."
        )

    try:
        X_df = build_feature_dataframe(req)
        raw_pred = pipeline.predict(X_df)[0]
        pred_pm25 = max(1.0, float(raw_pred))

        # Empirical standard error per city
        sigma = 9.3 if req.city == "karachi" else 31.8
        ci_lower = max(0.0, round(pred_pm25 - 1.96 * sigma, 1))
        ci_upper = round(pred_pm25 + 1.96 * sigma, 1)

        # Category probabilities
        cat_boundaries = [0.0, 9.0, 35.4, 55.4, 125.4, 225.4, 1000.0]
        cat_names = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
        probs = {}
        for cname, low, high in zip(cat_names, cat_boundaries[:-1], cat_boundaries[1:]):
            p = norm.cdf(high, loc=pred_pm25, scale=sigma) - norm.cdf(low, loc=pred_pm25, scale=sigma)
            probs[cname] = round(max(0.0, float(p)), 4)
        
        # Normalize probabilities
        total_p = sum(probs.values())
        if total_p > 0:
            probs = {k: round(v / total_p, 4) for k, v in probs.items()}

        advice = get_health_advice(pred_pm25)
        target_dt = datetime.datetime.now() + datetime.timedelta(hours=24)

        return ForecastResponse(
            status="success",
            city=req.city.capitalize(),
            target_time=target_dt.strftime("%Y-%m-%d %H:00 PKT"),
            predicted_pm25=round(pred_pm25, 1),
            confidence_interval_95=[ci_lower, ci_upper],
            aqi_index=advice["aqi"],
            aqi_category=advice["category"],
            hex_color=advice["hex_color"],
            category_probabilities=probs,
            health_advice=advice,
            drivers={
                "stagnation_index": round(X_df["stagnation_index"].values[0], 2),
                "smog_season_intensity": round(X_df["smog_season_intensity"].values[0], 2),
                "lahore_smog_risk": round(X_df["lahore_smog_risk"].values[0], 2),
                "persistence_baseline": req.pm25
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


# HTML Demo Frontend
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Next-Day Air Quality Forecaster | Local Demo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f6f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .card { border-radius: 12px; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
        .badge-aqi { padding: 8px 16px; border-radius: 20px; font-weight: 700; color: #fff; display: inline-block; }
        .pred-box { border-left: 6px solid #007bff; padding: 20px; border-radius: 10px; background: #fff; }
    </style>
</head>
<body class="py-4">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card p-4 mb-4">
                    <h2 class="mb-1 text-primary">🌫️ Next-Day PM2.5 Forecaster</h2>
                    <p class="text-muted">CT-378 Machine Learning | Local FastAPI Demo (Karachi & Lahore)</p>
                    <hr>
                    <div class="row">
                        <div class="col-md-6">
                            <h5 class="mb-3">Prediction Inputs</h5>
                            <div class="mb-3">
                                <label class="form-label font-weight-bold">Target City</label>
                                <select id="city" class="form-select">
                                    <option value="karachi">Karachi (Coastal)</option>
                                    <option value="lahore" selected>Lahore (Continental)</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Current PM2.5 Concentration (µg/m³)</label>
                                <input type="number" id="pm25" class="form-control" value="85" min="0" max="1000">
                            </div>
                            <div class="row">
                                <div class="col-6 mb-3">
                                    <label class="form-label">Temperature (°C)</label>
                                    <input type="number" id="temp" class="form-control" value="24" min="-10" max="60">
                                </div>
                                <div class="col-6 mb-3">
                                    <label class="form-label">Relative Humidity (%)</label>
                                    <input type="number" id="rh" class="form-control" value="70" min="0" max="100">
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-6 mb-3">
                                    <label class="form-label">Wind Speed (m/s)</label>
                                    <input type="number" id="wind" class="form-control" value="2.0" step="0.5" min="0">
                                </div>
                                <div class="col-6 mb-3">
                                    <label class="form-label">Month</label>
                                    <input type="number" id="month" class="form-control" value="11" min="1" max="12">
                                </div>
                            </div>
                            <button class="btn btn-primary w-100 py-2 fw-bold" onclick="runForecast()">⚡ Generate 24-Hour Forecast</button>
                            <div id="error-msg" class="alert alert-danger mt-3 d-none"></div>
                        </div>

                        <div class="col-md-6">
                            <h5 class="mb-3">Prediction Output (t + 24h)</h5>
                            <div id="result-box" class="pred-box">
                                <p class="text-muted mb-1">Click "Generate 24-Hour Forecast" to run inference.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runForecast() {
            const errDiv = document.getElementById("error-msg");
            errDiv.classList.add("d-none");
            
            const payload = {
                city: document.getElementById("city").value,
                pm25: parseFloat(document.getElementById("pm25").value),
                temperature: parseFloat(document.getElementById("temp").value),
                relative_humidity: parseFloat(document.getElementById("rh").value),
                wind_speed: parseFloat(document.getElementById("wind").value),
                month: parseInt(document.getElementById("month").value)
            };

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                if (!res.ok) {
                    errDiv.innerText = data.detail || "Validation Error";
                    errDiv.classList.remove("d-none");
                    return;
                }

                document.getElementById("result-box").innerHTML = `
                    <div style="font-size:2rem; font-weight:800; color:${data.hex_color};">
                        ${data.predicted_pm25} µg/m³
                    </div>
                    <div class="mb-2">
                        <span class="badge" style="background-color:${data.hex_color}; font-size:1rem;">${data.aqi_category}</span>
                        <span class="text-muted ms-2">EPA AQI: ~${data.aqi_index}</span>
                    </div>
                    <p class="text-secondary small mb-2">95% Confidence Interval: [${data.confidence_interval_95[0]} – ${data.confidence_interval_95[1]}] µg/m³</p>
                    <hr>
                    <h6 class="fw-bold mb-1">Actionable Health Guidance:</h6>
                    <p class="mb-1 text-dark small"><b>Summary:</b> ${data.health_advice.summary}</p>
                    <p class="mb-1 text-dark small"><b>General Public:</b> ${data.health_advice.general_public}</p>
                    <p class="mb-0 text-dark small"><b>Sensitive Groups:</b> ${data.health_advice.sensitive_groups}</p>
                `;
            } catch (err) {
                errDiv.innerText = "Network or server connection failed.";
                errDiv.classList.remove("d-none");
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
