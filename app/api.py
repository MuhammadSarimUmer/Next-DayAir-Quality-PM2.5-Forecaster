"""FastAPI service for the PM2.5 forecaster (Task 17).

Run:
    uvicorn app.api:app --reload --port 8000

Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src import config
from src.aqi import get_health_advice
from src.features import FEATURE_COLS
from src.predict import forecast, load_model, model_available, predict_pm25

app = FastAPI(
    title="Next-Day PM2.5 Forecast API",
    description="24-hour PM2.5 forecasts and EPA AQI advice for Karachi and Lahore.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

METADATA_PATH = config.ROOT / "models" / "model_metadata.json"


@lru_cache(maxsize=1)
def get_model():
    if not model_available():
        raise HTTPException(
            status_code=503,
            detail="Model artifact missing. Run `python -m src.train_final`.",
        )
    return load_model()


class PredictRequest(BaseModel):
    city: str = Field(..., examples=["lahore"])
    offline: bool = Field(
        False, description="Use the saved snapshot instead of live APIs."
    )
    prefer_offline: bool = Field(False, description="Try the saved snapshot first.")
    postprocess: bool = Field(
        False, description="Apply the experimental smog-recovery correction."
    )
    features: dict[str, Any] | None = Field(
        default=None,
        description="Optional complete feature row (all 47 features). Skips live fetch.",
    )

    @field_validator("city")
    @classmethod
    def valid_city(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in config.CITIES:
            raise ValueError(f"city must be one of {sorted(config.CITIES)}")
        return value


class ManualRequest(BaseModel):
    pm25: float = Field(..., ge=0, le=2000, examples=[85.0])


class ForecastResponse(BaseModel):
    city: str
    source: str
    feature_timestamp: str
    observed_at: str
    observed_pm25: float
    predicted_pm25: float
    aqi: int
    category: str
    hex_color: str
    summary: str
    general_public: str
    sensitive_groups: str
    actions: list[str]
    stale_hours: float
    notes: str


class ManualResponse(BaseModel):
    pm25: float
    aqi: int
    category: str
    hex_color: str
    summary: str
    general_public: str
    sensitive_groups: str
    actions: list[str]


def _features_from_dict(values: dict[str, Any], city: str) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in values]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"features is missing {len(missing)} columns: {missing[:8]}",
        )

    row: dict[str, Any] = {}
    invalid: list[str] = []
    for column in FEATURE_COLS:
        if column == "city":
            value = str(values.get("city", city)).lower().strip()
            if value not in config.CITIES:
                invalid.append(column)
                continue
            row[column] = value
            continue
        try:
            number = float(values[column])
        except (TypeError, ValueError):
            invalid.append(column)
            continue
        if not math.isfinite(number):
            invalid.append(column)
            continue
        row[column] = number

    if invalid:
        raise HTTPException(
            status_code=422,
            detail=(
                "features has invalid values (expected finite numbers, and city in "
                f"{sorted(config.CITIES)}): {invalid[:8]}"
            ),
        )
    return pd.DataFrame([row])


@app.get("/")
def root():
    return {
        "service": "next-day-pm25-forecaster",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "manual_aqi": "POST /aqi",
    }


@app.get("/health")
def health():
    meta: dict[str, Any] = {}
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "status": "ok" if model_available() else "model_missing",
        "model_loaded": model_available(),
        "model_path": str(config.ROOT / "models" / "final_pipeline.joblib"),
        "n_features": len(FEATURE_COLS),
        "test_metrics": meta.get("test_metrics_raw", {}),
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(request: PredictRequest):
    model = get_model()

    if request.features is not None:
        features = _features_from_dict(request.features, request.city)
        predicted = predict_pm25(model, features, postprocess=request.postprocess)
        advice = get_health_advice(predicted)
        return {
            "city": request.city.title(),
            "source": "request_features",
            "feature_timestamp": "n/a",
            "observed_at": "n/a",
            "observed_pm25": float(features.iloc[0]["pm25_now"]),
            "predicted_pm25": round(predicted, 2),
            "stale_hours": 0.0,
            "notes": "Prediction computed from caller-supplied features.",
            **advice,
        }

    try:
        result = forecast(
            request.city,
            model=model,
            offline=request.offline,
            prefer_offline=request.prefer_offline,
            postprocess=request.postprocess,
        )
    except Exception as exc:  # noqa: BLE001 - return a clean API error
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result


@app.post("/aqi", response_model=ManualResponse)
def manual_aqi(request: ManualRequest):
    return get_health_advice(request.pm25)


@app.get("/metadata")
def metadata():
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="model_metadata.json not found")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
