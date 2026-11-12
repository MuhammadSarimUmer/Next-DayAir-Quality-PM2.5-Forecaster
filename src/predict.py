"""Shared inference helpers used by Streamlit and FastAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import config
from src.aqi import get_health_advice
from src.live_data import get_live_features
from src.postprocess import apply_clearing_blend

MODEL_PATH = config.ROOT / "models" / "final_pipeline.joblib"


def model_available(path: Path = MODEL_PATH) -> bool:
    return path.exists()


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.train_final` first."
        )
    return joblib.load(path)


def predict_pm25(model, features: pd.DataFrame, *, postprocess: bool = False) -> float:
    """Predict 24h-ahead PM2.5 from a one-row (or many-row) feature frame."""
    raw = model.predict(features)
    if postprocess:
        raw = apply_clearing_blend(raw, features)
    return float(max(0.0, raw[0]))


def forecast(
    city: str,
    *,
    model=None,
    offline: bool = False,
    prefer_offline: bool = False,
    postprocess: bool = False,
    live: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full user-facing forecast: live inputs -> prediction -> AQI + advice."""
    model = model or load_model()
    live = live or get_live_features(city, offline=offline, prefer_offline=prefer_offline)
    predicted = predict_pm25(model, live["features"], postprocess=postprocess)
    advice = get_health_advice(predicted)

    return {
        "city": live["city"].title(),
        "source": live["source"],
        "feature_timestamp": str(live["feature_timestamp"]),
        "observed_at": str(live["observed_at"]),
        "observed_pm25": round(float(live["observed_pm25"]), 2),
        "predicted_pm25": round(predicted, 2),
        "stale_hours": live["stale_hours"],
        "notes": live["notes"],
        **advice,
    }
