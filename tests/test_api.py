"""Contract tests for the FastAPI service (Task 17)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src import config
from src.features import FEATURE_COLS

client = TestClient(app)


def _feature_row() -> dict[str, float | str]:
    path = config.PROCESSED_DIR / "live_fallback.csv"
    if not path.exists():
        pytest.skip("live_fallback.csv not available")
    row = pd.read_csv(path).iloc[0]
    features: dict[str, float | str] = {
        name: float(row[name]) for name in FEATURE_COLS if name != "city"
    }
    features["city"] = "lahore"
    return features


def test_health_reports_model_and_metrics():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == len(FEATURE_COLS) == 47
    assert "mae" in body["test_metrics"]


def test_predict_offline_returns_full_forecast():
    response = client.post("/predict", json={"city": "lahore", "offline": True})
    assert response.status_code == 200
    body = response.json()
    for field in [
        "city",
        "source",
        "predicted_pm25",
        "observed_pm25",
        "aqi",
        "category",
        "hex_color",
        "summary",
        "general_public",
        "sensitive_groups",
        "actions",
        "stale_hours",
    ]:
        assert field in body
    assert body["city"] == "Lahore"
    assert body["source"] == "offline"
    assert body["predicted_pm25"] >= 0
    assert body["aqi"] >= 0
    assert isinstance(body["actions"], list) and body["actions"]


def test_predict_rejects_unknown_city():
    response = client.post("/predict", json={"city": "islamabad", "offline": True})
    assert response.status_code == 422


def test_predict_rejects_incomplete_features():
    response = client.post(
        "/predict",
        json={"city": "lahore", "features": {"pm25_now": 100.0}},
    )
    assert response.status_code == 422
    assert "missing" in response.json()["detail"].lower()


def test_predict_accepts_caller_supplied_features():
    response = client.post(
        "/predict",
        json={"city": "lahore", "features": _feature_row()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "request_features"
    assert body["predicted_pm25"] >= 0


def test_predict_rejects_non_numeric_feature():
    features = _feature_row()
    features["pm25_now"] = "not-a-number"
    response = client.post(
        "/predict",
        json={"city": "lahore", "features": features},
    )
    assert response.status_code == 422
    assert "invalid values" in response.json()["detail"].lower()


def test_predict_rejects_non_finite_feature():
    features = _feature_row()
    features["wind_speed_10m"] = "NaN"
    response = client.post(
        "/predict",
        json={"city": "lahore", "features": features},
    )
    assert response.status_code == 422


def test_predict_rejects_invalid_feature_city():
    features = _feature_row()
    features["city"] = "islamabad"
    response = client.post(
        "/predict",
        json={"city": "lahore", "features": features},
    )
    assert response.status_code == 422


def test_manual_aqi_endpoint_matches_breakpoints():
    response = client.post("/aqi", json={"pm25": 35.5})
    assert response.status_code == 200
    body = response.json()
    assert body["aqi"] == 101
    assert body["category"] == "Unhealthy for Sensitive Groups"

    bad = client.post("/aqi", json={"pm25": -1})
    assert bad.status_code == 422
