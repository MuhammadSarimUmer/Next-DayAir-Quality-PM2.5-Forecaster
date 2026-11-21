import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "karachi" in data["cities"]
    assert "lahore" in data["cities"]


def test_predict_endpoint_valid():
    payload = {
        "city": "lahore",
        "pm25": 85.0,
        "temperature": 22.0,
        "relative_humidity": 70.0,
        "wind_speed": 1.5,
        "month": 11,
        "hour": 14
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["city"] == "Lahore"
    assert "predicted_pm25" in data
    assert "confidence_interval_95" in data
    assert len(data["confidence_interval_95"]) == 2
    assert "aqi_index" in data
    assert "aqi_category" in data
    assert "health_advice" in data


def test_predict_endpoint_invalid_city():
    payload = {
        "city": "islamabad",  # Invalid city
        "pm25": 50.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_negative_pm25():
    payload = {
        "city": "karachi",
        "pm25": -10.0  # Invalid PM2.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
