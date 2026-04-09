import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

# Тестовые данные
sample_features = {
    "complaint_status": 4,
    "num_reassignments": 1,
    "has_photo_evidence": 1,
    "is_monsoon_season": 0,
    "resolution_days": 19,
    "has_gps_location": 1,
    "repeat_complainant": 0,
    "severity": 2,
    "ward_code": 11,
    "complaint_channel": 0
}

def test_root():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    """Тест health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    """Тест эндпоинта /predict"""
    response = client.post("/predict", json={
        "features": sample_features,
        "request_id": "test_001"
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert 0 <= response.json()["prediction"] <= 1

def test_predict_batch():
    """Тест batch эндпоинта"""
    response = client.post("/predict_batch", json={
        "requests": [
            {"features": sample_features, "request_id": "1"},
            {"features": sample_features, "request_id": "2"}
        ]
    })
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2

def test_predict_invalid():
    """Тест на некорректные данные"""
    response = client.post("/predict", json={
        "features": {},
        "request_id": "invalid"
    })
    assert response.status_code == 500  # или 422, зависит от валидации