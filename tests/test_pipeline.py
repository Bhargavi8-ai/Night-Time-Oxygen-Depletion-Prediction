
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_dataset_size():
    df = pd.read_csv("data/features.csv")
    assert len(df) >= 250

def test_feature_columns():
    df = pd.read_csv("data/features.csv")
    assert "feed_per_unit_biomass" in df.columns
    assert "aeration_hours_per_area" in df.columns

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    sample = {
        "pond_area": 1.5,
        "stocking_density": 250,
        "oxygen_8pm": 5.2,
        "water_temperature": 31,
        "feed_quantity": 75,
        "aerator_hours": 4,
        "rainfall": 60
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_risk"] in ["High", "Normal"]
    assert "high_risk_probability" in result
    assert "additional_aeration_required" in result
