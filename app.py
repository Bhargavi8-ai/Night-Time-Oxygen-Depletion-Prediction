
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Night-Time Oxygen Depletion Prediction API")

# Load the trained model
model = joblib.load("models/shrimp_model.pkl")

feature_columns = [
    "pond_area",
    "stocking_density",
    "oxygen_8pm",
    "water_temperature",
    "feed_quantity",
    "aerator_hours",
    "rainfall",
    "feed_per_unit_biomass",
    "aeration_hours_per_area"
]

class PondInput(BaseModel):
    pond_area: float = Field(gt=0)
    stocking_density: float = Field(gt=0)
    oxygen_8pm: float = Field(gt=0)
    water_temperature: float
    feed_quantity: float = Field(ge=0)
    aerator_hours: float = Field(ge=0)
    rainfall: float = Field(ge=0)

@app.get("/")
def home():
    return {
        "message": "Night-Time Oxygen Depletion Prediction API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
def predict(data: PondInput):

    # Calculate engineered features
    pond_area_sq_m = data.pond_area * 4046.86

    estimated_biomass = (
        pond_area_sq_m *
        data.stocking_density *
        15 / 1000
    )

    feed_per_unit_biomass = (
        data.feed_quantity / estimated_biomass
        if estimated_biomass > 0 else 0
    )

    aeration_hours_per_area = (
        data.aerator_hours / data.pond_area
    )

    features = [
        data.pond_area,
        data.stocking_density,
        data.oxygen_8pm,
        data.water_temperature,
        data.feed_quantity,
        data.aerator_hours,
        data.rainfall,
        feed_per_unit_biomass,
        aeration_hours_per_area
    ]

    input_array = pd.DataFrame([features], columns=feature_columns)

    prediction = model.predict(input_array)[0]

    # Obtain High-risk probability
    probabilities = model.predict_proba(input_array)[0]
    classes = list(model.classes_)
    high_index = classes.index("High")
    high_probability = float(probabilities[high_index])

    if prediction == "High":
        message = "High oxygen depletion risk. Additional nighttime aeration may be required."
    else:
        message = "Normal oxygen risk predicted. Continue monitoring dissolved oxygen."

    return {
        "predicted_risk": str(prediction),
        "high_risk_probability": round(high_probability, 4),
        "additional_aeration_required": prediction == "High",
        "message": message
    }
