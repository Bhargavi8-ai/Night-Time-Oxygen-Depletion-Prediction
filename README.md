
# Night-Time Oxygen Depletion Prediction in Shrimp Ponds

## Project Overview
This project predicts the risk of night-time dissolved oxygen depletion
in shrimp aquaculture ponds using Machine Learning.

## Problem Statement
Dissolved oxygen levels may decrease during nighttime in shrimp ponds.
Low oxygen can cause shrimp stress and mortality. This project predicts
whether a pond may experience high oxygen depletion risk and provides
an alert for preventive aeration.

## Dataset
Synthetic pond-night data containing:
- Pond area
- Stocking density
- Dissolved oxygen at 8 PM
- Water temperature
- Feed quantity
- Aerator operating hours
- Rainfall

Two dataset versions:
- V1: Normal weather
- V2: Rainy weather

## Feature Engineering
- Feed per unit biomass
- Aeration hours per pond area

## Machine Learning Models
- Logistic Regression
- Random Forest

## Evaluation Metrics
- Accuracy
- High Risk Precision
- High Risk Recall
- High Risk F1-score

## Technologies
Python, Pandas, NumPy, Scikit-learn, MLflow, FastAPI,
Joblib, Pytest and GitHub Actions.

## API Endpoints
- GET /health
- POST /predict

## Run the project

Install dependencies:

    pip install -r requirements.txt

Start the API:

    uvicorn app:app --reload

API documentation:

    http://127.0.0.1:8000/docs

## Disclaimer
This is an academic prototype using synthetic data.
Real pond sensor data and field validation are required
before real-world deployment.
