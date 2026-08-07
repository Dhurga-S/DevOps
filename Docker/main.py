from pathlib import Path
from typing import Literal

import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "ca_housing_model.joblib"
METRICS_PATH = BASE_DIR.parent / "model" / "metrics.json"

app = FastAPI(
    title="California Housing Price Predictor",
    description="Predicts median house value for a California census block "
                 "group using a GradientBoostingRegressor trained on the "
                 "1990 California census housing dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model + metrics once at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

OCEAN_PROXIMITY_VALUES = tuple(METRICS["ocean_proximity_categories"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class HouseFeatures(BaseModel):
    longitude: float = Field(..., ge=-125, le=-113, example=-122.23)
    latitude: float = Field(..., ge=32, le=42.5, example=37.88)
    housing_median_age: float = Field(..., ge=0, le=60, example=41)
    total_rooms: float = Field(..., gt=0, example=880)
    total_bedrooms: float = Field(..., gt=0, example=129)
    population: float = Field(..., gt=0, example=322)
    households: float = Field(..., gt=0, example=126)
    median_income: float = Field(
        ..., gt=0, example=8.3252,
        description="Median income in the block group, in tens of thousands "
                     "of USD (e.g. 8.3252 = $83,252).",
    )
    ocean_proximity: Literal[
        "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"
    ] = Field(..., example="NEAR BAY")


class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if features.ocean_proximity not in OCEAN_PROXIMITY_VALUES:
        raise HTTPException(status_code=400, detail="Unknown ocean_proximity category.")

    row = pd.DataFrame([features.dict()])
    try:
        prediction = float(model.predict(row)[0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    prediction = max(prediction, 0.0)
    return PredictionResponse(
        predicted_price=round(prediction, 2),
        formatted_price=f"${prediction:,.0f}",
    )


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")
