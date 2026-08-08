from pathlib import Path
import os
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .predict import load_registered_model

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="California Housing Price API", version="1.0.0")
model = None

class HousingInput(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def root():
    return {"message": "California Housing Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.on_event("startup")
def startup():
    global model
    try:
        model = load_registered_model()
    except Exception as exc:
        print(f"Model loading failed at startup: {exc}")

@app.post("/predict")
def predict(payload: HousingInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Registered MLflow model is not available. Train/register the model first."
        )
    values = [[
        payload.MedInc, payload.HouseAge, payload.AveRooms, payload.AveBedrms,
        payload.Population, payload.AveOccup, payload.Latitude, payload.Longitude
    ]]
    prediction = float(model.predict(values)[0])
    return {"prediction": prediction, "unit": "hundreds of thousands of USD"}
