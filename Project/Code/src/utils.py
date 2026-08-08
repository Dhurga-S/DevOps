from pathlib import Path
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]
TARGET = "MedHouseVal"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "california_housing.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
