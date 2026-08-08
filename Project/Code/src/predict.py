import os
import mlflow
import mlflow.sklearn

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CaliforniaHousingModel")
MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION", "1")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

def load_registered_model():
    # Prefer the champion alias; fall back to an explicit version.
    try:
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except Exception:
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
