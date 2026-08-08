from pathlib import Path
import json
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_data, save_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CaliforniaHousingModel")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("California-Housing-Regression")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    results = {}
    best = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_params({
                "model": name,
                "test_size": 0.2,
                "random_state": 42,
            })
            if name == "RandomForest":
                mlflow.log_params({"n_estimators": 200, "max_depth": 20})
            elif name == "GradientBoosting":
                mlflow.log_params({"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3})

            mlflow.log_metrics(metrics)
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.head(2),
            )

            results[name] = {"run_id": run.info.run_id, **metrics}
            if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
                best = {"name": name, "run_id": run.info.run_id, "metrics": metrics}

    # Register the best run's model using its MLflow artifact URI.
    best_uri = f"runs:/{best['run_id']}/model"
    try:
        registration = mlflow.register_model(best_uri, MODEL_NAME)
        version = registration.version
        # Move the registered version to a useful alias when supported.
        try:
            mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
        except Exception:
            pass
    except Exception as exc:
        version = None
        print(f"Model registration warning: {exc}")

    Path(ROOT / "models").mkdir(exist_ok=True)
    save_json(ROOT / "models" / "metrics.json", results)
    save_json(ROOT / "models" / "best_model.json", {
        "model_name": best["name"],
        "run_id": best["run_id"],
        "metrics": best["metrics"],
        "registered_model": MODEL_NAME,
        "registered_version": version,
        "tracking_uri": TRACKING_URI,
    })

    print(json.dumps({
        "best_model": best,
        "registered_model": MODEL_NAME,
        "version": version,
    }, indent=2))

if __name__ == "__main__":
    main()
