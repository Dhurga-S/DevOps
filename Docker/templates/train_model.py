"""
Train a GradientBoostingRegressor on the California housing dataset
(the classic 1990 census dataset: longitude, latitude, housing_median_age,
total_rooms, total_bedrooms, population, households, median_income,
ocean_proximity -> median_house_value).

Run:
    python train_model.py

Produces:
    ca_housing_model.joblib   (sklearn Pipeline: preprocessing + model)
    metrics.json              (holdout RMSE / MAE / R2)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
TARGET = "median_house_value"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(HERE / "housing.csv")
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def eval_metrics(actual, pred):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test, preds)
    print(f"Holdout -> RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  R2: {r2:.4f}")

    joblib.dump(pipeline, HERE / "ca_housing_model.joblib")

    with open(HERE / "metrics.json", "w") as f:
        json.dump(
            {
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "ocean_proximity_categories": sorted(
                    df["ocean_proximity"].dropna().unique().tolist()
                ),
            },
            f,
            indent=2,
        )

    print(f"Saved model to {HERE / 'ca_housing_model.joblib'}")
    print(f"Saved metrics to {HERE / 'metrics.json'}")


if __name__ == "__main__":
    main()
