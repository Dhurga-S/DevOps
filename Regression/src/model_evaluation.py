import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Create metrics folder
os.makedirs("metrics", exist_ok=True)

# Load model
model = joblib.load("models/model.pkl")

# Load test data
X_test = pd.read_csv("data/features/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
    "R2": r2_score(y_test, y_pred)
}

print(json.dumps(metrics, indent=4))

# Save metrics
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model evaluation completed!")