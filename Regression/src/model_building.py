import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

n_estimators = params["model"]["n_estimators"]
random_state = params["model"]["random_state"]

# Create models folder
os.makedirs("models", exist_ok=True)

# Load training data
X_train = pd.read_csv("data/features/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Train model
model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=random_state
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained successfully!")