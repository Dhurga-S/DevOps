import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create features directory
os.makedirs("data/features", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Standard Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaled features
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    "data/features/X_train_scaled.csv",
    index=False
)

pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    "data/features/X_test_scaled.csv",
    index=False
)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("Feature engineering completed!")