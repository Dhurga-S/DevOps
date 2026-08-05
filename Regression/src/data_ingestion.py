import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Create raw data directory
os.makedirs("data/raw", exist_ok=True)

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = housing.frame

# Save dataset
df.to_csv("data/raw/california_housing.csv", index=False)

print("Dataset saved successfully!")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())