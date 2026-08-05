import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/california_housing.csv")

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Data preprocessing completed!")