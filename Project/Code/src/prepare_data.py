from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "california_housing.csv"

def main():
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
    print(df.head())

if __name__ == "__main__":
    main()
