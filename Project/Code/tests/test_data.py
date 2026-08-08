from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "california_housing.csv"

def test_dataset_has_expected_columns():
    if not DATA.exists():
        return
    df = pd.read_csv(DATA)
    expected = {
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"
    }
    assert expected.issubset(df.columns)
    assert len(df) > 1000
