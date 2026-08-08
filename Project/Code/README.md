# California Housing MLOps Capstone

End-to-end regression MLOps project using the California Housing dataset.

## Pipeline
Dataset → DVC → Training → MLflow → Model Registry → FastAPI → Docker → GitHub Actions

## Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is selected by lowest RMSE and registered in MLflow.

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
python src/prepare_data.py
dvc init
dvc add data/california_housing.csv
dvc repro
```

Start MLflow:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Train:
```bash
python src/train.py
```

Run API:
```bash
uvicorn src.app:app --reload --port 8000
```

Swagger: http://127.0.0.1:8000/docs

## API
POST `/predict`

Example:
```json
{
  "MedInc": 8.3,
  "HouseAge": 41.0,
  "AveRooms": 6.98,
  "AveBedrms": 1.02,
  "Population": 322.0,
  "AveOccup": 2.56,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

## DVC
```bash
dvc add data/california_housing.csv
git add data/california_housing.csv.dvc .gitignore
dvc repro
dvc status
```

## MLflow
```bash
mlflow ui
```
Compare runs, identify the lowest-RMSE model, and verify the registered model under **Models**.

## Docker
```bash
docker build -t california-housing-api .
docker run --rm -p 8000:8000 california-housing-api
```

## CI
GitHub Actions runs on push:
1. Checkout
2. Python setup
3. Install dependencies
4. Run tests
5. Build Docker image

## Submission screenshots
- MLflow experiment comparison
- Registered model
- DVC tracking/status
- GitHub Actions successful workflow
- FastAPI `/docs` or `/predict` response
