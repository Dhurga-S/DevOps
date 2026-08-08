FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY models ./models
COPY mlflow.db ./mlflow.db
COPY mlartifacts ./mlartifacts

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV MLFLOW_MODEL_NAME=CaliforniaHousingModel
ENV MLFLOW_MODEL_ALIAS=champion

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
