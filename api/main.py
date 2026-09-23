import h2o
import mlflow
import mlflow.h2o
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

class PredictionRequest(BaseModel):
    model_year: int
    make: str
    vehicle_class: str
    engine_size: float
    cylinders: int
    transmission: str
    fuel_type: str

app = FastAPI(title="Prediction de consommation de carburant")

h2o.connect(url="http://h2o:54321")
mlflow.set_tracking_uri("http://mlflow:5000")

model_uri = "models:/regression@champion"
model = mlflow.h2o.load_model(model_uri)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PredictionRequest):
    df = pd.DataFrame([{
        "Model year": data.model_year,
        "Make": data.make,
        "Vehicle class": data.vehicle_class,
        "Engine size (L)": data.engine_size,
        "Cylinders": data.cylinders,
        "Transmission": data.transmission,
        "Fuel type": data.fuel_type
    }])

    inputs = h2o.H2OFrame(df)
    predictions = model.predict(inputs)
    return {"prediction": float(predictions[0, 0])}

@app.post("/predict_class")
def predict_class(data: PredictionRequest):
    model_class= mlflow.h2o.load_model(
        "models:/classification@champion"
        )

    df = pd.DataFrame([{
        "Model year": data.model_year,
        "Make": data.make,
        "Vehicle class": data.vehicle_class,
        "Engine size (L)": data.engine_size,
        "Cylinders": data.cylinders,
        "Transmission": data.transmission,
        "Fuel type": data.fuel_type
    }])

    inputs = h2o.H2OFrame(df)
    predictions = model_class.predict(inputs)

    return{"classification": str(predictions[0, 0])}