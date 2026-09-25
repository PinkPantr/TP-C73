import h2o
import mlflow
import mlflow.h2o
import subprocess
from fastapi import FastAPI, HTTPException
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

# One training script at a time, with the current single API worker.
training_process = None
training_task = None


@app.post("/train/{task}", status_code=202)
async def start_training(task: str):
    global training_process, training_task

    if task == "regression":
        script = "src/train.py"
    elif task == "classification":
        script = "src/train_class.py"
    else:
        raise HTTPException(status_code=400, detail="Type d'entraînement inconnu.")

    if training_process is not None and training_process.poll() is None:
        raise HTTPException(status_code=409, detail="Un entraînement est déjà en cours.")

    training_process = subprocess.Popen(["python", "-u", script])
    training_task = task
    return {"status": "running", "task": training_task}


@app.get("/training/status")
async def training_status():
    if training_process is None:
        return {"status": "idle", "task": None}

    exit_code = training_process.poll()
    if exit_code is None:
        status = "running"
    elif exit_code == 0:
        status = "finished"
    else:
        status = "failed"

    return {"status": status, "task": training_task}
