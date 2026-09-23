import h2o
import mlflow
import mlflow.h2o
from preprocessing import prepare_data


mlflow.set_tracking_uri("http://mlflow:5000")
h2o.connect(url="http://h2o:54321")

model_uri = "models:/regression@champion"
model = mlflow.h2o.load_model(model_uri)

print("Modele charge: ", model.model_id)

df = prepare_data("data/my2015-2024-fuel-consumption-ratings.csv")
sample = df.loc[df["Model year"] == 2023].head(3)

features = [
    "Model year",
    "Make",
    "Vehicle class",
    "Engine size (L)",
    "Cylinders",
    "Transmission",
    "Fuel type"
]

inputs = h2o.H2OFrame(sample[features])
predictions = model.predict(inputs)

print(predictions)