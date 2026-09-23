import h2o
import mlflow
import mlflow.h2o

from preprocessing import split_data, prepare_data
from h2o.estimators import H2ORandomForestEstimator

csv_path = "data/my2015-2024-fuel-consumption-ratings.csv"

df= prepare_data(csv_path)
train_df, validation_df, test_df = split_data(df)

print("Entrainement: ", len(train_df))
print("Validation: ", len(validation_df))
print("Test final: ", len(test_df))

h2o.connect(url="http://h2o:54321")

train_h2o = h2o.H2OFrame(train_df)
validation_h2o = h2o.H2OFrame(validation_df)

print("Entrainement H2O: ", train_h2o.shape)
print("Validation H2O: ", validation_h2o.shape)

features = [
    "Model year",
    "Make",
    "Vehicle class",
    "Engine size (L)",
    "Cylinders",
    "Transmission",
    "Fuel type"
]
target_regression = "Combined (L/100 km)"

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("C73-Regression")

with mlflow.start_run(run_name="forest_50_depth_10"):
    model = H2ORandomForestEstimator(
        ntrees=50,
        max_depth=10,
        seed=42
    )
    
    model.train(
        x=features,
        y=target_regression,
        training_frame=train_h2o,
        validation_frame=validation_h2o
        )

    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("ntrees", 50)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("seed", 42)

    scores = model.model_performance(valid=True)

    mlflow.log_metric("validation_mae", scores.mae())
    mlflow.log_metric("validation_rmse", scores.rmse())
    mlflow.log_metric("validation_r2", scores.r2())

    mlflow.h2o.log_model(model, artifact_path="model")

