import h2o
import mlflow
import mlflow.h2o

from preprocessing import split_data, prepare_data
from h2o.automl import H2OAutoML

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

target_classification = "Classe smog"

train_h2o[target_classification] = train_h2o[target_classification].asfactor()
validation_h2o[target_classification] = validation_h2o[target_classification].asfactor()

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("C73-Classification")

for max_models in [5, 10, 20]:
    for balance_classes in [True, False]:

        with mlflow.start_run(
            run_name=f"automl_{max_models}_balance_{balance_classes}"
        ) as run:
            automl = H2OAutoML(
                max_models=max_models,
                seed=42,
                sort_metric="logloss",
                balance_classes=balance_classes,
                project_name="automl_" + run.info.run_id
            )

            automl.train(
                x=features,
                y=target_classification,
                training_frame=train_h2o,
                leaderboard_frame=validation_h2o
            )

            mlflow.log_param("model", "H2O AutoML")
            mlflow.log_param("max_models", max_models)
            mlflow.log_param("seed", 42)
            mlflow.log_param("balance_classes", balance_classes)
            mlflow.log_param("sort_metric", "logloss")

            scores = automl.leader.model_performance(validation_h2o)
            mlflow.log_metric("validation_logloss", scores.logloss())
            mlflow.h2o.log_model(automl.leader, artifact_path="model")

            print(automl.leaderboard)
