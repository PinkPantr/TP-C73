import h2o
import mlflow
import mlflow.h2o

from preprocessing import split_data, prepare_data

csv_path = "data/my2015-2024-fuel-consumption-ratings.csv"

df = prepare_data(csv_path)
train_df, validation_df, test_df = split_data(df)

mlflow.set_tracking_uri("http://mlflow:5000")
h2o.connect(url="http://h2o:54321")

test_h2o = h2o.H2OFrame(test_df)
test_h2o["Classe smog"] = test_h2o["Classe smog"].asfactor()

print ("Lignes du test final:", test_h2o.nrows)

regression = mlflow.h2o.load_model("models:/regression@champion")
classification = mlflow.h2o.load_model("models:/classification@champion")

scores_regression = regression.model_performance(test_h2o)

print("MAE test :", scores_regression.mae())
print("RMSE test :", scores_regression.rmse())
print("R2 test :", scores_regression.r2())

scores_classification = classification.model_performance(test_h2o)

print("Logloss test :", scores_classification.logloss())
print("Erreur moyenne par classe :", scores_classification.mean_per_class_error())
print(scores_classification.confusion_matrix())

mlflow.set_experiment("C73-Evaluation-finale")

with mlflow.start_run(run_name="test-final-champions"):
    mlflow.log_param("lignes_test", test_h2o.nrows)
    mlflow.log_param("modele_regression", regression.model_id)
    mlflow.log_param("modele_classification", classification.model_id)

    mlflow.log_metric("test_mae", scores_regression.mae())
    mlflow.log_metric("test_rmse", scores_regression.rmse())
    mlflow.log_metric("test_r2", scores_regression.r2())

    mlflow.log_metric("test_logloss", scores_classification.logloss())
    mlflow.log_metric(
        "test_mean_per_class_error",
        scores_classification.mean_per_class_error()
    )

    mlflow.log_text(
        str(scores_classification.confusion_matrix()),
        "matrice_confusion.txt"
    )
