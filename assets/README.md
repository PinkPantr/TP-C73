# Captures du TP C73

Captures des applications prises le 24 septembre 2026.

| Capture | Contenu |
|---|---|
| [01-streamlit-donnees.png](01-streamlit-donnees.png) | Aperçu des données dans Streamlit. |
| [02-streamlit-predictions.png](02-streamlit-predictions.png) | Formulaire, deux prédictions et description des modèles. |
| [03-fastapi-routes.png](03-fastapi-routes.png) | Routes de l’API dans Swagger. |
| [04-fastapi-regression.png](04-fastapi-regression.png) | Requête de régression et réponse HTTP 200. |
| [05-fastapi-classification.png](05-fastapi-classification.png) | Requête de classification et réponse HTTP 200. |
| [06-mlflow-runs-regression.png](06-mlflow-runs-regression.png) | Vue des runs de régression et de leurs paramètres. |
| [07-mlflow-comparaison-regression.png](07-mlflow-comparaison-regression.png) | Comparaison de cinq configurations, triées par RMSE de validation. |
| [08-mlflow-comparaison-classification.png](08-mlflow-comparaison-classification.png) | Comparaison des recherches AutoML avec max_models et balance_classes. |
| [09-mlflow-champion-regression.png](09-mlflow-champion-regression.png) | Modèle regression, version 2, avec l’alias champion. |
| [10-mlflow-champion-classification.png](10-mlflow-champion-classification.png) | Modèle classification, version 1, avec l’alias champion. |
| [11-mlflow-evaluation-finale.png](11-mlflow-evaluation-finale.png) | Run d’évaluation des deux champions sur le test final. |
| [12-mlflow-metriques-test.png](12-mlflow-metriques-test.png) | Graphiques des métriques du test final. |
| [13-mlflow-matrice-confusion.png](13-mlflow-matrice-confusion.png) | Matrice de confusion enregistrée comme artefact. |
| [14-minio-bucket.png](14-minio-bucket.png) | Bucket mlflow dans MinIO. |
| [15-minio-modele.png](15-minio-modele.png) | Fichiers du champion de régression stockés dans MinIO. |
| [16-pgadmin-metriques.png](16-pgadmin-metriques.png) | Métriques stockées dans PostgreSQL, affichées avec pgAdmin. |

La capture Streamlit et les réponses de l’API utilisent le même exemple : année 2024, marque Acura, catégorie Full-size, cylindrée 1,5 L, 4 cylindres, transmission AV7 et carburant Z. Résultats : environ 7,83 L/100 km et Note elevee. Cet exemple montre le fonctionnement de l’application ; les performances globales sont dans les captures du test final.

La comparaison AutoML montre aussi un ancien run échoué. Il ne fait pas partie des six recherches terminées.

Les fichiers image.png, image (1).png et image (2).png sont les captures déjà présentes dans ce dossier.
