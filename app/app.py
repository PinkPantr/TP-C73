import streamlit as st
import pandas as pd
import requests

st.title("Véhicules du Canada")
st.subheader("Consommation de carburant — 2015 à 2024")

fichier = "data/my2015-2024-fuel-consumption-ratings.csv"
df = pd.read_csv(fichier)

st.subheader("Dimensions du fichier")
col1, col2 = st.columns(2)
col1.metric("Lignes", df.shape[0])
col2.metric("Colonnes", df.shape[1])

st.subheader("Aperçu des données")
st.dataframe(df.head(10))

st.subheader("Prédire la consommation et la classe smog")
col1, col2, col3 = st.columns(3)

with col1:
    model_year = st.number_input(
        "Année modèle", min_value=2017, max_value=2024, value=2023
    )
    make = st.selectbox("Marque", sorted(df["Make"].unique()))
    vehicle_class = st.selectbox(
        "Catégorie", sorted(df["Vehicle class"].unique())
    )

with col2:
    engine_size = st.number_input(
        "Cylindrée (L)", min_value=0.1, value=1.5, step=0.1
    )
    cylinders = st.number_input("Cylindres", min_value=1, value=4)

with col3:
    transmission = st.selectbox(
        "Transmission", sorted(df["Transmission"].unique())
    )
    fuel_type = st.selectbox(
        "Type de carburant", sorted(df["Fuel type"].unique())
    )

if st.button("Prédire"):
    data = {
        "model_year": model_year,
        "make": make,
        "vehicle_class": vehicle_class,
        "engine_size": engine_size,
        "cylinders": cylinders,
        "transmission": transmission,
        "fuel_type": fuel_type
    }

    try:
        response = requests.post(
            "http://backend:8000/predict",
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        response_class = requests.post(
            "http://backend:8000/predict_class",
            json=data,
            timeout=30
        )
        response_class.raise_for_status()
        result_class = response_class.json()

        col_regression, col_classification = st.columns(2)

        with col_regression:
            st.metric(
                "Consommation prédite (L/100 km)",
                round(result["prediction"], 2)
            )
            st.write(
                "Régression — Random Forest : plusieurs arbres donnent une "
                "estimation, puis leurs prédictions sont moyennées pour "
                "estimer la consommation de carburant."
            )

        with col_classification:
            st.metric(
                "Classe smog prédite",
                result_class["classification"]
            )
            st.write(
                "Classification — XGBoost, sélectionné par H2O AutoML : "
                "le modèle prédit une note faible, moyenne ou élevée. "
                "Une note élevée correspond à moins de polluants "
                "responsables du smog."
            )

    except requests.exceptions.RequestException as e:
        st.error(f"La requête a échoué : {e}")


with st.expander("Entraîner les modèles"):
    st.write(
        "Régression : 36 configurations Random Forest. "
        "Classification : 6 recherches AutoML. "
        "Les résultats sont enregistrés dans MLflow."
    )
    st.caption(
        "Un seul entraînement à la fois. Les champions actuels sont conservés. "
        "Gardez les services démarrés pendant l'entraînement."
    )

    try:
        status_response = requests.get(
            "http://backend:8000/training/status", timeout=10
        )
        status_response.raise_for_status()
        training = status_response.json()
        running = training["status"] == "running"

        if running:
            st.info(f"Entraînement en cours : {training['task']}.")
        elif training["status"] == "finished":
            st.success(f"Entraînement terminé : {training['task']}. Consultez MLflow.")
        elif training["status"] == "failed":
            st.error(
                "L'entraînement a échoué. Consultez les logs du backend "
                "pour voir l'erreur."
            )
        else:
            st.write("Aucun entraînement lancé depuis le démarrage de l'API.")

        col1, col2 = st.columns(2)
        regression_clicked = col1.button("Entraîner la régression", disabled=running)
        classification_clicked = col2.button("Entraîner la classification", disabled=running)
        st.button("Actualiser le statut")

        if regression_clicked or classification_clicked:
            if regression_clicked:
                task = "regression"
            else:
                task = "classification"

            training_response = requests.post(
                f"http://backend:8000/train/{task}", timeout=10
            )
            if training_response.status_code == 409:
                st.warning("Un entraînement est déjà en cours. Actualisez le statut.")
            else:
                training_response.raise_for_status()
                st.rerun()

        st.markdown("[Voir les expériences dans MLflow](http://localhost:5000)")

    except requests.exceptions.RequestException:
        st.error("Impossible de joindre le service d'entraînement. Vérifiez FastAPI.")
