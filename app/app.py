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
