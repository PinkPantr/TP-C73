import streamlit as st
import pandas as pd

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