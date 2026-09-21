import pandas as pd

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    df = df.loc[df["Model year"] >= 2017].copy()

    df["Classe smog"] = pd.cut(
        df["Smog rating"],
        bins=[0, 3, 6, 10],
        labels=["Note faible", "Note moyenne", "Note elevee"]
    )

    return df

def split_data(df):
    train_df = df.loc[df["Model year"] < 2023].copy()
    validation_df = df.loc[df["Model year"] == 2023].copy()
    test_df = df.loc[df["Model year"] == 2024].copy()

    return train_df, validation_df, test_df