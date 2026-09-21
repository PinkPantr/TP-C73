from preprocessing import split_data, prepare_data

csv_path = "data/my2015-2024-fuel-consumption-ratings.csv"

df= prepare_data(csv_path)
train_df, validation_df, test_df = split_data(df)

print("Entrainement: ", len(train_df))
print("Validation: ", len(validation_df))
print("Test final: ", len(test_df))