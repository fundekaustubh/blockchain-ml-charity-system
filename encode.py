import pandas as pd


def encode(df):
    categorical_columns = []
    for col in df.columns:
        if df.dtypes[col] == "object" and col != "name":
            categorical_columns.append(col)
    temp_df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print("After encoding, shape of data is: ", temp_df.shape)
    return temp_df
