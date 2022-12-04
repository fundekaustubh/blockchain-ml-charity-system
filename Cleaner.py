import pandas as pd
from Models import Models


class Cleaner:
    def __init__(self, data):
        self.data = self.preprocess(data)

    def preprocess(self, data):
        print(data.shape)
        self.remove_null_values(data)
        print(data.shape)
        self.remove_unnecessary_textual_columns(data)
        print(data.shape)
        # self.encode_categorical_columns(data)
        # data = encode_categorical_columns(data)
        print("Cleaning done...")
        return data

    def remove_null_values(self, data):
        data.dropna(inplace=True)

    def remove_unnecessary_textual_columns(self, data):
        columns_to_be_removed = [
            "description",
            "leader",
            "ein",
            "motto",
            "subcategory",
            "ein",
        ]
        data.drop(columns_to_be_removed, inplace=True, axis=1)

    # def encode_categorical_columns(self, data):
    #     categorical_columns = []
    #     for col in data.columns:
    #     if data.dtypes[col] == 'object' and col != 'name':
    #         categorical_columns.append(col)
    #     categorical_columns
    #     return pd.get_dummies(data, columns=categorical_columns, drop_first=True)
