import pandas as pd


class Preprocessor:
    def __init__(self, data):
        self.data = data

    def remove_unnecessary_textual_columns(self):
        columns_to_be_removed = [
            "description",
            "leader",
            "ein",
            "motto",
            "subcategory",
            "ein",
        ]
        self.data.drop(columns_to_be_removed, inplace=True, axis=1)

    def remove_null_values(self):
        self.data.dropna(inplace=True)

    def preprocess(self):
        self.remove_unnecessary_textual_columns()
