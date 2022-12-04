import pandas as pd
import flask
import numpy as np
import asyncio
import random
import itertools
from Cleaner import Cleaner
from Models import Models
from flask import Flask, request, jsonify
from encode import encode
from functools import cmp_to_key

app = Flask(__name__)
df = pd.read_csv("./dataset.csv")
cleaned_df = Cleaner(df)
final_df = encode(cleaned_df.data)
models = Models(final_df)
print("Cleaning dataframe...")


def sort_comparator(m1, m2):
    if m1[1][1] > m2[1][1]:
        return -1
    elif m1[1][1] < m2[1][1]:
        return 1
    else:
        return 0


@app.route("/", methods=["GET"])
def hello():
    if request.method == "GET":
        return "Hi"


@app.route("/recommend", methods=["POST"])
def recommend():
    if request.method == "POST":
        # user_categories = request.get_json(force=True)["userInterests"]
        # category_count = len(user_categories)
        categories = cleaned_df.data["category"].unique()
        category_count = 50
        user_categories = [random.choice(categories) for i in range(category_count)]
        user_categories_dict = dict()
        for ele in user_categories:
            if ele not in user_categories_dict.keys():
                user_categories_dict[ele] = 0
            else:
                user_categories_dict[ele] += 1
        user_categories_dict = sorted(
            user_categories_dict.items(), key=lambda x: x[1], reverse=True
        )
        user_categories_dict_sorted = dict(user_categories_dict)
        recommended_charities = []
        temp_arr = [item for item in models.models_accuracy.items()]
        best_model = sorted(temp_arr, key=cmp_to_key(sort_comparator))[0]
        ucd = list(user_categories_dict_sorted.items())
        count = 0
        itr = -1
        count_limit = 3
        while count < count_limit and itr < len(ucd) - 1:
            itr += 1
            if f"category_{ucd[itr][0]}" not in final_df.columns:
                continue
            count += 1
            category = ucd[itr][0]
            sub_df = final_df[final_df[f"category_{category}"] == 1]
            X_new = sub_df.loc[:, ~sub_df.columns.isin(["name", "score"])]
            y_pred = best_model[1][0].predict(X_new)
            sub_df = sub_df.assign(accountability_score=y_pred).sort_values(
                by="accountability_score", ascending=False
            )
            new_charities = sub_df.head(count_limit - count + 1)["name"].tolist()
            recommended_charities.append(new_charities)
        recommended_charities = list(itertools.chain(*recommended_charities))
        return jsonify(recommended_charities)


# async def initiate_app():
# print("Starting preprocessing...")
# await asyncio.sleep(1)
# print("Ended preprocessing...")
# To execute an async function
# task = asyncio.create_task(async_func())
# await task


if __name__ == "__main__":
    # asyncio.run(initiate_app())
    app.run(debug=True)
    # asyncio.run(initiate_app())