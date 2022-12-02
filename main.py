import pandas as pd
import flask
import numpy as np
import asyncio
from preprocessing import Preprocessor
from flask import Flask, request

app = Flask(__name__)
df = pd.read_csv("./dataset.csv")
preprocessor = Preprocessor(df)


@app.route("/", methods=["GET"])
def hello():
    if request.method == "GET":
        return "Hi"


@app.route("/recommend", methods=["POST"])
def recommend():
    if request.method == "GET":
        return "Hi!"


async def initiate_app():
    print("Starting preprocessing...")
    await asyncio.sleep(1)
    print("Ended preprocessing...")
    # To execute an async function
    # task = asyncio.create_task(async_func())
    # await task


if __name__ == "__main__":
    asyncio.run(initiate_app())
    app.run(debug=True)
    # asyncio.run(initiate_app())