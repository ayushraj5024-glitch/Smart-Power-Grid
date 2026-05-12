from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

CORS(app)

model = joblib.load("../models/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    # Convert Watts to MW
    prediction_mw = prediction / 1000000

    return jsonify({
        "prediction": round(prediction_mw, 2)
    })

app.run(debug=True)