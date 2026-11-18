"""Flask web application for House Price Prediction.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.config import DESCRIPTIONS, FEATURES, METRICS_PATH, MODEL_DIR, MODEL_FILES
from src.train_models import run as train_all

app = Flask(__name__)

_models, _info = {}, {}


def load_artifacts():
    """Load trained models; train them first if they do not exist yet."""
    global _models, _info
    if not os.path.exists(METRICS_PATH):
        print("No trained models found - training now (takes ~1 minute)...")
        train_all()
    with open(METRICS_PATH) as f:
        _info = json.load(f)
    _models = {n: joblib.load(os.path.join(MODEL_DIR, fn)) for n, fn in MODEL_FILES.items()}


def predict_price(values, model_name):
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    return float(_models[model_name].predict(row)[0])


def parse_inputs(source):
    values = {}
    for f in FEATURES:
        raw = source.get(f)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"Missing value for {f}")
        try:
            values[f] = float(raw)
        except ValueError:
            raise ValueError(f"{f} must be a number")
    return values


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES, desc=DESCRIPTIONS,
                           stats=_info["feature_stats"], models=list(_models),
                           best=_info["best_model"], result=None, form={}, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form.get("model", _info["best_model"])
    if model_name not in _models:
        model_name = _info["best_model"]
    result, error, values = None, None, {}
    try:
        values = parse_inputs(request.form)
        price = predict_price(values, model_name)
        result = {"model": model_name, "price_k": round(price, 2),
                  "price_usd": f"${price * 1000:,.0f}",
                  "all": {n: round(predict_price(values, n), 2) for n in _models}}
    except ValueError as exc:
        error = str(exc)
    return render_template("index.html", features=FEATURES, desc=DESCRIPTIONS,
                           stats=_info["feature_stats"], models=list(_models),
                           best=_info["best_model"], result=result, form=request.form,
                           error=error, selected=model_name)


@app.route("/performance")
def performance():
    return render_template("performance.html", info=_info)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API. Body: {"model": "Random Forest", "CRIM": .., ... "LSTAT": ..}"""
    data = request.get_json(silent=True) or {}
    model_name = data.get("model", _info["best_model"])
    if model_name not in _models:
        return jsonify(error=f"Unknown model. Choose from {list(_models)}"), 400
    try:
        values = parse_inputs(data)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    price = predict_price(values, model_name)
    return jsonify(model=model_name, predicted_price_in_1000s=round(price, 2),
                   predicted_price_usd=round(price * 1000))


load_artifacts()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
