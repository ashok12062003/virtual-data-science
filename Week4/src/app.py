
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "Week4" / "models" / "produce_model.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run Week4/src/train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Week 4 Machine Learning Prediction API",
        "prediction_endpoint": "/predict",
        "method": "POST",
        "required_fields": ["Particulars", "Frequency", "Unit"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Provide a valid JSON object."}), 400

    required = ["Particulars", "Frequency", "Unit"]
    missing = [field for field in required if field not in data]

    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing
        }), 400

    input_df = pd.DataFrame([{
        field: str(data[field]) if data[field] is not None else "Unknown"
        for field in required
    }])

    prediction = float(model.predict(input_df)[0])

    return jsonify({
        "predicted_3_2014": prediction,
        "model": "RandomForestRegressor"
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
