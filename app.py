from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

from auth.jwt_verify import verify_jwt
from utils.recommend import generate_recommendations

app = Flask(__name__)
CORS(app)

# Load ML model
model = joblib.load(open("model/heart_model.pkl", "rb"))

# Load providers
with open("data/providers.json", "r") as f:
    providers = json.load(f)

@app.route("/api/predict", methods=["POST"])
def predict():
    user, error = verify_jwt()
    if error:
        return jsonify({"message": error}), 401

    data = request.get_json()
    features = np.array([[data[k] for k in [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]]])

    prediction = model.predict_proba(features)[0][1]
    risk_percentage = round(prediction * 100, 2)

    if risk_percentage < 33:
        risk_level = "Low"
    elif risk_percentage < 66:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return jsonify({
        "riskPercentage": risk_percentage,
        "riskLevel": risk_level,
        "recommendations": generate_recommendations(risk_level)
    })

@app.route("/api/healthcare-providers", methods=["GET"])
def get_providers():
    return jsonify(providers)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
