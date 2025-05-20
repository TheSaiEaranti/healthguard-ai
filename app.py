from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "👋 Welcome to HealthGuard AI API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = np.array([[
            data["age"],
            data["blood_pressure"],
            data["cholesterol"],
            data["glucose_level"],
            data["smoking"],
            data["bmi"]
        ]])

        probability = model.predict_proba(features)[0][1]
        label = int(probability >= 0.5)

        return jsonify({
            "risk_score": round(float(probability), 2),
            "high_risk": bool(label)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)



