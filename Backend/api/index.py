from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Global variable for the model
model = None

def load_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'alzheimers_risk_model.joblib')
        model = joblib.load(model_path)
    return model

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    model = load_model()
    input_data = np.array([[
        data['snpRiskAllele'],
        float(data['pValue']),
        float(data['orBeta']),
        float(data['riskAlleleFrequency']),
        float(data['pValueMlog'])
    ]])
    prediction = model.predict_proba(input_data)[0][1]  # Probability of class 1
    return jsonify({'risk': float(prediction * 100)})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# Vercel requires a handler function
def handler(request):
    with app.request_context(request):
        return app.full_dispatch_request()

# Load the model when the file is imported
load_model()