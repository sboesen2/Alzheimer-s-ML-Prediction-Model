from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from google.cloud import storage
import tempfile

app = Flask(__name__)
CORS(app)

# Global variable for the model
model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def load_model():
    global model
    if model is None:
        # Download the model file from Google Cloud Storage
        bucket_name = "alzheimers-backend-xgboost"
        source_blob_name = "alzheimers_risk_model.joblib"
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            download_blob(bucket_name, source_blob_name, temp_file.name)
            model = joblib.load(temp_file.name)
        os.unlink(temp_file.name)  # Delete the temporary file
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