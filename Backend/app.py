import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import shap
import json
from datetime import datetime
import random
from google.cloud import storage
import base64
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting application...")

# Load and decode credentials from environment variable
credentials_base64 = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_BASE64')
if credentials_base64:
    credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
    credentials_dict = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
else:
    # Fallback to file-based credentials for local development
    credentials = service_account.Credentials.from_service_account_file(
        os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '../alzheimer-model-8079e6d0ea47.json')
    )

# Create storage client
storage_client = storage.Client(credentials=credentials)

app = Flask(__name__)
CORS(app)

logger.info("Flask app created and CORS initialized")

def download_xgboost_dll(bucket_name, source_blob_name, destination_file_name):
    """Downloads xgboost.dll from Google Cloud Storage bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.info(f"Downloaded {source_blob_name} to {destination_file_name}")
    except Exception as e:
        logger.error(f"Error downloading xgboost.dll: {str(e)}")

# Before loading the model, download the xgboost.dll
bucket_name = "alzheimers-backend-xgboost"  # Replace with your bucket name
source_blob_name = "xgboost.dll"
destination_file_name = "../xgboost/lib/xgboost.dll"  # Path to store the DLL locally

download_xgboost_dll(bucket_name, source_blob_name, destination_file_name)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)

app.json_encoder = CustomJSONEncoder

# Load the model
try:
    logger.info("Attempting to load model...")
    model = joblib.load('../alzheimers_risk_model.joblib')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

if model is not None:
    categorical_features = ['STRONGEST SNP-RISK ALLELE']
    numeric_features = ['P-VALUE', 'OR or BETA', 'PVALUE_MLOG']
    special_numeric = ['RISK ALLELE FREQUENCY']

    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_names = (model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist() +
                     numeric_features +
                     special_numeric)

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    logger.info("Model preprocessing completed")
else:
    logger.warning("Model not loaded. Some functionality may be limited.")

@app.route('/')
def home():
    return "Alzheimer's Predictor Backend is running!"

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        sorted_importance = importance_df.sort_values('importance', ascending=False)
        top_features = sorted_importance.head(10)
        feature_importance_list = [
            {"feature": row['feature'], "importance": float(row['importance'])}
            for _, row in top_features.iterrows()
        ]
        logger.info("Feature importance retrieved successfully")
        return jsonify(feature_importance_list)
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        return jsonify({'error': 'Unable to retrieve feature importance'}), 500

def predict_alzheimers_risk(new_data):
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        return None, None
    try:
        logger.info(f"Prediction request received. Input data: {new_data}")
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
        preprocessed_data = model.named_steps['preprocessor'].transform(new_data)
        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        risk_probability = model.predict_proba(new_data)[0][1]
        logger.info(f"Raw prediction: {risk_probability}")
        shap_values = explainer.shap_values(preprocessed_data)
        logger.info(f"SHAP values calculated. Shape: {np.array(shap_values).shape}")
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        return risk_probability, shap_values
    except Exception as e:
        logger.error(f"Error in prediction function: {str(e)}")
        return None, None

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.json
        logger.info(f"Received data: {data}")
        required_fields = ['snpRiskAllele', 'pValue', 'orBeta', 'riskAlleleFrequency', 'pValueMlog']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        input_data = pd.DataFrame({
            'STRONGEST SNP-RISK ALLELE': [data['snpRiskAllele']],
            'P-VALUE': [float(data['pValue'])],
            'OR or BETA': [float(data['orBeta'])],
            'RISK ALLELE FREQUENCY': [float(data['riskAlleleFrequency'])],
            'PVALUE_MLOG': [float(data['pValueMlog'])]
        })
        risk_probability, shap_values = predict_alzheimers_risk(input_data)
        if risk_probability is None or shap_values is None:
            return jsonify({'error': 'Prediction failed'}), 500
        total_shap = np.sum(np.abs(shap_values[0]))
        risk_breakdown = {
            'totalRisk': float(risk_probability),
            'features': [
                {
                    'name': feature_names[i],
                    'contribution': float(shap_values[0][i]),
                    'relativeImportance': float(np.abs(shap_values[0][i]) / total_shap)
                } for i in range(len(feature_names))
            ]
        }
        response_data = {
            'risk': risk_probability * 100,
            'riskBreakdown': risk_breakdown,
            'shap_values': shap_values[0].tolist(),
            'feature_names': feature_names,
            'timestamp': datetime.now().isoformat()
        }
        logger.info(f"Prediction successful. Risk: {risk_probability * 100}%")
        return app.response_class(
            response=json.dumps(response_data, cls=CustomJSONEncoder),
            status=200,
            mimetype='application/json'
        )
    except ValueError as ve:
        logger.error(f"Value error in prediction: {str(ve)}")
        return jsonify({'error': f'Invalid input data: {str(ve)}'}), 400
    except Exception as e:
        logger.error(f"Error in prediction route: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/sample_data', methods=['GET'])
def get_sample_data():
    try:
        sample_data = {
            'snpRiskAllele': random.choice(['rs429358-C', 'rs7412-T', 'rs3752246-A']),
            'pValue': float(f"{random.uniform(1e-200, 1e-10):.2e}"),
            'orBeta': round(random.uniform(1.0, 5.0), 3),
            'riskAlleleFrequency': round(random.uniform(0.01, 0.5), 2),
            'pValueMlog': random.randint(100, 300)
        }
        logger.info("Sample data retrieved successfully")
        return jsonify(sample_data)
    except Exception as e:
        logger.error(f"Error retrieving sample data: {str(e)}")
        return jsonify({'error': 'Unable to retrieve sample data'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/test_gcs')
def test_gcs():
    try:
        bucket = storage_client.get_bucket('alzheimers-backend-xgboost')
        blobs = list(bucket.list_blobs())
        return jsonify([blob.name for blob in blobs])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)