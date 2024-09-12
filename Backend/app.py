import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import json
from datetime import datetime
from google.cloud import storage


# Custom JSON Encoder to handle NumPy and datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        if isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalars to native Python types
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO format
        return super().default(obj)


print(f"Using Python interpreter: {sys.executable}")

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting application...")

# Initialize Flask app and CORS
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder  # Set custom JSON encoder for the app
CORS(app)

logger.info("Flask app created and CORS initialized")

# Google Cloud Storage configuration
bucket_name = "alzheimers-models"  # Replace with your bucket name
pipeline_filename = "alzheimers_risk_model_retrained2.joblib"
local_pipeline_path = "/app/models/alzheimers_risk_model_retrained2.joblib"

# Set up Google Cloud Storage client
storage_client = storage.Client()


def download_pipeline_from_gcs(bucket_name, pipeline_filename, destination):
    """Download pipeline from GCS bucket to a local file."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(pipeline_filename)
        blob.download_to_filename(destination)
        logger.info(f"Downloaded pipeline from GCS: {pipeline_filename}")
    except Exception as e:
        logger.error(f"Error downloading pipeline from GCS: {str(e)}")


# Download pipeline from GCS if not available locally
if not os.path.exists(local_pipeline_path):
    logger.info("Pipeline not found locally. Downloading from Google Cloud Storage...")
    download_pipeline_from_gcs(bucket_name, pipeline_filename, local_pipeline_path)
else:
    logger.info("Pipeline already exists locally.")

# Load the pipeline (model + preprocessor)
try:
    pipeline = joblib.load(local_pipeline_path)
    logger.info("Pipeline (model + preprocessor) loaded successfully.")
except Exception as e:
    logger.error(f"Error loading pipeline: {str(e)}")
    pipeline = None


@app.route('/')
def home():
    return "Alzheimer's Predictor Backend is running!"


@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    # Placeholder route for future feature importance implementation
    return "Feature importance route is a placeholder."


def predict_alzheimers_risk(input_data):
    if pipeline is None:
        logger.error("Prediction attempted but pipeline is not loaded")
        return None
    try:
        logger.info(f"Prediction request received. Input data: {input_data}")

        # Apply the preprocessor and make the prediction
        risk_probability = pipeline.predict_proba(input_data)[:, 1][0]  # Get probability of class 1 (Alzheimer's)

        logger.info(f"Prediction successful. Risk Probability: {risk_probability}")
        return risk_probability
    except Exception as e:
        logger.error(f"Error in prediction function: {str(e)}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        # Extract the data from the request body
        data = request.json
        required_fields = ['snpRiskAllele', 'pValue', 'orBeta', 'riskAlleleFrequency', 'pValueMlog']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Prepare input data in the expected format
        input_data = pd.DataFrame({
            'STRONGEST SNP-RISK ALLELE': [data['snpRiskAllele']],
            'P-VALUE': [float(data['pValue'])],
            'OR or BETA': [float(data['orBeta'])],
            'RISK ALLELE FREQUENCY': [float(data['riskAlleleFrequency'])],
            'PVALUE_MLOG': [float(data['pValueMlog'])]
        })

        # Run prediction
        risk_probability = predict_alzheimers_risk(input_data)
        if risk_probability is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Format the response
        response_data = {
            'risk': risk_probability * 100,
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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Cloud Run expects the app to listen on port 8080
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
