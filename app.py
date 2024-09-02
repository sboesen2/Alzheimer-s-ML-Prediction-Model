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

# Version Checking Yessir
import sklearn
import joblib
import xgboost
import pandas
import numpy

print(f"scikit-learn version: {sklearn.__version__}")
print(f"joblib version: {joblib.__version__}")
print(f"xgboost version: {xgboost.__version__}")
print(f"pandas version: {pandas.__version__}")
print(f"numpy version: {numpy.__version__}")

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting application...")

app = Flask(__name__)
CORS(app)

logger.info("Flask app created and CORS initialized")

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

try:
    logger.info("Attempting to load model...")
    model = joblib.load('alzheimers_risk_model.joblib')
    logger.info(f"Model loaded successfully. Type: {type(model)}")
    if hasattr(model, 'steps'):
        logger.info("Pipeline steps:")
        for step_name, step_estimator in model.steps:
            logger.info(f"- {step_name}: {type(step_estimator).__name__}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Files in current directory: {os.listdir('.')}")
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
    except KeyError as ke:
        logger.error(f"Key error in prediction: {str(ke)}")
        return jsonify({'error': f'Missing required field: {str(ke)}'}), 400
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)