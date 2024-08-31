from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import shap
import json
from datetime import datetime
import logging
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

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
    model = joblib.load('alzheimers_risk_model.joblib')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

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

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    try:
        sorted_importance = importance_df.sort_values('importance', ascending=False)
        top_features = sorted_importance.head(10)
        feature_importance_list = [
            {"feature": row['feature'], "importance": float(row['importance'])}
            for _, row in top_features.iterrows()
        ]
        logging.info("Feature importance retrieved successfully")
        return jsonify(feature_importance_list)
    except Exception as e:
        logging.error(f"Error retrieving feature importance: {str(e)}")
        return jsonify({'error': 'Unable to retrieve feature importance'}), 500

def predict_alzheimers_risk(new_data):
    try:
        logging.info(f"Prediction request received. Input data: {new_data}")
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
        preprocessed_data = model.named_steps['preprocessor'].transform(new_data)
        logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        risk_probability = model.predict_proba(new_data)[0][1]
        logging.info(f"Raw prediction: {risk_probability}")
        shap_values = explainer.shap_values(preprocessed_data)
        logging.info(f"SHAP values calculated. Shape: {np.array(shap_values).shape}")
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        return risk_probability, shap_values
    except Exception as e:
        logging.error(f"Error in prediction function: {str(e)}")
        return None, None

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    try:
        data = request.json
        logging.info(f"Received data: {data}")
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
        logging.info(f"Prediction successful. Risk: {risk_probability * 100}%")
        return app.response_class(
            response=json.dumps(response_data, cls=CustomJSONEncoder),
            status=200,
            mimetype='application/json'
        )
    except ValueError as ve:
        logging.error(f"Value error in prediction: {str(ve)}")
        return jsonify({'error': f'Invalid input data: {str(ve)}'}), 400
    except Exception as e:
        logging.error(f"Error in prediction route: {str(e)}")
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
        logging.info("Sample data retrieved successfully")
        return jsonify(sample_data)
    except Exception as e:
        logging.error(f"Error retrieving sample data: {str(e)}")
        return jsonify({'error': 'Unable to retrieve sample data'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

if __name__ == '__main__':
    app.run(debug=True)