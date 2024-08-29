
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('alzheimers_risk_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get genetic data from the request
        genetic_data = request.json['genetic_data']
        
        # Validate input
        if len(genetic_data) != 5:
            return jsonify({'error': 'Please provide exactly 5 values for genetic data.'}), 400
        
        # Convert to numpy array and reshape
        features = np.array(genetic_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict_proba(features)[0][1]
        
        # Return the risk score
        return jsonify({'risk_score': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
