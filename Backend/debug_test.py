import requests
import json

url = 'http://localhost:5000/predict'
data = {
    'snpRiskAllele': 'rs429358-C',
    'pValue': '1e-200',
    'orBeta': '3.685',
    'riskAlleleFrequency': '0.15',
    'pValueMlog': '200'
}

response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(json.dumps(response.json(), indent=2))

# Check if all expected keys are in the response
expected_keys = ['risk', 'riskBreakdown', 'shap_values', 'feature_names']
for key in expected_keys:
    if key not in response.json():
        print(f"Warning: Expected key '{key}' not found in response")

# Print shape of SHAP values if present
if 'shap_values' in response.json():
    print(f"Shape of SHAP values: {len(response.json()['shap_values'])}")