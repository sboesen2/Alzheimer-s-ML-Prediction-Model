import base64

with open('alzheimers_risk_model.joblib', 'rb') as file:
    encoded = base64.b64encode(file.read()).decode('utf-8')

print(encoded)