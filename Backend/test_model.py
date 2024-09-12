import requests
import json

# Set the URL of your local Flask app or deployed API
# If testing locally, use: http://127.0.0.1:5000/predict
# If testing on Google Cloud Run, replace with your deployed URL
url = 'http://127.0.0.1:5000/predict'

# Example data for testing the model
test_data = {
    "snpRiskAllele": "rs429358-C",
    "pValue": 1e-200,
    "orBeta": 3.685,
    "riskAlleleFrequency": 0.15,
    "pValueMlog": 200
}

# Send POST request to the API
try:
    response = requests.post(url, json=test_data)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the prediction results
        prediction = response.json()
        print("Prediction response:")
        print(json.dumps(prediction, indent=4))
    else:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
