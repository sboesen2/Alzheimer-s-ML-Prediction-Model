# Use Miniconda as the base image
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy your application script and other necessary files into the container
COPY app.py /app/app.py

# Copy the service account key file into the container
COPY alzheimers-prediction-backend-b77c386fb552.json /app/service-account-key.json

# Install necessary Python packages via conda and pip
RUN conda install --yes python=3.8 && \
    conda install --yes pandas numpy scikit-learn xgboost joblib shap scipy && \
    pip install flask flask-cors google-cloud-storage

# Create models directory
RUN mkdir -p /app/models

# Set environment variable to use the service account key
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Expose the port for Flask (8080 for Cloud Run)
EXPOSE 8080

# Command to run the Flask app
CMD ["python", "/app/app.py"]
