import joblib
import pandas as pd
import numpy as np
import shap
import pprint

# Paths to model file
model_path = "alzheimers_risk_model_new.joblib"

# Sample data to test prediction
sample_data = pd.DataFrame({
    'STRONGEST SNP-RISK ALLELE': ['A'],
    'P-VALUE': [0.05],
    'OR or BETA': [1.5],
    'RISK ALLELE FREQUENCY': [0.2],
    'PVALUE_MLOG': [1.3]
})


def check_model():
    """Load and inspect the model."""
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")

        # Check if the model is a pipeline
        if hasattr(model, 'steps'):
            print("\nModel is a pipeline. Inspecting steps:")
            for step_name, step_obj in model.steps:
                print(f"Step: {step_name}")
                print(f"Step type: {type(step_obj)}")
        else:
            print("Model does not appear to be a pipeline.")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def check_feature_importance(model):
    """Check if feature importance is accessible from the model."""
    try:
        feature_importance = model.named_steps['classifier'].feature_importances_
        print("\nFeature importance extracted.")

        categorical_features = ['STRONGEST SNP-RISK ALLELE']
        numeric_features = ['P-VALUE', 'OR or BETA', 'PVALUE_MLOG']
        special_numeric = ['RISK ALLELE FREQUENCY']

        feature_names = (model.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(categorical_features).tolist() +
                         numeric_features +
                         special_numeric)

        pprint.pprint(feature_names)
        pprint.pprint(feature_importance)

    except Exception as e:
        print(f"Error accessing feature importance: {e}")


def check_shap_explainer(model):
    """Check if SHAP explainer can be initialized and used."""
    try:
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        print("SHAP explainer initialized.")
        return explainer
    except Exception as e:
        print(f"Error initializing SHAP explainer: {e}")
        return None


def check_prediction(model, explainer):
    """Test model's prediction and SHAP values."""
    try:
        preprocessed_data = model.named_steps['preprocessor'].transform(sample_data)
        print(f"Preprocessed data shape: {preprocessed_data.shape}")

        # Predict risk probability
        risk_probability = model.predict_proba(sample_data)[0][1]
        print(f"Risk probability: {risk_probability}")

        # Calculate SHAP values
        shap_values = explainer.shap_values(preprocessed_data)
        print(f"SHAP values calculated. Shape: {np.array(shap_values).shape}")

    except Exception as e:
        print(f"Error during prediction or SHAP value calculation: {e}")


if __name__ == "__main__":
    # Load and inspect the model
    model = check_model()

    if model:
        # Check feature importance
        check_feature_importance(model)

        # Initialize and check SHAP explainer
        explainer = check_shap_explainer(model)

        if explainer:
            # Check prediction functionality
            check_prediction(model, explainer)
