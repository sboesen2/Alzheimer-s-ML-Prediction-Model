import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
df = pd.read_csv(
    r'C:\Users\14807\OneDrive\Spreadsheets\Data Science Capstone\AI_ML_Biotech\Data Collection\Final Data Merging after Collection\merged_alzheimers_data.csv')

# Create binary target variable
df['is_alzheimers'] = df['MAPPED_TRAIT'].str.contains('Alzheimer', case=False, na=False).astype(int)

# Select relevant features
features = ['STRONGEST SNP-RISK ALLELE', 'P-VALUE', 'OR or BETA', 'RISK ALLELE FREQUENCY', 'PVALUE_MLOG']
X = df[features].copy()  # Create a copy to avoid SettingWithCopyWarning
y = df['is_alzheimers']

# Handle 'NR' in 'RISK ALLELE FREQUENCY'
X.loc[:, 'RISK ALLELE FREQUENCY'] = X['RISK ALLELE FREQUENCY'].replace('NR', np.nan)
X.loc[:, 'RISK ALLELE FREQUENCY'] = pd.to_numeric(X['RISK ALLELE FREQUENCY'], errors='coerce')

# Define numeric and categorical columns
numeric_features = ['P-VALUE', 'OR or BETA', 'PVALUE_MLOG']
categorical_features = ['STRONGEST SNP-RISK ALLELE']
special_numeric = ['RISK ALLELE FREQUENCY']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),
        ('special_num', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
            ('scaler', StandardScaler())
        ]), special_numeric)
    ])

# Create a pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(eval_metric='logloss', random_state=42))
])

# The rest of the code remains the same...

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = dict(zip(np.unique(y), len(y) / (len(np.unique(y)) * np.bincount(y))))

# Define hyperparameter space
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 4, 5],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.8, 0.9, 1.0],
    'classifier__scale_pos_weight': [class_weights[1] / class_weights[0]]
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=5, random_state=42,
                                   n_jobs=-1)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
logging.info(f"Best hyperparameters: {random_search.best_params_}")

# Evaluate the model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc')
print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# Function to predict risk for new data
def predict_alzheimers_risk(new_data):
    try:
        # Handle 'NR' in 'RISK ALLELE FREQUENCY' for new data
        new_data = new_data.copy()
        new_data.loc[:, 'RISK ALLELE FREQUENCY'] = new_data['RISK ALLELE FREQUENCY'].replace('NR', np.nan)
        new_data.loc[:, 'RISK ALLELE FREQUENCY'] = pd.to_numeric(new_data['RISK ALLELE FREQUENCY'], errors='coerce')

        # Make prediction
        risk_probability = best_model.predict_proba(new_data)[0][1]  # Probability of class 1 (Alzheimer's)
        return risk_probability * 100  # Convert to percentage
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return None


# Save the model
joblib.dump(best_model, 'alzheimers_risk_model_retrained.joblib')

# Example usage
new_person = pd.DataFrame({
    'STRONGEST SNP-RISK ALLELE': ['rs429358-C'],
    'P-VALUE': [1e-200],
    'OR or BETA': [3.685],
    'RISK ALLELE FREQUENCY': [0.15],
    'PVALUE_MLOG': [200]
}, index=[0])

risk = predict_alzheimers_risk(new_person)
if risk is not None:
    print(f"\nPredicted Alzheimer's risk: {risk:.2f}%")
else:
    print("Unable to predict risk due to an error.")