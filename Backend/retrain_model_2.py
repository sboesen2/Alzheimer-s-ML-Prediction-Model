import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import os

# Load data (update with your actual dataset path)
df = pd.read_csv(r'C:\Users\14807\OneDrive\Spreadsheets\Data Science Capstone\AI_ML_Biotech\Data Collection\Final Data Merging after Collection\merged_alzheimers_data.csv')

# Create binary target variable
df['is_alzheimers'] = df['MAPPED_TRAIT'].str.contains('Alzheimer', case=False, na=False).astype(int)

# Select relevant features
features = ['STRONGEST SNP-RISK ALLELE', 'P-VALUE', 'OR or BETA', 'RISK ALLELE FREQUENCY', 'PVALUE_MLOG']
X = df[features].copy()
y = df['is_alzheimers']

# Handle 'NR' in 'RISK ALLELE FREQUENCY'
X['RISK ALLELE FREQUENCY'] = X['RISK ALLELE FREQUENCY'].replace('NR', np.nan)
X['RISK ALLELE FREQUENCY'] = pd.to_numeric(X['RISK ALLELE FREQUENCY'], errors='coerce')

# Define numeric and categorical columns
numeric_features = ['P-VALUE', 'OR or BETA', 'PVALUE_MLOG']
categorical_features = ['STRONGEST SNP-RISK ALLELE']
special_numeric = ['RISK ALLELE FREQUENCY']

# Preprocessor
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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform training
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

# Define the model directory as the existing Backend folder
model_dir = r'C:\Users\14807\Alzheimers Project\Backend'

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the full pipeline inside the existing Backend directory
full_pipeline_path = os.path.join(model_dir, 'alzheimers_risk_model_retrained2.joblib')
joblib.dump(pipeline, full_pipeline_path)
print(f"Full pipeline saved to {full_pipeline_path}")

# Save the XGBoost classifier separately using Booster.save_model in binary format
xgb_model = pipeline.named_steps['classifier']
binary_model_path = os.path.join(model_dir, 'alzheimers_xgboost_model.model')
xgb_model._Booster.save_model(binary_model_path)  # Save the model in binary format
print(f"XGBoost model saved separately as 'alzheimers_xgboost_model.model'")
