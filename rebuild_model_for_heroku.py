import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv(r'C:\Users\14807\OneDrive\Spreadsheets\Data Science Capstone\AI_ML_Biotech\Data Collection\Final Data Merging after Collection\merged_alzheimers_data.csv')

# Create binary target variable
df['is_alzheimers'] = df['MAPPED_TRAIT'].str.contains('Alzheimer', case=False, na=False).astype(int)

# Select relevant features
features = ['STRONGEST SNP-RISK ALLELE', 'P-VALUE', 'OR or BETA', 'RISK ALLELE FREQUENCY', 'PVALUE_MLOG']
X = df[features].copy()
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
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Save the model
joblib.dump(best_model, 'alzheimers_risk_model_heroku.joblib')

print("Model rebuilt and saved as alzheimers_risk_model_heroku.joblib")