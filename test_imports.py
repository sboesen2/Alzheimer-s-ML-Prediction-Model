import sys
print(f"Python version: {sys.version}")

packages = ['numpy', 'scipy', 'pandas', 'sklearn', 'xgboost', 'shap', 'flask', 'google.cloud.storage', 'joblib']

for package in packages:
    try:
        module = __import__(package)
        print(f"{package} version: {module.__version__}")
    except ImportError as e:
        print(f"Error importing {package}: {e}")
    except AttributeError:
        print(f"{package} imported successfully, but no version information available.")

# Additional checks
import numpy as np
print(f"NumPy dtype size: {np.dtype(int).itemsize}")

import sklearn
print(f"Scikit-learn file location: {sklearn.__file__}")

import shap
print(f"SHAP file location: {shap.__file__}")