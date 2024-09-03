import os
import datetime

filename = "alzheimers_risk_model.joblib"
file_stats = os.stat(filename)
print(f"File: {filename}")
print(f"Size: {file_stats.st_size / (1024 * 1024):.2f} MB")
print(f"Last modified: {datetime.datetime.fromtimestamp(file_stats.st_mtime)}")
print(f"Last accessed: {datetime.datetime.fromtimestamp(file_stats.st_atime)}")
print(f"Creation time: {datetime.datetime.fromtimestamp(file_stats.st_ctime)}")