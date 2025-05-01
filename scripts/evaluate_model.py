import os
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import joblib

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to project root
X_TEST_PATH = os.path.join(SCRIPT_DIR, "..", "data", "X_test.csv")
Y_TEST_PATH = os.path.join(SCRIPT_DIR, "..", "data", "y_test.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "triage_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "scaler", "scaler.pkl")

# Load data
try:
    df = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)['ER_needed']
    print(f"Test data loaded from {X_TEST_PATH} and {Y_TEST_PATH}")
except FileNotFoundError:
    print(f"Error: Test data files not found")
    exit(1)

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Model and scaler loaded from {MODEL_PATH} and {SCALER_PATH}")
except FileNotFoundError:
    print(f"Error: Model or scaler files not found")
    exit(1)

# Features
features = ['SpO2', 'blood_pressure', 'temperature', 'chest_pain', 'shortness_of_breath', 'heart_disease', 'age', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking', 'syncope']
continuous_features = ['SpO2', 'blood_pressure', 'temperature', 'age']
binary_features = ['chest_pain', 'shortness_of_breath', 'heart_disease', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking', 'syncope']

# Scale continuous features
try:
    X_test_scaled = df.copy()
    X_test_scaled[continuous_features] = scaler.transform(df[continuous_features])
    X_test_scaled = X_test_scaled[features]
except ValueError as e:
    print(f"Scaler error: {e}")
    exit(1)

# Predict
try:
    y_pred = model.predict(X_test_scaled)
except ValueError as e:
    print(f"Prediction error: {e}")
    exit(1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.3f}")
print(f"Sensitivity (Recall): {sensitivity:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save metrics
os.makedirs(os.path.join(SCRIPT_DIR, "..", "evaluation"), exist_ok=True)
with open(os.path.join(SCRIPT_DIR, "..", "evaluation", "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Sensitivity (Recall): {sensitivity:.3f}\n")
    f.write(f"Precision: {precision:.3f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
print(f"Metrics saved to {os.path.join(SCRIPT_DIR, '..', 'evaluation', 'metrics.txt')}")