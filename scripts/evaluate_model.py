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

# Load data
try:
    df = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)['needs_er']
    print(f"Test data loaded from {X_TEST_PATH} and {Y_TEST_PATH}")
except FileNotFoundError:
    print(f"Error: Test data files not found")
    exit(1)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found")
    exit(1)

# Define features in the exact order used during training
features = [
    'SpO2', 'blood_pressure', 'temperature', 'chest_pain', 'shortness_of_breath',
    'heart_disease', 'age', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking',
    'syncope', 'pulse', 'blood_sugar', 'diabetes', 'mode_of_arrival', 'respiratory_rate',
    'altered_mental_status'
]
X_test = df[features]  # Reorder columns to match training order

# Predict
try:
    y_pred = model.predict(X_test)
except ValueError as e:
    print(f"Prediction error: {e}")
    exit(1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

# Print metrics
print(f"Accuracy: {accuracy:.3f}")
print(f"Sensitivity (Recall): {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save metrics
os.makedirs(os.path.join(SCRIPT_DIR, "..", "evaluation"), exist_ok=True)
with open(os.path.join(SCRIPT_DIR, "..", "evaluation", "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Sensitivity (Recall): {sensitivity:.3f}\n")
    f.write(f"Specificity: {specificity:.3f}\n")
    f.write(f"Precision: {precision:.3f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
print(f"Metrics saved to {os.path.join(SCRIPT_DIR, '..', 'evaluation', 'metrics.txt')}")

# Optional threshold analysis (if model provides probabilities)
if hasattr(model, 'predict_proba'):
    thresholds = [0.5, 0.7, 0.9]
    for thresh in thresholds:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred_thresh = (y_pred_prob >= thresh).astype(int)
        sensitivity_thresh = recall_score(y_test, y_pred_thresh)
        specificity_thresh = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recompute based on threshold
        print(f"Threshold {thresh}: Sensitivity = {sensitivity_thresh:.3f}, Specificity = {specificity_thresh:.3f}")
        with open(os.path.join(SCRIPT_DIR, "..", "evaluation", "metrics.txt"), "a") as f:
            f.write(f"\nThreshold {thresh}: Sensitivity = {sensitivity_thresh:.3f}, Specificity = {specificity_thresh:.3f}\n")