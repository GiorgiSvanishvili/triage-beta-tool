import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to project root
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "triage_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "scaler", "scaler.pkl")
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "X_test_scaled.csv")
TEST_LABELS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "y_test.csv")
EVAL_DIR = os.path.join(SCRIPT_DIR, "..", "evaluation")
METRICS_PATH = os.path.join(EVAL_DIR, "metrics.txt")
PLOT_PATH = os.path.join(EVAL_DIR, "confusion_matrix.png")

# Create evaluation directory
os.makedirs(EVAL_DIR, exist_ok=True)

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Model and scaler loaded from {MODEL_PATH} and {SCALER_PATH}")
except FileNotFoundError:
    print(f"Error: Model or scaler files not found")
    exit(1)

# Load test data
try:
    X_test_scaled = pd.read_csv(TEST_DATA_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)["label"]
    print(f"Test data loaded from {TEST_DATA_PATH} and {TEST_LABELS_PATH}")
except FileNotFoundError:
    print(f"Error: Test data files not found")
    exit(1)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # TP / (TP + FN)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)

# Print results
print(f"Accuracy: {accuracy:.4f} (percent of correct predictions)")
print(f"Sensitivity: {sensitivity:.4f} (percent of ER cases caught)")
print(f"Specificity: {specificity:.4f} (percent of Discharge cases caught)")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Safe to Discharge", "Needs ER Evaluation"]))

# Plot confusion matrix
try:
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "ER"], yticklabels=["Safe", "ER"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Confusion matrix plot saved to {PLOT_PATH}")
except Exception as e:
    print(f"Error plotting confusion matrix: {e}")

# Save metrics
with open(METRICS_PATH, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {sensitivity:.4f}\n")
    f.write("\nConfusion Matrix:\n{cm}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=["Safe to Discharge", "Needs ER Evaluation"]))
print(f"Results saved to {METRICS_PATH}")