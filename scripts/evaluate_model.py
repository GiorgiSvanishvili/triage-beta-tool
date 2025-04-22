import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up directories
os.makedirs("/Users/nn/PycharmProjects/triage-beta-tool/evaluation", exist_ok=True)

# Load the trained model and scaler
try:
    model = joblib.load("/Users/nn/PycharmProjects/triage-beta-tool/model/triage_model.pkl")
    print("Model loaded successfully")
except FileNotFoundError:
    print("Error: triage_model.pkl not found in triage-beta-tool/model/")
    exit(1)

try:
    scaler = joblib.load("/Users/nn/PycharmProjects/triage-beta-tool/scaler/scaler.pkl")
    print("Scaler loaded successfully")
except FileNotFoundError:
    print("Error: scaler.pkl not found in triage-beta-tool/model/")
    exit(1)

# Load test data
try:
    X_test_scaled = pd.read_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/X_test_scaled.csv")
    y_test = pd.read_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/y_test.csv")["label"]
    print("Test data loaded successfully")
except FileNotFoundError:
    print("Error: X_test_scaled.csv or y_test.csv not found in triage-beta-tool/data/")
    exit(1)

# Make predictions on test data
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
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Safe", "ER"], yticklabels=["Safe", "ER"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save plot
plt.savefig("/Users/nn/PycharmProjects/triage-beta-tool/evaluation/confusion_matrix.png")
plt.close()

# Save metrics to a text file
with open("/Users/nn/PycharmProjects/triage-beta-tool/evaluation/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write("\nConfusion Matrix:\n{cm}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=["Safe to Discharge", "Needs ER Evaluation"]))

print("Results saved in triage-beta-tool/evaluation/")