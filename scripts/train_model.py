import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/synthetic_data.csv")

# Encode the label (Needs ER Evaluation = 1, Safe to Discharge = 0)
df["label"] = df["label"].map({"Needs ER Evaluation": 1, "Safe to Discharge": 0})

# Define features and target
features = [
    "temperature",
    "spo2",
    "sbp",
    "heart_rate",
    "chest_pain",
    "dyspnea",
    "syncope",
    "age",
    "comorbidities"
]
X = df[features]
y = df["label"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale continuous features (temperature, spo2, sbp, heart_rate, age)
scaler = StandardScaler()
continuous_features = ["temperature", "spo2", "sbp", "heart_rate", "age"]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Quick check: training accuracy
train_accuracy = model.score(X_train_scaled, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# # Create model directory if it doesn't exist
# os.makedirs("triage-tool-beta/model", exist_ok=True)

# Save the model and scaler
joblib.dump(model, "/Users/nn/PycharmProjects/triage-beta-tool/model/triage_model.pkl")
joblib.dump(scaler, "/Users/nn/PycharmProjects/triage-beta-tool/scaler/scaler.pkl")

print("Model and scaler saved to triage-tool-beta/model/")