import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
try:
    df = pd.read_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/er_data.csv")
except FileNotFoundError:
    print("Error: er_data.csv not found in triage-beta-tool/data/")
    exit(1)

# Define features and target (using needs_er from er_data.csv)
features = [
    'SpO2', 'blood_pressure', 'temperature', 'pulse', 'blood_sugar',
    'respiratory_rate', 'chest_pain', 'shortness_of_breath', 'heart_disease',
    'age', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking',
    'syncope', 'diabetes', 'altered_mental_status', 'mode_of_arrival'
]
X = df[features]
y = df["needs_er"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale continuous features
scaler = StandardScaler()
continuous_features = ['SpO2', 'blood_pressure', 'temperature', 'pulse', 'blood_sugar', 'respiratory_rate', 'age']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

# Save unscaled test data for evaluate_model.py
os.makedirs("/Users/nn/PycharmProjects/triage-beta-tool/data", exist_ok=True)
X_test.to_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/X_test.csv", index=False)
y_test.to_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/y_test.csv", index=False)

# Optional: Save scaled data for reference
X_test_scaled.to_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/X_test_scaled.csv", index=False)
joblib.dump(scaler, "/Users/nn/PycharmProjects/triage-beta-tool/scaler/scaler.pkl")

print("Test data saved to triage-beta-tool/data/X_test.csv and y_test.csv")
print("Scaled test data saved to triage-beta-tool/data/X_test_scaled.csv")
print("Scaler saved to triage-beta-tool/scaler/scaler.pkl")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)