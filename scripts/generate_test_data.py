import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility (same as modeling)
np.random.seed(42)

# Load the dataset
try:
    df = pd.read_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/synthetic_data.csv")
except FileNotFoundError:
    print("Error: synthetic_data.csv not found in triage-tool-beta/data/")
    exit(1)

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

# Save test data
X_test_scaled.to_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/X_test_scaled.csv", index=False)
y_test.to_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/y_test.csv", index=False)

print("Test data saved to triage-beta-tool/data/X_test_scaled.csv and y_test.csv")
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_test shape:", y_test.shape)