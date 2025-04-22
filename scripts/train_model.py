import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to project root
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "synthetic_data.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "triage_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "scaler", "scaler.pkl")

# Load data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found")
    exit(1)

# Prepare features and target
X = df[["temperature", "spo2", "sbp", "heart_rate", "chest_pain",
        "dyspnea", "syncope", "age", "comorbidities"]]
y = df["label"].map({"Safe to Discharge": 0, "Needs ER Evaluation": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale continuous features
continuous_features = ["temperature", "spo2", "sbp", "heart_rate", "age"]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled = X_test.copy()
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")

# Save test data for evaluation
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "X_test_scaled.csv")
TEST_LABELS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "y_test.csv")
X_test_scaled.to_csv(TEST_DATA_PATH, index=False)
y_test.to_csv(TEST_LABELS_PATH, index=False)
print(f"Test data saved to {TEST_DATA_PATH} and {TEST_LABELS_PATH}")