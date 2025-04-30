import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to project root
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "synthetic_data.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "triage_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "scaler", "scaler.pkl")
X_TEST_PATH = os.path.join(SCRIPT_DIR, "..", "data", "X_test.csv")
Y_TEST_PATH = os.path.join(SCRIPT_DIR, "..", "data", "y_test.csv")

# Load data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found")
    exit(1)

# Features (11 total: separate stroke-related features)
features = ['SpO2', 'blood_pressure', 'temperature', 'chest_pain', 'shortness_of_breath', 'heart_disease', 'age', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking', 'syncope']
continuous_features = ['SpO2', 'blood_pressure', 'temperature', 'age']
binary_features = ['chest_pain', 'shortness_of_breath', 'heart_disease', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking', 'syncope']
X = df[features]
y = df['ER_needed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale continuous features only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
print(f"Scaler trained on continuous features: {continuous_features}")

# Train model with adjusted class weights
model = LogisticRegression(random_state=42, C=1.0, class_weight={0: 1, 1: 5})
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")

# Save test data for evaluation
os.makedirs(os.path.dirname(X_TEST_PATH), exist_ok=True)
X_test.to_csv(X_TEST_PATH, index=False)
pd.DataFrame({'ER_needed': y_test}).to_csv(Y_TEST_PATH, index=False)
print(f"Test data saved to {X_TEST_PATH} and {Y_TEST_PATH}")