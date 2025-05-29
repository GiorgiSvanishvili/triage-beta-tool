import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to project root
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "er_data.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "er_model.pkl")

# Load data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found")
    exit(1)

# Features (17 total)
features = [
    'SpO2', 'blood_pressure', 'temperature', 'chest_pain', 'shortness_of_breath',
    'heart_disease', 'age', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking',
    'syncope', 'pulse', 'blood_sugar', 'diabetes', 'mode_of_arrival', 'respiratory_rate',
    'altered_mental_status'
]
X = df[features]
y = df['needs_er']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)
print("Model trained successfully")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")