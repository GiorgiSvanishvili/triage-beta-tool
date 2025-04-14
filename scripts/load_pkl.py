import joblib

# Load the model
model = joblib.load("/Users/nn/PycharmProjects/triage-beta-tool/model/triage_model.pkl")
print("Model loaded:", model)

# Load the scaler
scaler = joblib.load("/Users/nn/PycharmProjects/triage-beta-tool/scaler/scaler.pkl")
print("Scaler loaded:", scaler)

# Optional: Test the model with a sample input
import pandas as pd
sample_data = pd.DataFrame({
    "temperature": [39.5],
    "spo2": [90.0],
    "sbp": [95.0],
    "heart_rate": [105.0],
    "chest_pain": [1],
    "dyspnea": [0],
    "syncope": [0],
    "age": [70],
    "comorbidities": [1]
})
# Scale continuous features
continuous_features = ["temperature", "spo2", "sbp", "heart_rate", "age"]
sample_data_scaled = sample_data.copy()
sample_data_scaled[continuous_features] = scaler.transform(sample_data[continuous_features])

# Predict
prediction = model.predict(sample_data_scaled)
print("Prediction:", "Needs ER Evaluation" if prediction[0] == 1 else "Safe to Discharge")