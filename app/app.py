import streamlit as st
import pandas as pd
import joblib
import os

# Get the directory of the current script (app.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to script's parent directory (triage-tool-beta/)
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "triage_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "scaler", "scaler.pkl")

# Set page title
st.title("Triage Tool Beta")

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.success("Model and scaler loaded successfully")
except FileNotFoundError:
    st.error(f"Error: Model or scaler files not found. Ensure {MODEL_PATH} and {SCALER_PATH} exist.")
    st.stop()

# Create input form
st.header("Enter Patient Data")
with st.form("triage_form"):
    temperature = st.slider("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.7, step=0.1)
    spo2 = st.slider("SpO2 (%)", min_value=80.0, max_value=100.0, value=98.0, step=0.1)
    sbp = st.slider("Systolic BP (mmHg)", min_value=70.0, max_value=200.0, value=113.0, step=1.0)
    heart_rate = st.slider("Heart Rate (bpm)", min_value=40.0, max_value=180.0, value=91.0, step=1.0)
    chest_pain = st.checkbox("Chest Pain", value=False)
    dyspnea = st.checkbox("Dyspnea (Shortness of Breath)", value=False)
    syncope = st.checkbox("Syncope (Fainting)", value=False)
    age = st.slider("Age (years)", min_value=18, max_value=100, value=28, step=1)
    comorbidities = st.checkbox("Comorbidities (Diabetes or Heart Disease)", value=False)

    # Submit button
    submitted = st.form_submit_button("Predict")

# Process prediction
if submitted:
    # Create input dataframe
    input_data = pd.DataFrame({
        "temperature": [temperature],
        "spo2": [spo2],
        "sbp": [sbp],
        "heart_rate": [heart_rate],
        "chest_pain": [1 if chest_pain else 0],
        "dyspnea": [1 if dyspnea else 0],
        "syncope": [1 if syncope else 0],
        "age": [age],
        "comorbidities": [1 if comorbidities else 0]
    })

    # Scale continuous features
    continuous_features = ["temperature", "spo2", "sbp", "heart_rate", "age"]
    input_scaled = input_data.copy()
    input_scaled[continuous_features] = scaler.transform(input_data[continuous_features])

    # Predict with custom threshold
    threshold = 0.7
    probability = model.predict_proba(input_scaled)[0]
    prediction = 1 if probability[1] > threshold else 0

    # Display result
    result = "Needs ER Evaluation" if prediction == 1 else "Safe to Discharge"
    st.header("Prediction")
    st.write(f"**Result**: {result}")
    st.write(f"**ER Probability**: {probability[1] * 100:.1f}%")
    st.write(f"**Discharge Probability**: {probability[0] * 100:.1f}%")

    # Highlight critical cases
    if prediction == 1:
        st.warning("This patient may need urgent care. Please review carefully.")