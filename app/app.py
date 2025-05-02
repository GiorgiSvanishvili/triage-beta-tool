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
    blood_pressure = st.slider("Blood Pressure (Systolic, mmHg)", min_value=70.0, max_value=200.0, value=113.0,
                               step=1.0)
    chest_pain = st.checkbox("Chest Pain", value=False)
    shortness_of_breath = st.checkbox("Shortness of Breath", value=False)
    unilateral_weakness = st.checkbox("Sudden Unilateral Weakness", value=False)
    trouble_speaking = st.checkbox("Trouble Speaking", value=False)
    trouble_walking = st.checkbox("Trouble Walking or Loss of Balance", value=False)
    syncope = st.checkbox("Syncope (Fainting)", value=False)
    age = st.slider("Age (years)", min_value=18, max_value=100, value=28, step=1)
    heart_disease = st.checkbox("Heart Disease", value=False)

    # Submit button
    submitted = st.form_submit_button("Predict")

# Process prediction
if submitted:
    # Create input dataframe matching model features
    input_data = pd.DataFrame({
        "SpO2": [spo2],
        "blood_pressure": [blood_pressure],
        "temperature": [temperature],
        "chest_pain": [1 if chest_pain else 0],
        "shortness_of_breath": [1 if shortness_of_breath else 0],
        "heart_disease": [1 if heart_disease else 0],
        "age": [age],
        "unilateral_weakness": [1 if unilateral_weakness else 0],
        "trouble_speaking": [1 if trouble_speaking else 0],
        "trouble_walking": [1 if trouble_walking else 0],
        "syncope": [1 if syncope else 0]
    })

    # Check for two or more stroke-related symptoms
    stroke_related = [unilateral_weakness, trouble_speaking, trouble_walking, syncope]
    if sum(stroke_related) >= 2:
        st.header("Prediction")
        st.write("**Result**: Needs ER Evaluation")
        st.write("**ER Probability**: 100.0%")
        st.write("**Discharge Probability**: 0.0%")
        st.warning("Two or more stroke-related symptoms detected. Immediate ER evaluation required.")
    else:
        # Scale continuous features
        continuous_features = ["SpO2", "blood_pressure", "temperature", "age"]
        binary_features = ["chest_pain", "shortness_of_breath", "heart_disease", "unilateral_weakness",
                           "trouble_speaking", "trouble_walking", "syncope"]

        try:
            scaled_data = scaler.transform(input_data[continuous_features])
            input_scaled = input_data.copy()
            input_scaled[continuous_features] = scaled_data
            input_scaled = input_scaled[
                ["SpO2", "blood_pressure", "temperature", "chest_pain", "shortness_of_breath", "heart_disease", "age",
                 "unilateral_weakness", "trouble_speaking", "trouble_walking", "syncope"]]
        except ValueError as e:
            st.error(f"Scaler error: {e}. Expected continuous features: {continuous_features}")
            st.stop()

        # Predict with adjusted threshold
        threshold = 0.25
        try:
            probability = model.predict_proba(input_scaled)[0]
            prediction = 1 if probability[1] > threshold else 0
            st.write(f"Prediction probability (ER): {probability[1]:.3f}")
        except ValueError as e:
            st.error(
                f"Prediction error: {e}. Expected features: {['SpO2', 'blood_pressure', 'temperature', 'chest_pain', 'shortness_of_breath', 'heart_disease', 'age', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking', 'syncope']}")
            st.stop()

        # Display result
        st.header("Prediction")
        st.write(f"**Result**: {'Needs ER Evaluation' if prediction == 1 else 'Safe to Discharge'}")
        st.write(f"**ER Probability**: {probability[1] * 100:.1f}%")
        st.write(f"**Discharge Probability**: {probability[0] * 100:.1f}%")

        # Highlight critical cases
        if prediction == 1:
            st.warning("This patient may need urgent care. Please review carefully.")