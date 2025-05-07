import streamlit as st
import pandas as pd
import os
import sys

# Ensure src directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Import prediction and recommendation modules
try:
    from predict import get_prediction
    from rules import get_recommendations
except ImportError as e:
    st.error(f"Import error: {str(e)}. Ensure predict.py and rules.py are in {SRC_DIR}")
    st.stop()

# Set page title
st.title("Triage Tool Beta")

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
    # Create input dataframe
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

    # Validate input data
    expected_features = [
        "SpO2", "blood_pressure", "temperature", "chest_pain",
        "shortness_of_breath", "heart_disease", "age",
        "unilateral_weakness", "trouble_speaking", "trouble_walking", "syncope"
    ]
    if list(input_data.columns) != expected_features:
        st.error(f"Input data has incorrect features. Expected: {expected_features}")
        st.stop()

    # Validate value ranges
    if not (80 <= input_data["SpO2"].iloc[0] <= 100):
        st.error(f"SpO2 value {input_data['SpO2'].iloc[0]} is out of range (80–100%)")
        st.stop()
    if not (70 <= input_data["blood_pressure"].iloc[0] <= 200):
        st.error(f"Blood pressure value {input_data['blood_pressure'].iloc[0]} is out of range (70–200 mmHg)")
        st.stop()
    if not (35 <= input_data["temperature"].iloc[0] <= 42):
        st.error(f"Temperature value {input_data['temperature'].iloc[0]} is out of range (35–42°C)")
        st.stop()
    if not (18 <= input_data["age"].iloc[0] <= 100):
        st.error(f"Age value {input_data['age'].iloc[0]} is out of range (18–100 years)")
        st.stop()

    # Get prediction
    try:
        prediction, er_probability, discharge_probability, message = get_prediction(input_data, SCRIPT_DIR)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

    # Display prediction
    st.header("Prediction")
    st.write(f"**Result**: {'Needs ER Evaluation' if prediction == 1 else 'Safe to Discharge'}")
    st.write(f"**ER Probability**: {er_probability * 100:.1f}%")
    st.write(f"**Discharge Probability**: {discharge_probability * 100:.1f}%")
    if message:
        st.warning(message)
    if prediction == 1:
        st.warning("This patient may need urgent care. Please review carefully.")

    # Diagnostic Test Recommendations (only if Needs ER Evaluation)
    if prediction == 1:
        st.header("Recommended Diagnostic Tests")
        try:
            recommendations = get_recommendations(input_data, er_probability)
        except Exception as e:
            st.error(f"Recommendation error: {str(e)}")
            st.stop()

        if recommendations:
            st.write("Based on the patient’s symptoms and risk profile, the following tests are recommended:")
            for rec in recommendations:
                st.write(rec)
        else:
            st.write(
                "No specific diagnostic tests recommended based on current criteria. Consider standard ER evaluation.")