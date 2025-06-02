import streamlit as st
import os
import sys

# Ensure project directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

# Import prediction and recommendation modules
try:
    from predict import predict_er
    from rules import get_recommendations
except ImportError as e:
    st.error(f"Import error: {str(e)}. Ensure predict.py and rules.py are in {PROJECT_DIR}")
    st.stop()

# Set page title
st.set_page_config(page_title="ER Triage Tool", layout="wide")
st.title("ER Triage Tool")

# Create input form
# st.header("Enter Patient Data")
with st.form("triage_form"):
    # Demographic inputs
    st.subheader("Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        sex = st.selectbox("Sex", ["Male", "Female"])
    with col2:
        race = st.selectbox("Race", ["Asian", "Black", "White", "Hispanic", "Other"])
    with col3:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)

    # Clinical inputs
    st.subheader("Vital Signs")
    spo2 = st.slider("SpO2 (%)", min_value=80.0, max_value=100.0, value=98.0, step=0.1)
    blood_pressure = st.slider("Systolic Blood Pressure (mmHg)", min_value=70.0, max_value=200.0, value=120.0, step=1.0)
    temperature = st.slider("Temperature (°C)", min_value=35.0, max_value=40.0, value=37.0, step=0.1)
    pulse = st.slider("Pulse (bpm)", min_value=40, max_value=180, value=75, step=1)
    blood_sugar = st.slider("Blood Glucose (mg/dL)", min_value=50, max_value=400, value=100, step=1)
    respiratory_rate = st.slider("Respiratory Rate (breaths/min)", min_value=8, max_value=40, value=16, step=1)

    st.subheader("Symptoms and History")
    chest_pain = st.checkbox("Chest Pain")
    shortness_of_breath = st.checkbox("Shortness of Breath")
    heart_disease = st.checkbox("Heart Disease")
    unilateral_weakness = st.checkbox("Unilateral Weakness")
    trouble_speaking = st.checkbox("Trouble Speaking")
    trouble_walking = st.checkbox("Trouble Walking")
    syncope = st.checkbox("Syncope")
    diabetes = st.checkbox("Diabetes")
    altered_mental_status = st.checkbox("Altered Mental Status (e.g., confusion)")
    mode_of_arrival = st.selectbox("Mode of Arrival", ["Walk-in", "Ambulance", "Other"])

    # Submit button
    submitted = st.form_submit_button("Predict")

# Process prediction
if submitted:
    # Create input dictionary
    input_data = {
        'SpO2': spo2,
        'blood_pressure': blood_pressure,
        'temperature': temperature,
        'pulse': pulse,
        'blood_sugar': blood_sugar,
        'respiratory_rate': respiratory_rate,
        'chest_pain': int(chest_pain),
        'shortness_of_breath': int(shortness_of_breath),
        'heart_disease': int(heart_disease),
        'age': age,
        'unilateral_weakness': int(unilateral_weakness),
        'trouble_speaking': int(trouble_speaking),
        'trouble_walking': int(trouble_walking),
        'syncope': int(syncope),
        'diabetes': int(diabetes),
        'altered_mental_status': int(altered_mental_status),
        'mode_of_arrival': {'Walk-in': 0, 'Ambulance': 1, 'Other': 2}[mode_of_arrival],
        'sex': {'Male': 0, 'Female': 1}[sex],
        'race': {'Asian': 0, 'Black': 1, 'White': 2, 'Hispanic': 3, 'Other': 4}[race]
    }

    # Validate value ranges
    if not (80 <= spo2 <= 100):
        st.error(f"SpO2 value {spo2} is out of range (80–100%)")
        st.stop()
    if not (70 <= blood_pressure <= 200):
        st.error(f"Blood pressure value {blood_pressure} is out of range (70–200 mmHg)")
        st.stop()
    if not (35 <= temperature <= 40):
        st.error(f"Temperature value {temperature} is out of range (35–40°C)")
        st.stop()
    if not (40 <= pulse <= 180):
        st.error(f"Pulse value {pulse} is out of range (40–180 bpm)")
        st.stop()
    if not (50 <= blood_sugar <= 400):
        st.error(f"Blood glucose value {blood_sugar} is out of range (50–400 mg/dL)")
        st.stop()
    if not (8 <= respiratory_rate <= 40):
        st.error(f"Respiratory rate value {respiratory_rate} is out of range (8–40 breaths/min)")
        st.stop()
    if not (18 <= age <= 100):
        st.error(f"Age value {age} is out of range (18–100 years)")
        st.stop()

    # Get prediction
    try:
        prob, weights_applied = predict_er(input_data, SCRIPT_DIR)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

    # Get recommendations
    try:
        recommendations = get_recommendations(input_data)
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        st.stop()

    # Display results
    st.header("Prediction")
    st.write(f"**ER Probability**: {prob * 100:.1f}%")
    with st.expander("View Recommended Actions", expanded=True):
        st.markdown("**Recommended Actions**")
        if recommendations and recommendations != ['No recommendations']:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.markdown("No specific actions")