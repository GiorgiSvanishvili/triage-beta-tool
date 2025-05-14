import pandas as pd
import joblib
import os


def get_prediction(input_data, script_dir):
    """
    Generate prediction and probabilities for ER evaluation.

    Args:
        input_data (pd.DataFrame): Input data with 11 features.
        script_dir (str): Directory of the calling script for path resolution.

    Returns:
        tuple: (prediction, er_probability, discharge_probability, message)
    """
    # Define paths relative to project root
    PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "triage_model.pkl")
    SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler", "scaler.pkl")

    # Load model and scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model or scaler file not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading model or scaler: {str(e)}")

    # Check for two or more stroke-related symptoms
    try:
        stroke_related = [
            input_data["unilateral_weakness"].iloc[0],
            input_data["trouble_speaking"].iloc[0],
            input_data["trouble_walking"].iloc[0],
            input_data["syncope"].iloc[0]
        ]
    except KeyError as e:
        raise KeyError(f"Missing stroke-related feature: {str(e)}")

    if sum(stroke_related) >= 1:
        return (
            1,
            1.0,
            0.0,
            "One or more stroke-related symptoms detected. Immediate ER evaluation required."
        )

    # Scale continuous features
    continuous_features = ["SpO2", "blood_pressure", "temperature", "age"]
    binary_features = [
        "chest_pain", "shortness_of_breath", "heart_disease",
        "unilateral_weakness", "trouble_speaking", "trouble_walking", "syncope"
    ]

    try:
        scaled_data = scaler.transform(input_data[continuous_features])
        input_scaled = input_data.copy()
        input_scaled[continuous_features] = scaled_data
        input_scaled = input_scaled[[
            "SpO2", "blood_pressure", "temperature", "chest_pain",
            "shortness_of_breath", "heart_disease", "age",
            "unilateral_weakness", "trouble_speaking", "trouble_walking", "syncope"
        ]]
    except ValueError as e:
        raise ValueError(f"Scaler error: {str(e)}. Expected continuous features: {continuous_features}")
    except Exception as e:
        raise Exception(f"Unexpected scaling error: {str(e)}")

    # Predict with adjusted threshold
    threshold = 0.25
    try:
        probability = model.predict_proba(input_scaled)[0]
        prediction = 1 if probability[1] > threshold else 0
        er_probability = probability[1]
        discharge_probability = probability[0]
    except ValueError as e:
        raise ValueError(f"Prediction error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected prediction error: {str(e)}")

    return (prediction, er_probability, discharge_probability, "")