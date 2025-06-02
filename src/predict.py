import os
import numpy as np
import joblib
from typing import Dict, List, Tuple


def qsofa_score(input_data: Dict) -> int:
    """Calculate qSOFA score."""
    bp = input_data.get('blood_pressure', 120)
    rr = input_data.get('respiratory_rate', 16)
    ams = input_data.get('altered_mental_status', 0)
    return (
        (1 if ams == 1 else 0) +
        (1 if rr >= 22 else 0) +
        (1 if bp <= 100 else 0)
    )


def stroke_symptoms_count(input_data: Dict) -> int:
    """Count stroke symptoms."""
    return sum([
        input_data.get('unilateral_weakness', 0),
        input_data.get('trouble_speaking', 0),
        input_data.get('trouble_walking', 0),
        input_data.get('syncope', 0)
    ])


def diabetes_glucose_check(input_data: Dict) -> bool:
    """Check diabetes with abnormal glucose."""
    diabetes = input_data.get('diabetes', 0)
    bs = input_data.get('blood_sugar', 100)
    return diabetes == 1 and (bs <= 70 or bs >= 200)


# Rule definitions: feature -> (condition_function, weight, description)
RULES = {
    'SpO2': [(lambda x: x < 90, 0.60, 'SpO2 < 90%')],
    'blood_pressure': [
        (lambda x: x < 90, 0.60, 'Blood pressure < 90 mmHg'),
        (lambda x: x > 140, 0.30, 'Blood pressure > 140 mmHg')
    ],
    'temperature': [(lambda x: x > 38, 0.50, 'Temperature > 38°C')],
    'chest_pain': [(lambda x: x == 1, 0.60, 'Chest pain present')],
    'shortness_of_breath': [(lambda x: x == 1, 0.50, 'Shortness of breath present')],
    'heart_disease': [(lambda x: x == 1, 0.40, 'Heart disease present')],
    'age': [(lambda x: x >= 65, 0.30, 'Age ≥ 65 years')],
    'pulse': [
        (lambda x: x < 60, 0.80, 'Pulse < 60 bpm (hypothermia risk)'),
        (lambda x: x > 100, 0.60, 'Pulse > 100 bpm')
    ],
    'blood_sugar': [(lambda x: x <= 70 or x >= 272, 0.70, 'Blood glucose ≤ 70 or ≥ 272 mg/dL')],
    'mode_of_arrival': [(lambda x: x == 1, 0.60, 'Ambulance arrival')],
    'respiratory_rate': [
        (lambda x: x < 8, 0.30, 'Respiratory rate < 8'),
        (lambda x: 21 <= x <= 24, 0.30, 'Respiratory rate 21–24'),
        (lambda x: x >= 25, 0.60, 'Respiratory rate ≥ 25')
    ],
    '_qsofa': [
        (lambda _: qsofa_score(_) == 1, 0.50, 'qSOFA score = 1'),
        (lambda _: qsofa_score(_) >= 2, 1.00, 'qSOFA score ≥ 2')
    ],
    '_stroke': [(lambda _: stroke_symptoms_count(_) >= 1, 0.80, 'Stroke symptoms present')],
    '_diabetes_glucose': [(lambda _: diabetes_glucose_check(_), 0.80, 'Diabetes with blood glucose ≤ 70 or ≥ 200 mg/dL')]
}


def predict_er(input_data: Dict, script_dir: str) -> Tuple[float, List[Tuple[float, str]]]:
    """
    Generate ER probability with multiplicative weighting using optimized rules.

    Args:
        input_data: Input data with 17 features (sex, race ignored).
        script_dir: Directory of the calling script for path resolution.

    Returns:
        Tuple of (adjusted_probability, weights_applied).
    """
    # Load model
    PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "../.."))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "er_model.pkl")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    # Define features
    features = [
        'SpO2', 'blood_pressure', 'temperature', 'chest_pain', 'shortness_of_breath',
        'heart_disease', 'age', 'unilateral_weakness', 'trouble_speaking', 'trouble_walking',
        'syncope', 'pulse', 'blood_sugar', 'diabetes', 'mode_of_arrival', 'respiratory_rate',
        'altered_mental_status'
    ]

    # Prepare input
    try:
        X = np.array([[input_data.get(f, 0) for f in features]])
    except Exception as e:
        raise ValueError(f"Error preparing input data: {str(e)}. Expected features: {features}")

    # Predict base probability
    try:
        base_prob = model.predict_proba(X)[0][1]
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

    # Apply weights
    weights_applied = []
    adjusted_prob = base_prob

    for feature, rule_list in RULES.items():
        # Special features start with '_'
        value = input_data if feature.startswith('_') else input_data.get(feature, 0)
        for condition_fn, weight, description in rule_list:
            if condition_fn(value):
                adjusted_prob *= (1 + weight)
                weights_applied.append((weight, description))

    # Cap probability at 1.0
    adjusted_prob = min(adjusted_prob, 1.0)

    return adjusted_prob, weights_applied