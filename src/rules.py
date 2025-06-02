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
    """Check diabetes with high glucose."""
    diabetes = input_data.get('diabetes', 0)
    bs = input_data.get('blood_sugar', 100)
    return diabetes == 1 and bs >= 200

# Rule definitions: feature -> (condition_function, recommendation, weight)
RULES = {
    'SpO2': [
        (lambda x: x < 90, "Hypoxia Assessment: Administer oxygen, consider imaging", 0.60)
    ],
    'blood_pressure': [
        (lambda x: x < 90, "Hypotension Assessment: Fluid resuscitation, monitor vitals", 0.60),
        (lambda x: x > 140, "Hypertension Assessment: Monitor, consider antihypertensives", 0.30)
    ],
    'temperature': [
        (lambda x: x > 38, "Infection Assessment: Initiate blood work, antipyretics", 0.50)
    ],
    'chest_pain': [
        (lambda x: x == 1, "Cardiac Assessment: ECG, troponin test", 0.60)
    ],
    'shortness_of_breath': [
        (lambda x: x == 1, "Pulmonary Assessment: Oxygen, BNP test", 0.50)
    ],
    'heart_disease': [
        (lambda x: x == 1, "Cardiac Monitoring: Assess for exacerbation", 0.40)
    ],
    'age': [
        (lambda x: x >= 65, "Geriatric Assessment: Evaluate frailty, comorbidities", 0.30)
    ],
    'pulse': [
        (lambda x: x < 60, "Bradycardia Assessment: Check hypothermia, ECG", 0.80),
        (lambda x: x > 100, "Tachycardia Assessment: ECG, evaluate cause", 0.60)
    ],
    'blood_sugar': [
        (lambda x: x <= 70, "Hypoglycemia Treatment: Administer glucose", 0.70),
        (lambda x: x >= 272, "Hyperglycemia Assessment: Insulin, monitor", 0.70)
    ],
    'respiratory_rate': [
        (lambda x: x < 8, "Respiratory Assessment: Possible opioid toxicity or stroke", 0.30),
        (lambda x: 21 <= x <= 24, "Pulmonary Evaluation: Moderate tachypnea, assess hypoxia", 0.30),
        (lambda x: x >= 25, "Pulmonary Evaluation: Severe tachypnea, risk of decompensation", 0.60)
    ],
    'mode_of_arrival': [
        (lambda x: x == 1, "Rapid Assessment: Ambulance arrival indicates high acuity", 0.60)
    ],
    '_qsofa': [
        (lambda _: qsofa_score(_) == 1, "Monitor for Sepsis: Intermediate risk", 0.50),
        (lambda _: qsofa_score(_) >= 2, "Sepsis Workup: Initiate protocol (blood cultures, CRP)", 1.00)
    ],
    '_stroke': [
        (lambda _: stroke_symptoms_count(_) >= 1, "Stroke Assessment: Urgent imaging (MRI/CT)", 0.80)
    ],
    '_diabetes_glucose': [
        (lambda _: diabetes_glucose_check(_), "Diabetic Infection Risk: Blood cultures, antibiotics", 0.80)
    ]
}

def get_recommendations(input_data: Dict) -> List[str]:
    """
    Generate sorted recommendations based on input data using a rule-based system.

    Args:
        input_data: Input data with 17 features (sex, race ignored).

    Returns:
        Sorted list of recommendation strings.
    """
    recommendations = []

    for feature, rule_list in RULES.items():
        # Special features start with '_'
        value = input_data if feature.startswith('_') else input_data.get(feature, 0)
        for condition_fn, recommendation, weight in rule_list:
            if condition_fn(value):
                recommendations.append((recommendation, weight))

    # Sort by weight (descending) and extract unique recommendations
    recommendations.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_recommendations = []
    for rec, _ in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)

    return unique_recommendations if unique_recommendations else ["No recommendations"]