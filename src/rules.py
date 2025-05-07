def get_recommendations(input_data, er_probability):
    """
    Evaluate diagnostic rules and return a list of recommended tests.

    Args:
        input_data (pd.DataFrame): Input data with patient features.
        er_probability (float): Probability of needing ER evaluation.

    Returns:
        list: List of test recommendation strings.
    """
    # Define diagnostic recommendation rules
    DIAGNOSTIC_RULES = {
        "stroke_suspected": {
            "condition": lambda data, prob: (
                                                    data["unilateral_weakness"] + data["trouble_speaking"] +
                                                    data["trouble_walking"] + data["syncope"] >= 1
                                            ) and prob > 0.5,
            "tests": [
                "- **Brain MRI with DWI**: High sensitivity and specificity for diagnosing acute stroke."
            ]
        },
        "cardiac_risk": {
            "condition": lambda data, prob: (
                    (data["blood_pressure"] > 140 or data["blood_pressure"] < 90 or
                     data["SpO2"] < 92) and data["heart_disease"] == 1 and data["age"] >= 65
            ),
            "tests": [
                "- **Stress Echocardiography**: Assesses heart function under stress to detect coronary artery disease.",
                "- **NT-proBNP or BNP Blood Test**: Evaluates heart failure risk.",
                "- **Coronary Calcium Score (CAC) – Non-contrast CT**: Quantifies coronary artery calcification."
            ]
        },
        "pulmonary_issues": {
            "condition": lambda data, prob: (
                    data["SpO2"] < 92 and data["shortness_of_breath"] == 1 and prob > 0.5
            ),
            "tests": [
                "- **Chest X-ray**: Detects pneumonia, consolidation, or other lung abnormalities.",
                "- **Arterial Blood Gas (ABG)**: Assesses oxygenation and CO2 levels.",
                "- **Complete Blood Count (CBC)**: Checks for infection (e.g., elevated white blood cells)."
            ]
        },
        "infection_sepsis": {
            "condition": lambda data, prob: data["temperature"] > 38 and prob > 0.5,
            "tests": [
                "- **Blood Cultures**: Identifies bloodstream infections.",
                "- **C-reactive Protein (CRP) and Procalcitonin**: Biomarkers for systemic infection or sepsis.",
                "- **Urinalysis and Urine Culture**: Rules out urinary tract infection."
            ]
        },
        "cardiac_ischemia": {
            "condition": lambda data, prob: (
                    data["chest_pain"] == 1 and data["heart_disease"] == 1 and prob > 0.5
            ),
            "tests": [
                "- **Electrocardiogram (ECG)**: Detects ischemic changes (e.g., ST elevation).",
                "- **Troponin I or T Blood Test**: Highly specific for myocardial injury.",
                "- **Coronary Angiography** (if initial tests confirm ischemia): Identifies coronary artery blockages."
            ]
        }
    }

    recommendations = []
    input_dict = input_data.iloc[0].to_dict()

    for condition, rule in DIAGNOSTIC_RULES.items():
        if rule["condition"](input_dict, er_probability):
            recommendations.extend(rule["tests"])

    return recommendations