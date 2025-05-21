def get_recommendations(input_data, er_probability):
    """
    Evaluate diagnostic rules and return a list of recommended tests.

    Args:
        input_data (pd.DataFrame): Input data with patient features.
        er_probability (float): Probability of needing ER evaluation.

    Returns:
        list: List of test recommendation strings.
    """
    recommendations = []
    input_dict = input_data.iloc[0].to_dict()

    # Cardiac Risk (umbrella category)
    if input_dict["heart_disease"] == 1 and er_probability > 0.5:
        cardiac_risk_tests = []

        # Subcategory: Cardiac Ischemia
        if input_dict["chest_pain"] == 1:
            cardiac_risk_tests.extend([
                "- **Electrocardiogram (ECG)**: Detects ischemic changes (e.g., ST elevation).",
                "- **Troponin I or T Blood Test**: Highly specific for myocardial injury.",
                "- **Coronary Angiography** (if initial tests confirm ischemia): Identifies coronary artery blockages."
            ])

        # Subcategory: Congestive Heart Failure (CHF)
        if input_dict["shortness_of_breath"] == 1:
            cardiac_risk_tests.extend([
                "- **NT-proBNP or BNP Blood Test**: Evaluates heart failure risk.",
                "- **Echocardiogram**: Assesses heart function and ejection fraction for CHF diagnosis."
            ])

        # Subcategory: General Cardiac Risk
        if (input_dict["blood_pressure"] > 140 or input_dict["blood_pressure"] < 90 or
            input_dict["SpO2"] < 92) and input_dict["age"] >= 65:
            cardiac_risk_tests.extend([
                "- **Stress Echocardiography**: Assesses heart function under stress to detect coronary artery disease.",
                "- **Coronary Calcium Score (CAC) – Non-contrast CT**: Quantifies coronary artery calcification."
            ])

        if cardiac_risk_tests:
            recommendations.append("**Cardiac Risk Tests**: " + ", ".join(set(cardiac_risk_tests)))

    # Stroke Suspected
    if (input_dict["unilateral_weakness"] + input_dict["trouble_speaking"] +
        input_dict["trouble_walking"] + input_dict["syncope"]) >= 1 and er_probability > 0.5:
        recommendations.append("- **Brain MRI with DWI**: High sensitivity and specificity for diagnosing acute stroke.")

    # Pulmonary Issues
    if input_dict["SpO2"] < 92 and input_dict["shortness_of_breath"] == 1 and er_probability > 0.5:
        recommendations.extend([
            "- **Chest X-ray**: Detects pneumonia, consolidation, or other lung abnormalities.",
            "- **Arterial Blood Gas (ABG)**: Assesses oxygenation and CO2 levels.",
            "- **Complete Blood Count (CBC)**: Checks for infection (e.g., elevated white blood cells)."
        ])

    # Infection/Sepsis
    if input_dict["temperature"] > 38 and er_probability > 0.5:
        recommendations.extend([
            "- **Blood Cultures**: Identifies bloodstream infections.",
            "- **C-reactive Protein (CRP) and Procalcitonin**: Biomarkers for systemic infection or sepsis.",
            "- **Urinalysis and Urine Culture**: Rules out urinary tract infection."
        ])

    return recommendations if recommendations else ["- **No specific tests recommended**."]