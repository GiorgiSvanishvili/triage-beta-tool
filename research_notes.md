## Research Summary
### Objective
Identify clinical indicators for a beta triage tool to classify patients as "Needs ER Evaluation" or "Safe to Discharge" using vital signs and medical history.

### Selected Indicators
Based on research indicating high correlation with emergency admission:
- **Vital Signs**:
  - Temperature: ≥ 39.4°C (high-grade fever, linked to severe infection).
  - SpO2: < 92% (indicates hypoxia, per triage guidelines).
  - Systolic BP: < 97 mmHg (suggests hypotension or shock).
  - Heart Rate: > 101 bpm (tachycardia, per ESI/MTS).
- **Symptoms** (binary):
  - Chest Pain: Yes/No (linked to cardiac emergencies).
  - Dyspnea: Yes/No (indicates respiratory distress).
  - Syncope: Yes/No (suggests neurological/cardiac issues).
- **Medical History**:
  - Age: Continuous, with > 65 years as high risk.
  - Comorbidities: Yes/No for diabetes or heart disease (increases risk).

### Decision Logic
- ER evaluation triggered by any: Temp ≥ 39.4°C, SpO2 < 92%, SBP < 97 mmHg, HR > 101 bpm, chest pain, dyspnea, or syncope.
- Age > 65 or comorbidities amplify risk, potentially triggering ER for borderline cases.

### Notes
- Uncontrolled bleeding excluded for beta due to complexity in synthetic data.
- Indicators align with triage systems (ESI, MTS) and literature on ER predictors.