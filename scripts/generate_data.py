import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of patients
n_patients = 5000

# Generate synthetic data
data = {
    # Vital signs
    "temperature": np.random.normal(36.8, 0.5, n_patients),  # Normal: ~36-37.5°C
    "spo2": np.random.normal(97, 2, n_patients),           # Normal: ~95-100%
    "sbp": np.random.normal(120, 15, n_patients),          # Normal: ~100-140 mmHg
    "heart_rate": np.random.normal(80, 10, n_patients),    # Normal: ~60-100 bpm
    # Symptoms (binary)
    "chest_pain": np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),  # ~15% prevalence
    "dyspnea": np.random.choice([0, 1], n_patients, p=[0.80, 0.20]),     # ~20% prevalence
    "syncope": np.random.choice([0, 1], n_patients, p=[0.90, 0.10]),     # ~10% prevalence
    # Medical history
    "age": np.random.gamma(3, 15, n_patients),                   # Skewed, mean ~50
    "comorbidities": np.random.choice([0, 1], n_patients, p=[0.70, 0.30]),  # ~30% prevalence
}

# Create DataFrame
df = pd.DataFrame(data)

# Adjust distributions for realism
# Temperature: Add high-grade fever cases
fever_mask = np.random.choice([True, False], n_patients, p=[0.10, 0.90])
df.loc[fever_mask, "temperature"] = np.random.uniform(39.4, 41.0, fever_mask.sum())

# SpO2: Add hypoxia cases
hypoxia_mask = np.random.choice([True, False], n_patients, p=[0.15, 0.85])
df.loc[hypoxia_mask, "spo2"] = np.random.uniform(80, 91.9, hypoxia_mask.sum())

# SBP: Add hypotension cases
hypotension_mask = np.random.choice([True, False], n_patients, p=[0.20, 0.80])
df.loc[hypotension_mask, "sbp"] = np.random.uniform(70, 96.9, hypotension_mask.sum())

# Heart rate: Add tachycardia cases
tachycardia_mask = np.random.choice([True, False], n_patients, p=[0.20, 0.80])
df.loc[tachycardia_mask, "heart_rate"] = np.random.uniform(101.1, 140, tachycardia_mask.sum())

# Age: Cap at 100 and ensure minimum 18
df["age"] = df["age"].clip(18, 100)

# Comorbidities: Higher prevalence in older patients
df.loc[df["age"] > 65, "comorbidities"] = np.random.choice(
    [0, 1], len(df[df["age"] > 65]), p=[0.40, 0.60]
)

# Round continuous variables
df["temperature"] = df["temperature"].round(1)
df["spo2"] = df["spo2"].round(1)
df["sbp"] = df["sbp"].round(0)
df["heart_rate"] = df["heart_rate"].round(0)
df["age"] = df["age"].round(0)

# Assign labels based on decision logic
df["label"] = 0  # Default: Safe to Discharge (0)
# Needs ER Evaluation (1) if any critical condition
er_conditions = (
    (df["temperature"] >= 39.4) |
    (df["spo2"] < 92) |
    (df["sbp"] < 97) |
    (df["heart_rate"] > 101) |
    (df["chest_pain"] == 1) |
    (df["dyspnea"] == 1) |
    (df["syncope"] == 1)
)
df.loc[er_conditions, "label"] = 1

# Amplify risk for age > 65 or comorbidities (e.g., borderline cases)
borderline_conditions = (
    ((df["temperature"] >= 38.5) & (df["temperature"] < 39.4)) |
    ((df["spo2"] >= 92) & (df["spo2"] < 94)) |
    ((df["sbp"] >= 97) & (df["sbp"] < 100)) |
    ((df["heart_rate"] >= 95) & (df["heart_rate"] <= 101))
) & ((df["age"] > 65) | (df["comorbidities"] == 1))
df.loc[borderline_conditions, "label"] = 1

# Convert label to string for clarity
df["label"] = df["label"].map({1: "Needs ER Evaluation", 0: "Safe to Discharge"})

# Save to CSV
df.to_csv("/Users/nn/PycharmProjects/triage-beta-tool/data/synthetic_data.csv", index=False)

# Preview
print(df.head())
print(df["label"].value_counts())

# validation
print(df.describe())
print(df["label"].value_counts())
print(df[df["age"] > 65]["comorbidities"].mean())  # Should be ~0.6