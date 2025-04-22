import pandas as pd
import numpy as np
import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define output path relative to project root
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "synthetic_data.csv")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data (based on your original logic)
n_samples = 5000
df = pd.DataFrame({
    "temperature": np.random.normal(36.8, 0.5, n_samples),
    "spo2": np.random.uniform(80, 100, n_samples),
    "sbp": np.random.normal(120, 15, n_samples),
    "heart_rate": np.random.normal(80, 10, n_samples),
    "chest_pain": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    "dyspnea": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    "syncope": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    "age": np.random.randint(18, 100, n_samples),
    "comorbidities": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# Label data based on clinical thresholds
df["label"] = 0  # Default: Safe to Discharge
# ER conditions
er_conditions = (
    (df["temperature"] >= 38.5) |
    (df["spo2"] < 92) |
    (df["sbp"] < 97) |
    (df["heart_rate"] > 101) |
    (df["chest_pain"] == 1) |
    (df["dyspnea"] == 1) |
    (df["syncope"] == 1)
)
df.loc[er_conditions, "label"] = 1
# Borderline conditions
borderline_conditions = (
    ((df["temperature"] >= 38.5) & (df["temperature"] < 39.4)) |
    ((df["spo2"] >= 92) & (df["spo2"] < 94)) |
    ((df["sbp"] >= 97) & (df["sbp"] < 100)) |
    ((df["heart_rate"] >= 95) & (df["heart_rate"] <= 101))
) & ((df["age"] > 65) | (df["comorbidities"] == 1))
df.loc[borderline_conditions & (df["label"] == 0), "label"] = 1

# Convert labels to strings
df["label"] = df["label"].map({0: "Safe to Discharge", 1: "Needs ER Evaluation"})

# Ensure data directory exists
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# Save data
df.to_csv(DATA_PATH, index=False)
print(f"Data saved to {DATA_PATH}")