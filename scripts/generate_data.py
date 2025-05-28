import os
import numpy as np
import pandas as pd

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data (1000 patients)
n_samples = 1000

# Features
data = {
    'SpO2': np.clip(np.random.normal(98, 2, n_samples), 80, 100),
    'blood_pressure': np.clip(np.random.normal(120, 20, n_samples), 70, 200),
    'temperature': np.clip(np.random.normal(37, 0.5, n_samples), 35, 40),
    'chest_pain': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'shortness_of_breath': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'heart_disease': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'age': np.clip(np.ceil(np.random.normal(50, 15, n_samples)), 18, 100),
    'unilateral_weakness': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'trouble_speaking': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'trouble_walking': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
    'syncope': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'pulse': np.clip(np.random.normal(75, 15, n_samples), 40, 180),
    'blood_sugar': np.clip(np.random.normal(100, 30, n_samples), 50, 400),
    'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'mode_of_arrival': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
    'respiratory_rate': np.clip(np.random.normal(16, 4, n_samples), 8, 40),
    'altered_mental_status': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate needs_er based on finalized thresholds
df['needs_er'] = (
    (df['SpO2'] < 90) |
    (df['blood_pressure'] < 90) |
    (df['temperature'] > 38) |
    (df['chest_pain'] == 1) |
    (df['shortness_of_breath'] == 1) |
    (df['heart_disease'] == 1) |
    (df['unilateral_weakness'] == 1) |
    (df['trouble_speaking'] == 1) |
    (df['trouble_walking'] == 1) |
    (df['syncope'] == 1) |
    (df['pulse'] < 60) | (df['pulse'] > 100) |
    ((df['blood_sugar'] <= 70) | (df['blood_sugar'] >= 272)) |
    ((df['diabetes'] == 1) & ((df['blood_sugar'] <= 70) | (df['blood_sugar'] >= 200))) |
    (df['respiratory_rate'] < 8) | (df['respiratory_rate'] >= 25) |
    (df['altered_mental_status'] == 1) |
    # qSOFA: 2 or more of (altered mental status, respiratory rate ≥ 22, BP ≤ 100)
    (
        (
            (df['altered_mental_status'] == 1).astype(int) +
            (df['respiratory_rate'] >= 22).astype(int) +
            (df['blood_pressure'] <= 100).astype(int)
        ) >= 2
    )
).astype(int)

# Save to CSV
os.makedirs(os.path.join(SCRIPT_DIR, "..", "data"), exist_ok=True)
output_path = os.path.join(SCRIPT_DIR, "..", "data", "er_data.csv")
df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")