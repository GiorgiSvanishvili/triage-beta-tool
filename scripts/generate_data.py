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
    'SpO2': np.random.normal(97, 2, n_samples).clip(80, 100),
    'blood_pressure': np.random.normal(120, 15, n_samples).clip(80, 180),
    'temperature': np.random.normal(37, 0.5, n_samples).clip(35, 40),
    'chest_pain': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'shortness_of_breath': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'heart_disease': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'age': np.random.randint(18, 90, n_samples),
    'unilateral_weakness': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'trouble_speaking': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'trouble_walking': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
    'syncope': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate ER_needed with adjusted weights
er_probs = (
    0.75 * (df['SpO2'] < 92) +  # Slightly reduced
    0.65 * df['chest_pain'] +   # Slightly reduced
    0.55 * df['shortness_of_breath'] +
    0.45 * df['heart_disease'] +
    0.85 * df['unilateral_weakness'] +  # Slightly reduced
    0.85 * df['trouble_speaking'] +
    0.85 * df['trouble_walking'] +
    0.85 * df['syncope'] +
    0.45 * (df['blood_pressure'] > 140) +
    0.35 * (df['temperature'] > 38)
).clip(0, 1)

df['ER_needed'] = np.random.binomial(1, er_probs)

# Save to CSV
os.makedirs(os.path.join(SCRIPT_DIR, "..", "data"), exist_ok=True)
df.to_csv(os.path.join(SCRIPT_DIR, "..", "data", "synthetic_data.csv"), index=False)
print(f"Data saved to {os.path.join(SCRIPT_DIR, '..', 'data', 'synthetic_data.csv')}")