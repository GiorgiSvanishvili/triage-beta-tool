# Triage Tool Beta

## What It Is

The **Triage Tool Beta** is a web app that helps healthcare providers decide if a patient needs **Emergency Room (ER) care** or can be **safely discharged**. It analyzes 9 patient features, like oxygen levels, blood pressure, and symptoms (e.g., chest pain), using machine learning.

**Benefits**:
- **Safe**: Detects 95.1% of ER cases, ensuring critical patients are identified.
- **Accurate**: Correctly predicts 92.3% of cases.
- **Easy to Use**: Simple web interface for entering patient data.

This beta uses synthetic data for testing. It predicts based on clinical rules (e.g., low oxygen or chest pain = ER). Future versions will use real patient data.

## How to Run

Run the tool on your computer with these steps.

### Prerequisites
- Python 3.7–3.10.
- pip (included with Python).
- Internet to install libraries.

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/GiorgiSvanishvili/triage-beta-tool
   cd triage-beta-tool

2. **Install Dependencies**:
   ```bash
    pip install -r requirements.txt
3. **Run the Web App**:
    ```bash
    streamlit run app/app.py
