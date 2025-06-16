# Triage Tool

## What It Is

The **Triage Tool** was developed as a pre-triage application to assist individuals outside hospital environments in identifying potential emergency conditions, enabling timely intervention. It analyzes 17 patient features, like oxygen levels, blood pressure, and symptoms (e.g., chest pain), using machine learning.

**Benefits**:
- Predicts needs_er using a RandomForestClassifier trained on synthetic data.
- Detects more than 95% of ER cases, ensuring critical patients are identified.
- Supports input of vital signs, symptoms, and medical history.
- Simple web interface for entering patient data.
- Deployable via Streamlit for user-friendly interaction.

## How to Run

Try the app online or run it locally.

### Option 1: Use the Online App
- Visit: [https://triage-beta-tool.streamlit.app](https://triage-beta-tool.streamlit.app/)
- Enter data (e.g., SpO2 = 98%, no chest pain) with sliders/checkboxes.
- Click **Predict** to see the ER or Discharge result.

### Option 2: Run Locally
### Prerequisites
- Python 3.7–3.10.
- pip (included with Python).
- Internet to install libraries.

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/GiorgiSvanishvili/triage-beta-tool
   cd triage-beta-tool

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
      
3. **Run the Web App**:
    ```bash
    streamlit run app/app.py

Replace app.py with your Streamlit script path (e.g., /scripts/app.py).
The app will open in your default browser at http://localhost:8501.

## Usage
   
- Generate test data with generate_data.py.
- Train the model with train_model.py.
- Evaluate with evaluate_model.py to update metrics.txt.

## Documentation

  See Triage Tool Beta [Documentation](https://docs.google.com/document/d/1FniBsc5VBB5BXAKLZxAYxwkVUe46tqF7ZZ1omVzNRfQ/) for detailed information on aims, creation, logic, and results.
   
