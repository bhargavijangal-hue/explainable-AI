# ClinExplain AI — Explainable Clinical Decision Support Website

A runnable Streamlit educational prototype demonstrating:
- A machine-learning classifier trained on scikit-learn's built-in Breast Cancer Wisconsin (Diagnostic) dataset
- LIME local feature explanations
- A simple quantile-based counterfactual search
- Model evaluation metrics and a dashboard
- CSV download of the current input feature values

## Important safety note
This is a coursework / research prototype, **not a medical device**. It is not validated for clinical use and must not be used to diagnose, treat, or rule out disease. The counterfactual examples are hypothetical model inputs, not medical advice or causal treatment recommendations.

## Requirements
- Python 3.10 or newer recommended
- Internet access for the initial package installation

## Run locally (Windows)
1. Extract the ZIP file.
2. Open Command Prompt or PowerShell in the extracted `explainable_ai_clinical_website` folder.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
4. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the website:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Run on macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Pages
- **Dashboard:** model overview, sample label balance, workflow and limitations.
- **Diagnosis & Explanation:** choose an example, edit dataset feature values, get a prediction, LIME explanation, counterfactual what-if examples, and download input values as CSV.
- **Model Evaluation:** held-out accuracy, ROC-AUC, confusion matrix, and classification report.
- **About Project:** project purpose, stack and future work.

## Design / implementation notes
- The model trains automatically the first time the app runs and is cached for the session.
- The app does not store patient data or connect to any hospital system.
- The built-in dataset contains numeric tumour-measurement features. It is not a general patient vital-sign dataset.
- LIME explanations are local approximations and can vary with settings.
- The counterfactual function is an intentionally simple demonstration. For stronger research results, compare with a dedicated method such as DiCE and add clinically justified feature constraints.

## Suggested project extensions
1. Allow uploading a CSV with the same feature schema (validate it carefully).
2. Compare Logistic Regression, Random Forest and Gradient Boosting.
3. Add global feature importance and calibration plots.
4. Add a dedicated DiCE counterfactual implementation.
5. Add automated tests and external validation on an independent dataset.
6. Deploy through Streamlit Community Cloud or another approved hosting provider only after reviewing privacy, security and intended-use requirements.
