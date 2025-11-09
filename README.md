# Open-Pit Mine Rockfall Risk Assessment via Data Analytics & Visualization

_A Project for Data Analytics & Visualization (DAV) Course)_

This project addresses problem statement **SIH25071** from the Smart India Hackathon and focuses on predicting rockfall risk in open-pit mines using a high-fidelity synthetic dataset informed by real-world Kaggle statistics.

---

## 1. Project Overview

Objective: develop a robust risk assessment system to classify rockfall risk into four categories: **Low, Medium, High, Critical**, using engineered synthetic data and machine learning (final model: **XGBClassifier**).

Key points:
- High-fidelity synthetic data modeled on real rainfall and seismic statistics.
- Displacement is engineered as the strongest predictor.
- Emphasis on realistic class imbalance and model interpretability.

### Data Strategy (DAV Method)
- Analyze statistical properties of public datasets (rainfall zero-inflation, seismic long-tail).
- Generate a unified 20,000-sample dataset using those statistics.
- Engineer logical correlations so displacement, water pressure, seismic activity, and vibration jointly drive risk.

Kaggle driver sources:
- Rainfall: https://www.kaggle.com/datasets/sujithmandala/rainfall-dataset-for-simple-time-series-analysis
- Seismic: https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023

---

## 2. Core Objectives

- Create a high-fidelity synthetic dataset from statistical drivers.
- Perform comprehensive EDA (distributions, correlation heatmaps, outlier analysis).
- Preprocess correctly (StandardScaler, LabelEncoder) and handle imbalance (class_weight or sampling).
- Evaluate models and select XGBClassifier based on targeted metrics (recall for Critical class, precision, confusion matrix).
- Use SHAP for model interpretation and feature importance.

---

## 3. Dataset Composition

- File: `rockfall_synthetic_data.csv` (20,000 samples)
- Features:
    - Displacement (mm) — primary predictor
    - Joint Water Pressure (kPa)
    - Seismic Activity (Magnitude)
    - Vibration Level (score)
    - Rainfall (mm/24h)
- Engineered correlations: displacement derived from water pressure, seismic, and vibration.
- Risk distribution (approximate): Low 50.5%, Medium 40.5%, High 7.3%, Critical 1.7%.

---

## 4. Technology Stack

- Python 3.13+
- pandas, numpy, scikit-learn, xgboost
- matplotlib, seaborn, plotly
- SHAP for interpretability
- Streamlit for deployment
- Jupyter Notebook for development

---

## 5. Setup Instructions

### Kaggle API (required)
1. Create a Kaggle account and generate an API token (`kaggle.json`) from Account → API.
2. Place `kaggle.json`:
     - Windows:
         ```powershell
         mkdir C:\Users\<YourUsername>\.kaggle
         Move-Item ~/Downloads/kaggle.json C:\Users\<YourUsername>\.kaggle\kaggle.json
         ```
     - Mac/Linux:
         ```bash
         mkdir -p ~/.kaggle
         mv ~/Downloads/kaggle.json ~/.kaggle/
         chmod 600 ~/.kaggle/kaggle.json
         ```
3. Accept dataset terms on Kaggle by visiting the two dataset pages and clicking "Download".

---

## 6. How to Run

1. Clone repository:
     ```powershell
     git clone [repository-link]
     cd open-pit-mine-rockfall-prediction
     ```
2. Create and activate virtual environment:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1   # Windows PowerShell
     # or
     source .venv/bin/activate    # macOS/Linux
     ```
3. Install dependencies:
     ```powershell
     pip install -r requirements.txt
     ```
4. (Optional) Register Jupyter kernel:
     ```powershell
     python -m ipykernel install --user --name=rockfall-venv --display-name="Python (rockfall-venv)"
     ```
5. Run notebooks in order (notebooks/):
     1. `01_data_sourcing_and_generation.ipynb`  ← downloads Kaggle drivers
     2. `02_exploratory_data_analysis.ipynb`
     3. `03_preprocessing_and_modelling.ipynb`
     4. `04_model_interpretation_and_results.ipynb`
6. Run Streamlit app (after notebooks complete):
     ```powershell
     streamlit run app.py
     ```

---

## 7. Notes & Best Practices

- Use stratified 80/20 train-test split for reliable evaluation.
- Monitor metrics per class (especially recall for Critical).
- Use SHAP explanations to validate that displacement and engineered relationships drive model decisions.

---

## 8. Acknowledgement

This project used assistance from a large language model to help design the data strategy and documentation.

---

If you need further condensation, detail adjustments, or a specific README section reworded, specify which part.
