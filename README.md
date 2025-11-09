# Open-Pit Mine Rockfall Risk Assessment via Data Analytics & Visualization

_A Project for Data Analytics & Visualization (DAV) Course_

This project addresses problem statement **SIH25071** from the Smart India Hackathon, adapted for the Data Analytics & Visualization (DAV) course (G5AD21DAV) at Rashtriya Raksha University. It focuses specifically on **rockfall prediction in open-pit mining operations** using advanced data analytics and machine learning.

---

## 1. Project Overview

The objective is to develop a robust risk assessment system for predicting rockfall events in open-pit mines using data analytics and machine learning classification models. This project uses a meticulously engineered **high-fidelity synthetic dataset** where critical features like rainfall and seismic activity are statistically modeled on **real-world Kaggle distributions** to ensure realism.

By analyzing geological and environmental factors, we classify rockfall risk into four distinct categories: **Low, Medium, High, and Critical**. This project demonstrates the practical application of data mining, feature engineering, **robust ensemble modeling (XGBoost)**, and visualization for enhancing mining safety and operational decision-making.

### Our Data Strategy: High-Fidelity Synthetic Modeling

While direct "rockfall event" datasets are proprietary and rarely shared, we use a superior approach: **High-Fidelity Synthetic Modeling informed by Real Data Statistics.**

#### Synthetic Mine Sensor Data
We generate synthetic data that accurately reflects:

1. **Real Mine Monitoring Systems:** Our features mirror actual slope stability monitoring:
   - **Displacement (mm):** The single strongest predictor of risk.
   - **Joint Water Pressure (kPa):** Pore pressure in rock joints/fractures (positively correlated with rainfall).
   - **Seismic Activity (Magnitude):** Micro-seismic monitoring for ground instability (statistically modeled on global earthquake data).
   - **Vibration Level (Score):** Blast-induced and machinery vibration tracking (positively correlated with seismic activity).
   - **Rainfall (mm/24h):** Precipitation infiltration analysis (statistically modeled on real weather data).

2. **Statistical Realism (The DAV Method):** Instead of merging unrelated files, we analyzed the real-world statistical properties (e.g., zero-inflation for rain, long-tail for seismic) from public Kaggle datasets and used those properties to build our unique dataset.

---

## 2. Core Objectives

-   **Data Generation & Sourcing:** Create a high-fidelity synthetic dataset by statistically modeling external factors using data from **Kaggle Rainfall and Seismic datasets**.
-   **Exploratory Data Analysis:** Perform comprehensive EDA including correlation heatmaps, distribution analysis, outlier detection, and visually confirm the engineered feature-risk relationships.
-   **Data Pre-processing & Modeling:** Apply correct preprocessing (StandardScaler, LabelEncoder) and employ a **class_weight='balanced'** strategy to tackle the realistic **imbalanced dataset** problem.
-   **Advanced Predictive Modeling:** Train a ladder of models and select the **XGBClassifier** as the final, most robust model for deployment.
-   **Model Evaluation & Interpretation:** Generate comprehensive performance metrics (Confusion Matrix, Recall, Precision) and use feature importance to **justify the final model choice** over the potentially overfit Random Forest.

---

## 3. Dataset Composition

Our final dataset (`rockfall_synthetic_data.csv`, 20,000 samples) is characterized by:

### Features & Logic
- **5 Core Geotechnical Features** (as listed above).
- **Engineered Correlations:** Displacement is a function of Water Pressure, Seismic Activity, and Vibration Level.
- **Risk Distribution:** **Realistic Imbalance:** The distribution is approximately **50.5% Low, 40.5% Medium, 7.3% High, and 1.7% Critical**. (This is a major correction from the old, flawed "balanced" approach).

### Real-World Statistical Drivers (Kaggle Sources)
- **Rainfall Driver:** [Rainfall Dataset for Simple Time Series Analysis](https://www.kaggle.com/datasets/sujithmandala/rainfall-dataset-for-simple-time-series-analysis)
- **Seismic Driver:** [All the Earthquakes Dataset : from 1990-2023](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023)

### Data Processing Summary
- **Combined Sample Size:** 20,000 observations for robust ML training.
- **Risk Categories:** 4-class classification (Low, Medium, High, Critical).
- **Train-Test Split:** 80/20 **stratified** split for balanced evaluation.

---

## 4. Technology Stack

-   **Language:** Python 3.13+
-   **Data Science & ML:** pandas, numpy, scikit-learn, **xgboost**
-   **Data Visualization:** matplotlib, seaborn, plotly (interactive dashboards)
-   **Model Interpretability:** SHAP (SHapley Additive exPlanations)
-   **Data Source:** Kaggle API for statistical drivers
-   **Deployment:** Streamlit
-   **Development Environment:** Jupyter Notebook

---

## 5. Setup Instructions

### IMPORTANT: Kaggle API Setup (Required!)

This project downloads real **Rainfall and Seismic driver data** from Kaggle. You **must** set up Kaggle API credentials before running the notebooks.

#### Step 1 & 2: Get API Token and Place `kaggle.json`
*(Instructions are correct and unchanged)*

#### Step 3: Accept Dataset Terms on Kaggle
**IMPORTANT:** Before running the notebooks, you must accept the dataset's terms for the two driver datasets:
1. Visit: [Rainfall Dataset for Simple Time Series Analysis](https://www.kaggle.com/datasets/sujithmandala/rainfall-dataset-for-simple-time-series-analysis)
2. Visit: [All the Earthquakes Dataset : from 1990-2023](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023)
3. Click the **"Download"** button on both pages (this accepts the terms).

---

### How to Run the Project

*(Setup Instructions for Environment/Dependencies remain correct)*

7.  **Run Notebooks Sequentially:**
    Execute in the `/notebooks` directory in this **new, clean** order:
    1.  `01_data_sourcing_and_generation.ipynb` ← Downloads Kaggle driver data here!
    2.  `02_exploratory_data_analysis.ipynb`
    3.  `03_preprocessing_and_modelling.ipynb`
    4.  `04_model_interpretation_and_results.ipynb`

    *The final notebook displays the Confusion Matrix, Feature Importance, and the analysis justifying the XGBoost model selection.*

---

## 6. Acknowledgement

This project was developed with assistance from Gemini, a large language model by Google, which helped generate the high-fidelity synthetic dataset, re-engineer the professional workflow, and validate the model interpretation.