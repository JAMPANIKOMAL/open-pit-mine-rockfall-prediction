# Open-Pit Mine Rockfall Risk Assessment via Data Analytics & Visualization

_A Project for Data Analytics & Visualization (DAV) Course_

This project addresses problem statement **SIH25071** from the Smart India Hackathon, adapted for the Data Analytics & Visualization (DAV) course (G5AD21DAV) at Rashtriya Raksha University. It focuses specifically on **rockfall prediction in open-pit mining operations** using advanced data analytics and machine learning.

---

## 1. Project Overview

The objective is to develop a robust risk assessment system for predicting rockfall events in open-pit mines using data analytics and machine learning classification models. This project combines **synthetic mine-specific sensor data** (seismic activity, vibration, water pressure, displacement, rainfall) with **real-world industrial mining process data from Kaggle** to create a comprehensive dataset that mirrors actual mine monitoring systems.

By analyzing geological, environmental, and operational factors, we classify rockfall risk into four distinct categories: **Low, Medium, High, and Critical**. This project demonstrates the practical application of data mining, feature engineering, ensemble modeling, and interactive visualization for enhancing mining safety and operational decision-making.

### Why This Hybrid Data Approach?

This project employs a **hybrid data strategy** combining synthetic mine monitoring data with real industrial mining sensor data:

#### Synthetic Rockfall Sensor Data
While direct "rockfall event" datasets are proprietary and rarely shared by mining companies due to safety/liability concerns, we generate synthetic data that accurately reflects:

1. **Real Mine Monitoring Systems:** Our features mirror actual slope stability monitoring:
   - **Seismic Activity:** Micro-seismic monitoring is standard in large open-pit mines
   - **Vibration Level:** Blast-induced and machinery vibration tracking
   - **Joint Water Pressure:** Pore pressure monitoring (primary rockfall trigger)
   - **Displacement:** Prism/GPS/InSAR systems for millimeter-level detection
   - **Rainfall:** Precipitation infiltration analysis

2. **Controlled Risk Patterns:** Synthetic generation allows us to:
   - Demonstrate clear feature-risk relationships for academic learning
   - Ensure balanced risk categories for fair model evaluation
   - Simulate rare critical scenarios often underrepresented in real data

#### Real Industrial Mining Data (Kaggle)
We integrate the **Quality Prediction in Mining Process** dataset (430 votes, 19K+ downloads, 1.0 usability):
- **Source:** [edumagalhaes/quality-prediction-in-a-mining-process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
- **Real industrial sensor data** from flotation plant operations (continuous monitoring)
- **Validates our approach:** Shows real mining processes exhibit similar sensor patterns
- **Adds diversity:** Includes operational parameters like iron concentrate, silica levels

This combination provides:
**Academic rigor** with real industrial data  
**Mine-specific focus** unlike general landslide datasets  
**Comprehensive features** covering geological + operational factors  
**Large sample size** for robust machine learning (20,000+ samples)

---

## 2. Core Objectives

-   **Data Generation & Integration:** Generate synthetic mine-specific sensor data AND integrate real industrial mining process data from Kaggle's highest-rated mining dataset.
-   **Exploratory Data Analysis:** Perform comprehensive EDA including correlation heatmaps, distribution analysis, outlier detection, statistical summaries, and feature relationship visualization.
-   **Data Pre-processing:** Handle missing values via median imputation, encode categorical risk levels, perform feature scaling, and create stratified train-test splits.
-   **Advanced Predictive Modeling:** Apply and compare 6+ classification algorithms including:
    - Traditional ML: Logistic Regression, Decision Tree, Naive Bayes, KNN
    - Ensemble Methods: Random Forest, Gradient Boosting (XGBoost/LightGBM)
    - Support Vector Machines with kernel optimization
    - Hyperparameter tuning via GridSearchCV with cross-validation
-   **Model Evaluation & Visualization:** Generate comprehensive performance metrics:
    - Confusion matrices for multi-class evaluation
    - ROC curves and AUC scores for each risk category
    - Precision-Recall curves for imbalanced scenarios
    - Feature importance rankings (tree-based & SHAP values)
    - Model comparison dashboards

---

## 3. Dataset Composition

Our integrated dataset (20,000+ samples) combines two complementary data sources:

### Synthetic Rockfall Sensor Data (10,000 samples)
**Purpose:** Simulate mine-specific slope monitoring sensor readings

**Features:**
- `seismic_activity` - Micro-seismic monitoring for ground instability (Richter equivalent)
- `vibration_level` - Blast-induced and machinery vibration (mm/s)
- `joint_water_pressure` - Pore pressure in rock joints/fractures (kPa) - **PRIMARY FAILURE TRIGGER**
- `displacement_mm` - Slope displacement via prism/GPS monitoring (mm)
- `rainfall_mm` - Precipitation infiltration analysis (mm)

**Risk Distribution:** Balanced across Low (25%), Medium (25%), High (25%), Critical (25%)

**Validation:** Features match industry-standard slope stability monitoring systems (SIH25071)

### Real Industrial Mining Data (10,000+ samples)
**Source:** [Quality Prediction in Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)  
**Rating:** 430 votes, 19,694 downloads, 1.0 usability score ⭐

**Features from Flotation Plant Monitoring:**
- `% Iron Concentrate` - Iron content in final concentrate
- `% Silica Concentrate` - Silica impurity levels
- `Starch Flow` - Flotation reagent dosage (m³/h)
- `Amina Flow` - Collector reagent flow (m³/h)
- `Ore Pulp Flow` - Feed rate (t/h)
- `Ore Pulp pH` - Chemical environment
- `Ore Pulp Density` - Slurry concentration (kg/cm³)
- `Flotation Column Levels` - Process control parameters
- `Air Flow` - Bubble generation rates

**Relevance to Rockfall Prediction:**
- Demonstrates **continuous sensor monitoring** approach used in mines
- **Operational parameters** show how mining processes affect slope stability
- **Time-series patterns** validate predictive modeling for mining safety

### Data Integration & Preprocessing
- **Combined Sample Size:** 20,000+ observations for robust ML training
- **Missing Value Handling:** Median imputation with SimpleImputer
- **Feature Scaling:** Standardization for distance-based algorithms (SVM, KNN)
- **Risk Categories:** 4-class classification (Low, Medium, High, Critical)
- **Train-Test Split:** 80/20 stratified split for balanced evaluation

---

## 4. Technology Stack

-   **Language:** Python 3.13+
-   **Data Science & ML:** pandas, numpy, scikit-learn, xgboost, lightgbm
-   **Data Visualization:** matplotlib, seaborn, plotly (interactive dashboards)
-   **Model Interpretability:** SHAP (SHapley Additive exPlanations)
-   **Data Source:** Kaggle API for industrial mining datasets
-   **Development Environment:** Jupyter Notebook

---

## 5. Setup Instructions

### IMPORTANT: Kaggle API Setup (Required!)

This project downloads real landslide data from Kaggle. You **must** set up Kaggle API credentials before running the notebooks.

#### Step 1: Get Your Kaggle API Token
1. Go to [Kaggle.com](https://www.kaggle.com/) and sign in (create account if needed)
2. Click on your profile picture (top right) → **Account**
3. Scroll down to **API** section
4. Click **Create New API Token**
5. This downloads a file called `kaggle.json` to your computer

#### Step 2: Place kaggle.json in the Correct Location
**Windows Users:**
```powershell
# Create .kaggle folder in your user directory
mkdir C:\Users\<YourUsername>\.kaggle

# Move the downloaded kaggle.json to this folder
# Final location: C:\Users\<YourUsername>\.kaggle\kaggle.json
```

**Mac/Linux Users:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 3: Accept Dataset Terms on Kaggle
**IMPORTANT:** Before running the notebooks, you must accept the dataset's terms:
1. Visit: https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process
2. Log in to your Kaggle account
3. Click the **"Download"** button (this accepts the terms)
4. You can close the page (download will be handled by the notebook)

This step is required only once. If you skip this, you'll get a "403 Forbidden" error.

---

### How to Run the Project

1.  **Clone Repository:**
    ```powershell
    git clone https://github.com/JAMPANIKOMAL/open-pit-mine-rockfall-prediction.git
    cd open-pit-mine-rockfall-prediction
    ```

2.  **Create Virtual Environment:**
    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Setup Jupyter Kernel:**
    ```powershell
    python -m ipykernel install --user --name=rockfall-venv --display-name="Python (rockfall-venv)"
    ```
    This makes your virtual environment available as a kernel in Jupyter.

5.  **Launch Jupyter Notebook:**
    ```powershell
    jupyter notebook
    ```

6.  **Select the Correct Kernel:**
    - Open any notebook (e.g., `01_data_generation_and_exploration.ipynb`)
    - Look at the **top right corner** of the notebook interface
    - Click on the kernel name (might show "Python 3")
    - Select **"Python (rockfall-venv)"** from the dropdown
    - Verify by running: `import sys; print(sys.executable)` - should show your `.venv` path

7.  **Run Notebooks Sequentially:**
    Execute in the `/notebooks` directory in this order:
    1.  `01_data_generation_and_exploration.ipynb` ← Downloads Kaggle data here!
    2.  `02_data_preprocessing.ipynb`
    3.  `03_model_development.ipynb`
    4.  `04_results_visualization.ipynb`

    *The final notebook displays comprehensive performance evaluation including confusion matrices, ROC curves, and feature importance for all 6 models.*

---

### Troubleshooting

**Problem:** `OSError: Could not find kaggle.json`
- **Solution:** Ensure `kaggle.json` is in `C:\Users\<YourUsername>\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux)

**Problem:** `403 Forbidden` when downloading dataset
- **Solution:** Go to [this dataset](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) and click "Download" to accept terms, then re-run

**Problem:** `ModuleNotFoundError: No module named 'cgi'` (Python 3.13+)
- **Solution:** This project now uses `kaggle` CLI instead of `opendatasets` for Python 3.13 compatibility. Make sure to run the first cell in each notebook to install dependencies.

**Problem:** Notebook hangs during download
- **Solution:** Check your internet connection. The dataset is ~1MB and should download quickly

---

## Acknowledgement

This project was developed with assistance from Gemini, a large language model by Google, which helped generate the synthetic dataset, structure the workflow, and create documentation.
