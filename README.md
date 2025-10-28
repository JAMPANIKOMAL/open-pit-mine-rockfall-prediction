# Rockfall Risk Assessment via Data Analytics & Visualization

_A Project for Data Analytics & Visualization (DAV) Course_

This project explores the problem statement SIH25071 from the Smart India Hackathon, adapted for the Data Analytics & Visualization (DAV) course (G5AD21DAV) at Rashtriya Raksha University.

---

## 1. Project Overview

The objective is to utilize data analytics techniques and machine learning—specifically classification models—to predict the risk level of rockfall events in open-pit mines. This project combines **synthetic sensor data** (seismic activity, vibration, water pressure, displacement, rainfall) with **real-world landslide data from Kaggle** (temperature, humidity, moisture, slope angle) to create a comprehensive dataset. By analyzing these geological and environmental factors, we aim to classify risk into distinct categories (Low, Medium, High, Critical). This project demonstrates the application of data mining, integration, and visualization principles for enhancing mining safety.

### Why Synthetic Data?

While real-world rockfall sensor data from mines is ideal, it faces several practical challenges:

1. **Proprietary & Confidential:** Mining companies treat sensor data as proprietary information due to safety and competitive reasons, making it rarely publicly available.

2. **Safety & Liability Concerns:** Real rockfall incidents involve fatalities and legal implications, making companies hesitant to share incident data.

3. **Lack of Public Datasets:** Unlike general landslides, specific "rockfall in open-pit mines with continuous sensor monitoring" datasets are virtually non-existent on platforms like Kaggle.

4. **Academic Demonstration:** For a DAV course project, synthetic data allows us to:
   - **Control feature relationships** to demonstrate clear risk patterns
   - **Ensure balanced classes** for better model training
   - **Simulate critical scenarios** that might be rare in real data
   - **Validate our modeling pipeline** before applying to real-world data

By **combining synthetic rockfall data with real landslide data from Kaggle**, we create a rich, diverse dataset that:
- Includes mine-specific sensors (seismic, vibration, water pressure) via synthesis
- Incorporates real environmental patterns (temperature, humidity, slope) from Kaggle
- Provides ~15,000 samples for robust model training
- Demonstrates data integration skills crucial for real-world analytics

---

## 2. Core Objectives

-   **Data Generation & Integration:** Create synthetic rockfall sensor data AND integrate real-world landslide data from Kaggle to build a comprehensive, diverse dataset.
-   **Exploratory Data Analysis:** Perform in-depth analysis including correlation analysis, distribution visualization, outlier detection, and missing value assessment.
-   **Data Pre-processing:** Handle missing values through imputation, encode categorical features (risk levels), and split data for model training and evaluation.
-   **Predictive Modeling:** Apply and evaluate 6+ classification algorithms (Logistic Regression, Random Forest, SVM, Decision Tree, Naive Bayes, KNN) with hyperparameter tuning to identify the best-performing model.
-   **Data Visualization & Interpretation:** Create comprehensive visualizations including confusion matrices, ROC curves, precision-recall curves, and feature importance plots to evaluate and interpret model performance.

---

## 3. Dataset Composition

Our integrated dataset (~15,000 samples) combines two complementary data sources:

### Synthetic Rockfall Sensor Data (10,000 samples)
**Purpose:** Simulate mine-specific sensor readings not available in public datasets

**Features:**
- `seismic_activity` - Ground vibration from geological instability (Richter scale equivalent)
- `vibration_level` - Equipment/blasting vibration readings (mm/s)
- `joint_water_pressure` - Water pressure in rock joints (kPa)
- `displacement_mm` - Rock face displacement measurements (mm)
- `rainfall_mm` - Precipitation levels (mm)

**Risk Distribution:** Balanced across Low (25%), Medium (25%), High (25%), Critical (25%)

### Real Kaggle Landslide Data (5,000 samples)
**Source:** [Landslide Dataset for Classification](https://www.kaggle.com/datasets/snehilmathur/landslide-dataset-for-classification)

**Features:**
- `Temperature` - Ambient temperature (°C)
- `Humidity` - Atmospheric humidity (%)
- `Rain` - Rainfall intensity (mm)
- `Moisture` - Soil moisture content (%)
- `Slope_Angle` - Terrain slope angle (degrees)
- `Soil_Type` - Geological composition (Clay, Sandy, Loamy, Rocky)

**Risk Distribution:** Real-world patterns with class imbalance (standardized to our categories)

### Data Integration Strategy
- **Missing value handling:** Median imputation for features unique to each source
- **Risk category alignment:** Standardized to 4 levels (Low, Medium, High, Critical)
- **Feature diversity:** 14+ features covering geological, environmental, and geotechnical factors

---

## 4. Technology Stack

-   **Language:** Python 3
-   **Data Science & ML:** pandas, numpy, scikit-learn
-   **Data Visualization:** matplotlib, seaborn
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

#### Step 3: Verify Setup
Your `kaggle.json` file should look like this:
```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key_here"
}
```

---

### How to Run the Project

1.  **Clone Repository:**
    ```powershell
    git clone <repository-url>
    cd rockfall-prediction-system
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
- **Solution:** Go to [this dataset](https://www.kaggle.com/datasets/snehilmathur/landslide-dataset-for-classification) and click "Download" to accept terms, then re-run

**Problem:** Notebook hangs during download
- **Solution:** Check your internet connection. The dataset is ~1MB and should download quickly

---

## Acknowledgement

This project was developed with assistance from Gemini, a large language model by Google, which helped generate the synthetic dataset, structure the workflow, and create documentation.
