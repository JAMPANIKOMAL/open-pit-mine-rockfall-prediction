# Rockfall Risk Assessment via Data Analytics & Visualization

_A Project for Data Analytics & Visualization (DAV) Course_

This project explores the problem statement SIH25071 from the Smart India Hackathon, adapted for the Data Analytics & Visualization (DAV) course (G5AD21DAV) at Rashtriya Raksha University.

---

## 1. Project Overview

The objective is to utilize data analytics techniques and machine learning—specifically classification models—to predict the risk level of rockfall events in open-pit mines. This project combines **synthetic sensor data** (seismic activity, vibration, water pressure, displacement, rainfall) with **real-world landslide data from Kaggle** (temperature, humidity, moisture, slope angle) to create a comprehensive dataset. By analyzing these geological and environmental factors, we aim to classify risk into distinct categories (Low, Medium, High, Critical). This project demonstrates the application of data mining, integration, and visualization principles for enhancing mining safety.

---

## 2. Core Objectives

-   **Data Generation & Integration:** Create synthetic rockfall sensor data AND integrate real-world landslide data from Kaggle to build a comprehensive, diverse dataset.
-   **Exploratory Data Analysis:** Perform in-depth analysis including correlation analysis, distribution visualization, outlier detection, and missing value assessment.
-   **Data Pre-processing:** Handle missing values through imputation, encode categorical features (risk levels), and split data for model training and evaluation.
-   **Predictive Modeling:** Apply and evaluate 6+ classification algorithms (Logistic Regression, Random Forest, SVM, Decision Tree, Naive Bayes, KNN) with hyperparameter tuning to identify the best-performing model.
-   **Data Visualization & Interpretation:** Create comprehensive visualizations including confusion matrices, ROC curves, precision-recall curves, and feature importance plots to evaluate and interpret model performance.

---

## 3. Technology Stack

-   **Language:** Python 3
-   **Data Science & ML:** pandas, numpy, scikit-learn
-   **Data Visualization:** matplotlib, seaborn
-   **Development Environment:** Jupyter Notebook

---

## 4. How to Run

1.  **Environment Setup:**
    * Clone the repository.
    * Create and activate a Python virtual environment:
        ```sh
        python -m venv venv
        # Activate (example for bash/zsh):
        source venv/bin/activate
        # Or (example for Windows cmd):
        venv\Scripts\activate
        ```
    * Install dependencies:
        ```sh
        pip install -r requirements.txt
        ```
2.  **Launch Jupyter:** Start Jupyter Notebook or JupyterLab from your activated environment.
3.  **Run Notebooks:** Execute the notebooks sequentially within the `/notebooks` directory:
    1.  `01_data_generation_and_exploration.ipynb`
    2.  `02_data_preprocessing.ipynb`
    3.  `03_model_development.ipynb`
    4.  `04_results_visualization.ipynb`

    *The final notebook (`04_...`) displays the performance evaluation (like the confusion matrix) for the best model chosen in notebook 03.*

---

## Acknowledgement

This project was developed with assistance from Gemini, a large language model by Google, which helped generate the synthetic dataset, structure the workflow, and create documentation.
