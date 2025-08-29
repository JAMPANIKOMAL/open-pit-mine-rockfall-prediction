# AI-Based Rockfall Prediction System

A data science project to predict rockfall risk in open-pit mines, inspired by the Smart India Hackathon (SIH) problem statement.

## Project Vision

This project addresses SIH25071: "AI-Based Rockfall Prediction and Alert System for Open-Pit Mines." Mining safety is crucial, and AI-driven prediction of geological instability can help prevent accidents and save lives.

The objective is to build a machine learning model that analyzes simulated sensor data to classify the immediate risk of a rockfall event. This serves as a proof-of-concept for a real-time monitoring and alert system.

The workflow is notebook-based, similar to the biodiversity-edna-analysis project.

## Core Features

- **Realistic Synthetic Data:** Custom-generated dataset simulating real-world geological and environmental sensor readings.
- **Predictive Modeling:** Machine learning models classify rockfall risk into categories (Low, Medium, High, Critical).
- **Feature Importance Analysis:** Identifies key sensor readings that predict rockfall risk.
- **Clear Visualizations:** Intuitive plots and matrices to present model performance and data insights.

## Technology Stack

- **Language:** Python 3
- **Data Science & ML:** pandas, scikit-learn, numpy
- **Data Visualization:** matplotlib, seaborn
- **Development Environment:** Jupyter Notebook

## Project Structure

The analysis is organized into four sequential notebooks:

1. **01_data_generation_and_exploration.ipynb:** Create and explore the synthetic dataset.
2. **02_data_preprocessing.ipynb:** Clean, encode, and prepare data for modeling.
3. **03_model_development.ipynb:** Train and evaluate classification models.
4. **04_results_visualization.ipynb:** Visualize results, including accuracy and key predictors.

## How to Run

1. Install Python and required libraries:  
     ```
     pip install -r requirements.txt
     ```
2. Launch Jupyter Notebook or JupyterLab.
3. Open and run the notebooks in order, starting with `01_...`.

## Acknowledgement

This project was developed with assistance from Gemini, a large language model by Google, which helped generate the synthetic dataset, structure the workflow, and create documentation.
