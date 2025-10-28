"""
Open-Pit Mine Rockfall Risk Assessment System
Web Application using Streamlit

Author: JAMPANIKOMAL
Course: Data Analytics & Visualization (G5AD21DAV)
Institution: Rashtriya Raksha University
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Rockfall Risk Assessment",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Function to apply theme-specific CSS
def apply_theme_css(theme):
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#000000'
        secondary_bg = '#F5F5F5'
        border_color = '#CCCCCC'
        metric_bg = '#F0F0F0'
    else:  # dark
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        secondary_bg = '#262730'
        border_color = '#444444'
        metric_bg = '#1E1E1E'
    
    st.markdown(f"""
    <style>
    /* Override Streamlit default styles */
    .stApp {{
        background-color: {bg_color} !important;
    }}
    
    /* Main content area */
    .main .block-container {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {secondary_bg} !important;
    }}
    
    /* All text elements */
    p, span, div, label, h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
    }}
    
    /* Headers */
    .main-header {{
        font-size: 2.5rem;
        color: {text_color} !important;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid {text_color};
        padding-bottom: 10px;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {text_color} !important;
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    /* Risk level boxes */
    .risk-critical {{
        background-color: #000000;
        color: white;
        padding: 20px;
        border: 2px solid {border_color};
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }}
    .risk-high {{
        background-color: #333333;
        color: white;
        padding: 20px;
        border: 2px solid {border_color};
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }}
    .risk-medium {{
        background-color: #666666;
        color: white;
        padding: 20px;
        border: 2px solid {border_color};
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }}
    .risk-low {{
        background-color: #CCCCCC;
        color: black;
        padding: 20px;
        border: 2px solid {border_color};
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }}
    .metric-card {{
        background-color: {metric_bg};
        padding: 15px;
        border: 1px solid {border_color};
        border-left: 4px solid {text_color};
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)

# Load models and metadata
@st.cache_resource
def load_models():
    """Load trained models and metadata"""
    try:
        model_path = Path('models/best_model.pkl')
        metadata_path = Path('models/model_metadata.pkl')
        
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Try loading label encoder if it exists
        label_encoder = None
        if metadata.get('uses_encoded_labels', False):
            le_path = Path('models/label_encoder.pkl')
            if le_path.exists():
                with open(le_path, 'rb') as f:
                    label_encoder = pickle.load(f)
        
        return best_model, metadata, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure you have run notebooks 01-03 to generate the models.")
        return None, None, None

# Load test data for analysis
@st.cache_data
def load_test_data():
    """Load test data for model evaluation"""
    try:
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        return X_test, y_test
    except Exception as e:
        st.warning(f"Test data not available: {str(e)}")
        return None, None

# Main app
def main():
    # Apply theme CSS
    apply_theme_css(st.session_state.theme)
    
    # Header
    st.markdown('<div class="main-header">Open-Pit Mine Rockfall Risk Assessment System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Safety Monitoring for Mining Operations</div>', 
                unsafe_allow_html=True)
    
    # Load models
    best_model, metadata, label_encoder = load_models()
    
    if best_model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Theme toggle button at top of sidebar
    if st.sidebar.button("Toggle Light/Dark Mode"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()
    
    st.sidebar.markdown(f"**Current Theme:** {st.session_state.theme.capitalize()}")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Select Page", 
                            ["Home", 
                             "Risk Prediction", 
                             "Model Performance",
                             "About Project"])
    
    # Display model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Model")
    st.sidebar.info(f"""
    **Model:** {metadata.get('model_name', 'Unknown')}  
    **Accuracy:** {metadata.get('test_accuracy', 0)*100:.2f}%  
    **F1-Score:** {metadata.get('test_f1_score', 0)*100:.2f}%
    """)
    
    # Page routing
    if page == "Home":
        show_home_page(metadata)
    elif page == "Risk Prediction":
        show_prediction_page(best_model, metadata, label_encoder)
    elif page == "Model Performance":
        show_performance_page(best_model, metadata, label_encoder)
    elif page == "About Project":
        show_about_page()

def show_home_page(metadata):
    """Home page with project overview"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", f"{metadata.get('test_accuracy', 0)*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Categories", "4 Levels")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", metadata.get('model_name', 'Unknown'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project overview
    st.header("Project Overview")
    st.write("""
    This AI-powered system predicts rockfall risk levels in open-pit mining operations using 
    real-time sensor data. The system analyzes multiple environmental and geological parameters 
    to classify risk into four categories: **Low**, **Medium**, **High**, and **Critical**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Features")
        st.markdown("""
        - Real-time Risk Assessment: Instant predictions from sensor data
        - Multi-level Classification: 4-tier risk severity system
        - Advanced ML Models: XGBoost, LightGBM, ensemble methods
        - Hybrid Dataset: 20,000+ samples (synthetic + real mining data)
        - High Accuracy: State-of-the-art performance metrics
        """)
    
    with col2:
        st.subheader("Monitored Parameters")
        st.markdown("""
        - Seismic Activity: Ground vibrations and tremors
        - Vibration Sensors: Structural movement detection
        - Water Pressure: Pore pressure in rock formations
        - Ground Displacement: Slope deformation monitoring
        - Rainfall: Precipitation impact on stability
        """)
    
    st.markdown("---")
    
    # Risk level explanation
    st.header("Risk Level Guide")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="risk-low">LOW</div>', unsafe_allow_html=True)
        st.write("**Safe Conditions**")
        st.write("Normal operations")
    
    with col2:
        st.markdown('<div class="risk-medium">MEDIUM</div>', unsafe_allow_html=True)
        st.write("**Monitor Closely**")
        st.write("Increase surveillance")
    
    with col3:
        st.markdown('<div class="risk-high">HIGH</div>', unsafe_allow_html=True)
        st.write("**Action Required**")
        st.write("Restrict access zones")
    
    with col4:
        st.markdown('<div class="risk-critical">CRITICAL</div>', unsafe_allow_html=True)
        st.write("**Immediate Evacuation**")
        st.write("Emergency protocols")

def show_prediction_page(best_model, metadata, label_encoder):
    """Interactive prediction page"""
    
    st.header("Rockfall Risk Prediction")
    st.write("Enter sensor readings to get real-time risk assessment")
    
    # Input method selection
    input_method = st.radio("Select Input Method:", 
                            ["Manual Entry", "Upload CSV File", "Use Sample Data"])
    
    if input_method == "Manual Entry":
        st.subheader("Enter Sensor Readings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            seismic = st.slider("Seismic Activity (m/s²)", 0.0, 10.0, 2.5, 0.1,
                               help="Ground acceleration from seismic sensors")
            vibration = st.slider("Vibration Level (mm/s)", 0.0, 50.0, 10.0, 1.0,
                                 help="Structural vibration intensity")
            water_pressure = st.slider("Water Pressure (kPa)", 0.0, 500.0, 150.0, 10.0,
                                      help="Pore water pressure in rock mass")
        
        with col2:
            displacement = st.slider("Ground Displacement (mm)", 0.0, 100.0, 20.0, 1.0,
                                    help="Cumulative slope movement")
            rainfall = st.slider("Rainfall (mm/hr)", 0.0, 50.0, 5.0, 1.0,
                                help="Current precipitation rate")
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'seismic_activity': [seismic],
            'vibration_sensor': [vibration],
            'water_pressure': [water_pressure],
            'ground_displacement': [displacement],
            'rainfall': [rainfall]
        })
        
        if st.button("Predict Risk Level", type="primary"):
            make_prediction(best_model, input_data, metadata, label_encoder)
    
    elif input_method == "Upload CSV File":
        st.subheader("Upload CSV File")
        st.info("CSV should contain columns: seismic_activity, vibration_sensor, water_pressure, ground_displacement, rainfall")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(input_data.head())
                
                if st.button("Predict Risk Levels", type="primary"):
                    predictions = best_model.predict(input_data)
                    
                    if label_encoder:
                        predictions_display = label_encoder.inverse_transform(predictions)
                    else:
                        predictions_display = predictions
                    
                    results = input_data.copy()
                    results['Predicted_Risk'] = predictions_display
                    
                    st.success("Predictions completed!")
                    st.dataframe(results)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="rockfall_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Sample Data
        st.subheader("Sample Scenarios")
        
        sample_scenarios = {
            "Safe Conditions (Low Risk)": {
                'seismic_activity': 1.0,
                'vibration_sensor': 5.0,
                'water_pressure': 100.0,
                'ground_displacement': 5.0,
                'rainfall': 2.0
            },
            "Moderate Warning (Medium Risk)": {
                'seismic_activity': 3.5,
                'vibration_sensor': 15.0,
                'water_pressure': 200.0,
                'ground_displacement': 25.0,
                'rainfall': 10.0
            },
            "High Alert (High Risk)": {
                'seismic_activity': 6.0,
                'vibration_sensor': 30.0,
                'water_pressure': 350.0,
                'ground_displacement': 50.0,
                'rainfall': 25.0
            },
            "Emergency (Critical Risk)": {
                'seismic_activity': 8.5,
                'vibration_sensor': 45.0,
                'water_pressure': 450.0,
                'ground_displacement': 80.0,
                'rainfall': 40.0
            }
        }
        
        scenario = st.selectbox("Select a scenario:", list(sample_scenarios.keys()))
        
        input_data = pd.DataFrame([sample_scenarios[scenario]])
        
        st.write("**Scenario Parameters:**")
        st.dataframe(input_data)
        
        if st.button("Predict Risk Level", type="primary"):
            make_prediction(best_model, input_data, metadata, label_encoder)

def make_prediction(model, input_data, metadata, label_encoder):
    """Make prediction and display results"""
    
    try:
        prediction = model.predict(input_data)[0]
        
        # Get display label
        if label_encoder:
            prediction_label = label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = prediction
        
        # Get probability if available
        try:
            probabilities = model.predict_proba(input_data)[0]
            has_proba = True
        except:
            has_proba = False
        
        # Display prediction
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Risk level display
        risk_classes = {
            'Low': 'risk-low',
            'Medium': 'risk-medium',
            'High': 'risk-high',
            'Critical': 'risk-critical'
        }
        
        risk_class = risk_classes.get(prediction_label, 'risk-low')
        st.markdown(f'<div class="{risk_class}">Risk Level: {prediction_label.upper()}</div>', 
                   unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("Recommended Actions")
        
        recommendations = {
            'Low': [
                "Continue normal mining operations",
                "Maintain regular monitoring schedule",
                "Document current sensor readings"
            ],
            'Medium': [
                "Increase monitoring frequency",
                "Visual inspection of slope areas",
                "Alert geological team",
                "Consider restricting non-essential personnel"
            ],
            'High': [
                "Restrict access to high-risk zones",
                "Evacuate non-essential personnel",
                "Emergency team on standby",
                "Continuous real-time monitoring",
                "Halt operations in affected areas"
            ],
            'Critical': [
                "IMMEDIATE EVACUATION REQUIRED",
                "Activate emergency protocols",
                "Alert all personnel immediately",
                "Complete shutdown of operations",
                "Emergency response team deployment",
                "Incident command center activation"
            ]
        }
        
        for rec in recommendations.get(prediction_label, []):
            st.write(rec)
        
        # Probability distribution
        if has_proba:
            st.markdown("---")
            st.subheader("Confidence Distribution")
            
            class_names = metadata.get('risk_categories', ['Low', 'Medium', 'High', 'Critical'])
            class_names_sorted = sorted(class_names)
            
            prob_df = pd.DataFrame({
                'Risk Level': class_names_sorted,
                'Probability': probabilities * 100
            })
            
            fig = px.bar(prob_df, x='Risk Level', y='Probability',
                        color='Probability',
                        color_continuous_scale=['white', 'gray', 'black'],
                        labels={'Probability': 'Probability (%)'},
                        title='Risk Level Probability Distribution')
            
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Input summary
        st.markdown("---")
        st.subheader("Input Data Summary")
        st.dataframe(input_data)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

def show_performance_page(best_model, metadata, label_encoder):
    """Model performance analysis page"""
    
    st.header("Model Performance Analysis")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    if X_test is None or y_test is None:
        st.warning("Test data not available. Please run notebooks 01-03 first.")
        return
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", metadata.get('model_name', 'Unknown'))
    with col2:
        st.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0)*100:.2f}%")
    with col3:
        st.metric("F1-Score", f"{metadata.get('test_f1_score', 0)*100:.2f}%")
    with col4:
        st.metric("Test Samples", len(y_test))
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    class_names = metadata.get('risk_categories', ['Low', 'Medium', 'High', 'Critical'])
    class_names_sorted = sorted(class_names)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use appropriate colormap based on theme
    cmap = 'Greys' if st.session_state.theme == 'light' else 'gray'
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names_sorted,
                yticklabels=class_names_sorted,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Risk Level', fontweight='bold')
    ax.set_ylabel('True Risk Level', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    
    # Set face color based on theme
    if st.session_state.theme == 'dark':
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#2D2D2D')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("Classification Report")
    
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, 
                                   target_names=class_names_sorted,
                                   output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    metrics_df = report_df.loc[class_names_sorted, ['precision', 'recall', 'f1-score', 'support']]
    
    st.dataframe(metrics_df.style.background_gradient(cmap='binary', subset=['precision', 'recall', 'f1-score']))
    
    # Metrics explanation
    with st.expander("Understanding the Metrics"):
        st.write("""
        **Precision**: Of all predictions for a risk level, how many were correct?
        - High precision = fewer false alarms
        
        **Recall**: Of all actual instances of a risk level, how many did we catch?
        - High recall = fewer missed dangerous situations (most important for safety!)
        
        **F1-Score**: Harmonic mean of precision and recall
        - Balanced metric for overall performance
        
        **Support**: Number of actual occurrences of each class in the test data
        """)

def show_about_page():
    """About page with project information"""
    
    st.header("About This Project")
    
    st.markdown("""
    ## Academic Project
    
    **Course**: Data Analytics & Visualization (G5AD21DAV)  
    **Institution**: Rashtriya Raksha University  
    **Author**: JAMPANIKOMAL  
    **Problem Statement**: SIH25071 - Rockfall Risk Assessment in Open-Pit Mines
    
    ---
    
    ## Project Objectives
    
    This project develops an AI-powered system for real-time rockfall risk assessment in 
    open-pit mining operations. The system uses machine learning to analyze sensor data 
    and predict risk levels, enabling proactive safety measures.
    
    ---
    
    ## Dataset
    
    **Hybrid Approach (20,000+ samples)**:
    - Synthetic mine slope monitoring data (10,000 samples)
    - Real industrial mining process data from Kaggle (10,000+ samples)
    - Features: Seismic activity, vibration, water pressure, displacement, rainfall
    - 4-level classification: Low, Medium, High, Critical
    
    ---
    
    ## Machine Learning Models
    
    **Models Evaluated**:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Decision Tree
    - Random Forest
    - K-Nearest Neighbors
    - Naive Bayes
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Voting Classifier (Ensemble)
    - Stacking Classifier (Ensemble)
    
    **Best Model Selection**: Based on accuracy, F1-score, and cross-validation
    
    ---
    
    ## Technology Stack
    
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn, XGBoost, LightGBM
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Web Framework**: Streamlit
    - **Explainable AI**: SHAP
    
    ---
    
    ## Project Structure
    
    ```
    open-pit-mine-rockfall-prediction/
    ├── notebooks/
    │   ├── 01_data_generation_and_exploration.ipynb
    │   ├── 02_data_preprocessing.ipynb
    │   ├── 03_model_development.ipynb
    │   └── 04_results_visualization.ipynb
    ├── data/
    │   ├── rockfall_data.csv
    │   └── processed/
    ├── models/
    │   ├── best_model.pkl
    │   ├── model_metadata.pkl
    │   └── all_models.pkl
    ├── app.py (this Streamlit app)
    ├── requirements.txt
    └── README.md
    ```
    
    ---
    
    ## GitHub Repository
    
    **Repository**: [github.com/JAMPANIKOMAL/open-pit-mine-rockfall-prediction](https://github.com/JAMPANIKOMAL/open-pit-mine-rockfall-prediction)
    
    ---
    
    ## Contact
    
    For questions or collaboration opportunities:
    - **GitHub**: JAMPANIKOMAL
    - **Project**: Open-Pit Mine Rockfall Prediction System
    
    ---
    
    ## Acknowledgments
    
    - Rashtriya Raksha University for academic support
    - Kaggle community for mining dataset (edumagalhaes)
    - Open-source ML/AI community
    - Smart India Hackathon (SIH25071 Problem Statement)
    """)

# Run the app
if __name__ == "__main__":
    main()
