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

# Base directory for resolving data and model paths reliably when Streamlit
# changes the current working directory. Use the app file location as root.
BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="Rockfall Risk Assessment",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS styling
st.markdown("""
    <style>
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid #4A90E2;
        padding-bottom: 10px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #B0B0B0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Risk level boxes - subtle professional colors */
    .risk-critical {
        background-color: #8B0000;
        color: white;
        padding: 20px;
        border: 2px solid #A52A2A;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .risk-high {
        background-color: #D35400;
        color: white;
        padding: 20px;
        border: 2px solid #E67E22;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .risk-medium {
        background-color: #7D6608;
        color: white;
        padding: 20px;
        border: 2px solid #9A7D0A;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .risk-low {
        background-color: #1E8449;
        color: white;
        padding: 20px;
        border: 2px solid #27AE60;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border: 1px solid #333333;
        border-left: 4px solid #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and metadata
@st.cache_resource
def load_models():
    """Load trained models and metadata"""
    try:
        model_path = BASE_DIR / 'models' / 'best_model.pkl'
        metadata_path = BASE_DIR / 'models' / 'model_metadata.pkl'
        
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Try loading label encoder if it exists
        label_encoder = None
        if metadata.get('uses_encoded_labels', False):
            le_path = BASE_DIR / 'models' / 'label_encoder.pkl'
            if le_path.exists():
                with open(le_path, 'rb') as f:
                    label_encoder = pickle.load(f)
        
        return best_model, metadata, label_encoder
    except Exception as e:
        # Provide the fully resolved paths in the message to make debugging easier
        st.error(f"Error loading models: {str(e)}")
        st.info(f"Checked paths: {model_path} and {metadata_path}.\nPlease ensure you have run notebooks 01-03 to generate the models and that the files exist.")
        return None, None, None

# Load test data for analysis
@st.cache_data
def load_test_data():
    """Load test data for model evaluation"""
    try:
        x_path = BASE_DIR / 'data' / 'processed' / 'X_test.csv'
        y_path = BASE_DIR / 'data' / 'processed' / 'y_test.csv'

        X_test = pd.read_csv(str(x_path))
        y_test = pd.read_csv(str(y_path)).values.ravel()
        return X_test, y_test
    except Exception as e:
        # Show resolved path to help the user fix missing file issues
        st.warning(f"Test data not available: {str(e)}")
        st.info(f"Tried loading: {x_path} and {y_path}")
        return None, None


def prepare_input_for_model(df: pd.DataFrame, metadata: dict, model=None) -> pd.DataFrame:
    """Normalize and align input dataframe columns to the feature names the model expects.

    This function will:
    - lower-case and strip incoming column names
    - attempt to use metadata feature list if available
    - fall back to model.feature_names_in_ when present
    - apply a small synonym map for common column name variants
    - return a dataframe with columns ordered to match the model
    """
    # Defensive copy
    df = df.copy()

    # normalize incoming column names
    original_cols = list(df.columns)
    norm_cols = {c: c.strip().lower() for c in original_cols}
    df.columns = [c.strip().lower() for c in original_cols]

    # Determine target feature names from metadata or model
    target_features = None
    if metadata:
        for key in ("feature_names", "feature_columns", "model_features", "features"):
            if key in metadata and metadata[key]:
                target_features = list(metadata[key])
                break

    if target_features is None and model is not None:
        feat_attr = getattr(model, "feature_names_in_", None)
        if feat_attr is not None:
            target_features = list(feat_attr)

    # Normalize target features if present
    if target_features is not None:
        target_norm = [str(c).strip().lower() for c in target_features]
    else:
        target_norm = None

    # small synonym map for common naming differences
    synonyms = {
        'ground_displacement': 'displacement_mm',
        'displacement': 'displacement_mm',
        'displacement_mm': 'displacement_mm',
        'vibration_sensor': 'vibration_level',
        'vibration_level': 'vibration_level',
        'vibration': 'vibration_level',
        'rainfall': 'rainfall_mm',
        'rainfall_mm': 'rainfall_mm',
        'water_pressure': 'joint_water_pressure',
        'joint_water_pressure': 'joint_water_pressure',
        'seismic_activity': 'seismic_activity'
    }

    # If we know the desired features, build mapping to them
    if target_norm is not None:
        col_map = {}
        for src in df.columns:
            # direct match
            if src in target_norm:
                # map to the original target name (preserve original casing from metadata)
                mapped = target_features[target_norm.index(src)]
                col_map[src] = mapped
                continue

            # synonym match: find a target that equals the synonym
            syn = synonyms.get(src)
            if syn and syn in target_norm:
                mapped = target_features[target_norm.index(syn)]
                col_map[src] = mapped
                continue

            # try reverse: if any target_norm equals a synonym of src
            for t_norm, t_orig in zip(target_norm, target_features):
                if t_norm in synonyms.values() and t_norm == synonyms.get(src, ''):
                    col_map[src] = t_orig
                    break

        # Rename incoming columns to match model expected names
        if col_map:
            df = df.rename(columns=col_map)

        # Check for missing features
        missing = [f for f in target_features if f not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing}. Available: {list(df.columns)}")

        # Reorder to target feature order
        df = df[target_features]
        return df

    # If we don't know expected features, return df as-is but normalized
    return df


def _map_prediction_to_label(pred, metadata, label_encoder):
    """Robust mapping from model prediction to human-friendly label.

    Based on analysis, the model training used alphabetical encoding, but we want
    to show the logical relationship: higher sensor values = higher risk
    """
    try:
        if label_encoder is not None:
            try:
                return label_encoder.inverse_transform([pred])[0]
            except Exception:
                pass

        # Based on training data analysis:
        # Label 0: highest sensor values (originally Critical, but was encoded as 0)
        # Label 1: medium-high values (originally High, but was encoded as 1) 
        # Label 2: lowest sensor values (originally Low, but was encoded as 2)
        # Label 3: low-medium values (originally Medium, but was encoded as 3)
        #
        # The model learned: high sensor values → predict 0, low sensor values → predict 2
        # We want to show: high sensor values → "Critical", low sensor values → "Low"
        
        actual_mapping = {
            0: 'Critical',  # Model predicts 0 for highest sensor values
            1: 'High',      # Model predicts 1 for medium-high sensor values  
            2: 'Low',       # Model predicts 2 for lowest sensor values
            3: 'Medium'     # Model predicts 3 for low-medium sensor values
        }
        
        if isinstance(pred, (int, np.integer)):
            return actual_mapping.get(int(pred), str(pred))
            
        if isinstance(pred, str) and pred.isdigit():
            return actual_mapping.get(int(pred), pred)
            
        # Try case-insensitive matching 
        for name in actual_mapping.values():
            if str(pred).strip().lower() == str(name).strip().lower():
                return name

    except Exception:
        pass

    return str(pred)

# Main app
def main():
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

    # Build HTML cards to avoid stray empty boxes and provide precise layout
    acc_percent = int(round(metadata.get('test_accuracy', 0) * 100))
    model_name = metadata.get('model_name', 'Unknown')

    card_template = """
    <div style='background:#1E1E1E;padding:18px;border:1px solid #333333;border-left:4px solid #333333;border-radius:8px;min-height:110px;'>
        <div style='color:#B0B0B0;font-size:13px;margin-bottom:6px;'>{label}</div>
        <div style='font-size:34px;color:white;margin-top:6px;'>{value}</div>
        <div style='height:8px'></div>
    </div>
    """

    # Column 1: Model Accuracy (static display)
    with col1:
        html = card_template.format(label='Model Accuracy', value=f'{acc_percent}%')
        st.markdown(html, unsafe_allow_html=True)

    # Column 2: Risk Categories
    with col2:
        html = card_template.format(label='Risk Categories', value='4 Levels')
        st.markdown(html, unsafe_allow_html=True)

    # Column 3: Model Type
    with col3:
        html = card_template.format(label='Model Type', value=model_name)
        st.markdown(html, unsafe_allow_html=True)

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
    
    st.info("ℹ️ **Note**: Higher sensor readings indicate higher risk levels - more seismic activity, vibration, water pressure, ground displacement, and rainfall suggest increased rockfall danger.")
    
    # Input method selection
    input_method = st.radio("Select Input Method:", 
                            ["Manual Entry", "Upload CSV File", "Use Sample Data"])
    
    if input_method == "Manual Entry":
        st.subheader("Enter Sensor Readings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            seismic = st.slider("Seismic Activity (m/s²)", 0.0, 2.0, 0.5, 0.05,
                               help="Ground acceleration from seismic sensors")
            vibration = st.slider("Vibration Level (mm/s)", 0.0, 8.0, 2.0, 0.1,
                                 help="Structural vibration intensity")
            water_pressure = st.slider("Water Pressure (kPa)", 0.0, 600.0, 150.0, 10.0,
                                      help="Pore water pressure in rock mass")
        
        with col2:
            displacement = st.slider("Ground Displacement (mm)", 0.0, 12.0, 6.0, 0.5,
                                    help="Cumulative slope movement")
            rainfall = st.slider("Rainfall (mm/hr)", 0.0, 70.0, 30.0, 1.0,
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
                    try:
                        input_prepared = prepare_input_for_model(input_data, metadata, best_model)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.info("Make sure your CSV columns match the model features. Example: displacement_mm, vibration_level, rainfall_mm, joint_water_pressure, seismic_activity")
                        input_prepared = None

                    if input_prepared is not None:
                        predictions = best_model.predict(input_prepared)

                        # Robust mapping of predictions to display labels
                        try:
                            predictions_display = [_map_prediction_to_label(p, metadata, label_encoder) for p in predictions]
                        except Exception:
                            # fallback to raw predictions
                            predictions_display = [str(p) for p in predictions]

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
                'seismic_activity': 0.15,
                'vibration_sensor': 1.0,
                'water_pressure': 150.0,
                'ground_displacement': 3.0,
                'rainfall': 15.0
            },
            "Moderate Warning (Medium Risk)": {
                'seismic_activity': 0.35,
                'vibration_sensor': 2.3,
                'water_pressure': 220.0,
                'ground_displacement': 7.0,
                'rainfall': 30.0
            },
            "High Alert (High Risk)": {
                'seismic_activity': 0.65,
                'vibration_sensor': 3.0,
                'water_pressure': 350.0,
                'ground_displacement': 8.0,
                'rainfall': 45.0
            },
            "Emergency (Critical Risk)": {
                'seismic_activity': 1.2,
                'vibration_sensor': 5.5,
                'water_pressure': 500.0,
                'ground_displacement': 10.5,
                'rainfall': 60.0
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
        # Align input columns to model's expected features
        try:
            input_prepared = prepare_input_for_model(input_data, metadata, model)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Check that input feature names match those used during training. Example features: displacement_mm, vibration_level, rainfall_mm, joint_water_pressure, seismic_activity")
            return

        prediction = model.predict(input_prepared)[0]
        
        # Map prediction to human-friendly label
        prediction_label = _map_prediction_to_label(prediction, metadata, label_encoder)
        
        # Get probability if available
        try:
            probabilities = model.predict_proba(input_prepared)[0]
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
        
        # Case-insensitive lookup for CSS class
        risk_classes_lower = {k.lower(): v for k, v in risk_classes.items()}
        risk_class = risk_classes_lower.get(str(prediction_label).strip().lower(), 'risk-low')
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
        
        # Use case-insensitive match to find canonical recommendation key
        canonical_key = None
        for k in recommendations.keys():
            if k.lower() == str(prediction_label).strip().lower():
                canonical_key = k
                break
        if canonical_key is None:
            canonical_key = prediction_label

        for rec in recommendations.get(canonical_key, []):
            st.write(rec)
        
        # Probability distribution
        if has_proba:
            st.markdown("---")
            st.subheader("Confidence Distribution")

            # Determine display names for classes in the same order as model.classes_
            model_classes = getattr(model, 'classes_', None)

            if model_classes is not None:
                # If we have a label encoder, use it to map model.classes_
                if label_encoder is not None:
                    try:
                        class_names_display = list(label_encoder.inverse_transform(model_classes))
                    except Exception:
                        class_names_display = [str(c) for c in model_classes]
                else:
                    # If metadata provides an ordered list, map integer class labels to that list
                    if metadata and metadata.get('risk_categories'):
                        mc = []
                        for c in model_classes:
                            if isinstance(c, (int, np.integer)):
                                idx = int(c)
                                try:
                                    mc.append(metadata['risk_categories'][idx])
                                except Exception:
                                    mc.append(str(c))
                            else:
                                mc.append(str(c))
                        class_names_display = mc
                    else:
                        class_names_display = [str(c) for c in model_classes]
            else:
                # Fallback to metadata order or default order
                class_names_display = metadata.get('risk_categories', ['Low', 'Medium', 'High', 'Critical'])

            # Build dataframe aligning probabilities with the display names
            prob_df = pd.DataFrame({
                'Risk Level': class_names_display,
                'Probability': list(probabilities * 100)
            })

            fig = px.bar(prob_df, x='Risk Level', y='Probability',
                        color='Probability',
                        color_continuous_scale='Blues',
                        labels={'Probability': 'Probability (%)'},
                        title='Risk Level Probability Distribution')

            fig.update_layout(
                showlegend=False,
                height=400,
                plot_bgcolor='#262730',
                paper_bgcolor='#0E1117',
                font=dict(color='white')
            )
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
        st.markdown("### Model")
        st.write(metadata.get('model_name', 'Unknown'))

    # Test Accuracy as percent + progress bar
    with col2:
        acc_percent = int(round(metadata.get('test_accuracy', 0) * 100))
        st.markdown("### Test Accuracy")
        st.markdown(f"<h2 style='color: white; margin: 0;'>{acc_percent:.0f}%</h2>", unsafe_allow_html=True)
        st.progress(acc_percent)

    # F1-Score as percent + progress bar
    with col3:
        f1_percent = int(round(metadata.get('test_f1_score', 0) * 100))
        st.markdown("### F1-Score")
        st.markdown(f"<h2 style='color: white; margin: 0;'>{f1_percent:.0f}%</h2>", unsafe_allow_html=True)
        st.progress(f1_percent)

    with col4:
        st.markdown("### Test Samples")
        st.write(len(y_test))
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    class_names = metadata.get('risk_categories', ['Low', 'Medium', 'High', 'Critical'])
    class_names_sorted = sorted(class_names)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dark theme for matplotlib
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#262730')
    
    # Use professional color scheme for heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_sorted,
                yticklabels=class_names_sorted,
                cbar_kws={'label': 'Count'},
                annot_kws={'color': 'white'})
    
    ax.set_xlabel('Predicted Risk Level', fontweight='bold', color='white')
    ax.set_ylabel('True Risk Level', fontweight='bold', color='white')
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14, color='white')
    ax.tick_params(colors='white')
    
    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
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
    
    st.dataframe(metrics_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))
    
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
