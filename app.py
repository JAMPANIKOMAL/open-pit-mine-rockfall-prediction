import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- 1. CONFIGURATION AND ASSET LOADING ---

# Define paths relative to the app.py file
# This assumes the 'models' folder is next to app.py
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
LE_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.pkl') # Our XGBoost Champion

# List of features used for model training (MUST match Notebook 3)
FEATURE_COLS = [
    'rainfall_mm_past_24h', 
    'seismic_activity', 
    'joint_water_pressure_kPa', 
    'vibration_level', 
    'displacement_mm'
]

@st.cache_resource
def load_assets():
    """Loads the necessary preprocessors and the trained model."""
    try:
        # Load Preprocessors
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(LE_PATH, 'rb') as f:
            le = pickle.load(f)
        # Load Model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return scaler, le, model
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Ensure all files are in the 'models' folder.")
        st.error(f"Missing file: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading model assets: {e}")
        st.stop()

# Load the assets once
scaler, le, model = load_assets()

# --- 2. PREDICTION FUNCTION ---

def make_prediction(input_data, scaler, model, le):
    """Scales the input, predicts the class, and returns the original label."""
    # 1. Convert input data to a DataFrame (MUST use the correct column order/names)
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLS)
    
    # 2. Scale the input data (CRITICAL step)
    input_scaled = scaler.transform(input_df)
    
    # 3. Predict the encoded class (0, 1, 2, 3)
    prediction_encoded = model.predict(input_scaled)
    
    # 4. Inverse transform to get the text label (XGBoost returns float, so we cast to int)
    predicted_label = le.inverse_transform(prediction_encoded.astype(int))[0]
    
    return predicted_label

# --- 3. STREAMLIT UI LAYOUT ---

st.set_page_config(page_title="Mine Rockfall Risk Predictor", layout="wide")

st.title("Open-Pit Mine Rockfall Risk Predictor")
st.markdown("---")
st.subheader("Sensor Input Parameters")
st.markdown("Adjust the sliders below to simulate geotechnical sensor readings.")


# Define risk levels for styling
RISK_LEVEL_MAP = {
    'Critical': ('#8B0000', 'Immediate Evacuation and Stabilization Required.'),
    'High': ('#D35400', 'Close Monitoring and Restricted Operations.'),
    'Medium': ('#7D6608', 'Elevated Risk. Monitor for changes every 2 hours.'),
    'Low': ('#1E8449', 'Minimal Risk. Routine Monitoring.')
}

# --- Input Sliders (Using Correct Feature Names) ---
col1, col2, col3 = st.columns(3)

# The ranges are based on our EDA findings for a realistic interface.
with col1:
    # 1. Displacement (Strongest Predictor)
    displacement_mm = st.slider("Displacement (mm)", 
                                min_value=0.0, max_value=45.0, value=5.0, step=0.1)
    
    # 2. Water Pressure
    joint_water_pressure_kPa = st.slider("Water Pressure (kPa)", 
                                         min_value=10.0, max_value=80.0, value=35.0, step=1.0)
with col2:
    # 3. Seismic Activity
    seismic_activity = st.slider("Seismic Activity (Magnitude)", 
                                 min_value=0.0, max_value=7.0, value=0.5, step=0.1)
    
    # 4. Vibration Level
    vibration_level = st.slider("Vibration Level (Score)", 
                                min_value=0.0, max_value=1.5, value=0.3, step=0.01)

with col3:
    # 5. Rainfall (Weakest Predictor)
    rainfall_mm_past_24h = st.slider("Rainfall Past 24h (mm)", 
                                     min_value=0.0, max_value=25.0, value=2.0, step=0.1)

# --- Organize Input Data ---
input_data = {
    'rainfall_mm_past_24h': rainfall_mm_past_24h,
    'seismic_activity': seismic_activity,
    'joint_water_pressure_kPa': joint_water_pressure_kPa,
    'vibration_level': vibration_level,
    'displacement_mm': displacement_mm
}

# --- 4. PREDICTION AND OUTPUT ---

st.markdown("---")
if st.button("Assess Rockfall Risk", type="primary"):
    
    # Perform prediction
    predicted_risk = make_prediction(input_data, scaler, model, le)
    
    # Get color and description
    color, description = RISK_LEVEL_MAP.get(predicted_risk, ('#808080', 'Unknown Risk'))
    
    # Display the result using markdown and HTML styling
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.4);">
            <h2 style="color: white; margin-top: 0px;">Predicted Risk Level: {predicted_risk.upper()}</h2>
            <p style="font-size: 18px; color: white;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Input Data Summary")
    st.dataframe(pd.DataFrame([input_data]))

# --- End of Streamlit Code ---