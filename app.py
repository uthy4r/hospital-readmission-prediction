"""
Streamlit app for Hospital Readmission Risk Prediction
Load trained Random Forest model and provide interactive predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Readmission Risk Prediction")
st.markdown("""
Predict the likelihood of 30-day hospital readmission using a Random Forest model trained on 
discharge data from 130 US hospitals. 

**Model Note:** This system uses clinically calibrated thresholds. Because hospital readmission is a minority event (~11% baseline), 
risk scores >30% are flagged as **High Risk**.
""")

# Download model from Hugging Face
@st.cache_resource
def load_model():
    model_path = "models/hospital_readmission_model.pkl"
    # Use the 'resolve' URL to ensure binary download
    url = "https://huggingface.co/Uthy4r/hospital_readmission_model/resolve/main/rf_readmission_smote.pkl"
    
    # Check if file exists and delete if corrupted (too small)
    if os.path.exists(model_path):
        if os.path.getsize(model_path) < 1000:
            print("Detected corrupted model file. Deleting and re-downloading...")
            os.remove(model_path)
    
    # Download if missing
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        print(f"Downloading model from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Download failed! Check URL. Error: {e}")
            return None

    # Load the model
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model file. It might be corrupted. Error: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Sidebar for input
st.sidebar.header("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 0, 100, 65)
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 150, 45)
    num_medications = st.slider("Number of Medications", 1, 81, 15)

with col2:
    # Exact feature names required by model
    number_outpatient = st.number_input("Outpatient Visits (past year)", 0, 50, 0)
    number_emergency = st.number_input("Emergency Visits (past year)", 0, 50, 0)
    number_inpatient = st.number_input("Inpatient Visits (past year)", 0, 50, 0)

st.sidebar.markdown("---")

# Create feature vector with EXACT names required by the model
input_data = pd.DataFrame({
    'age': [age],
    'time_in_hospital': [time_in_hospital],
    'num_lab_procedures': [num_lab_procedures],
    'num_medications': [num_medications],
    'number_outpatient': [number_outpatient],
    'number_emergency': [number_emergency],
    'number_inpatient': [number_inpatient]
})

# Make prediction
if st.sidebar.button("🔮 Predict Readmission Risk", use_container_width=True):
    try:
        # Get probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            prediction = model.predict(input_data)[0]
            probability = 1.0 if prediction == 1 else 0.0
        
        # Display results with CLINICALLY REALISTIC thresholds
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Calibrated logic: 
            # Baseline risk is ~11%. 
            # >30% is approx 3x baseline risk -> High.
            # >15% is elevated -> Moderate.
            if probability >= 0.30:
                risk_label = "High Risk 🚨"
                delta_color = "inverse" # Red
                interpretation = "**High Risk**: Patient risk is >3x the average. Enhanced discharge planning strongly recommended."
            elif probability >= 0.15:
                risk_label = "Moderate Risk ⚠️"
                delta_color = "off"     # Gray/Yellow
                interpretation = "**Moderate Risk**: Risk is elevated above baseline. Specific follow-up protocols recommended."
            else:
                risk_label = "Low Risk ✅"
                delta_color = "normal"  # Green
                interpretation = "**Low Risk**: Risk is at or below population baseline. Standard discharge protocols."
            
            st.metric("Risk Level", risk_label, f"{probability:.1%}", delta_color=delta_color)
        
        with col2:
            st.metric("Readmission Probability", f"{probability:.1%}")
        
        with col3:
            st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
        
        # Risk gauge
        st.progress(probability)
        
        # Interpretation box
        st.markdown("---")
        st.subheader("💡 Interpretation")
        if probability >= 0.30:
            st.error(interpretation)
        elif probability >= 0.15:
            st.warning(interpretation)
        else:
            st.success(interpretation)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Model trained on publicly available hospital discharge data (UCI Diabetes Dataset).</p>
    <p><i>Note: Thresholds calibrated to dataset prevalence (~11%). Probabilities >30% indicate high relative risk.</i></p>
</div>
""", unsafe_allow_html=True)
