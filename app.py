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
discharge data from 130 US hospitals. This model was developed using SMOTE to handle class imbalance.
""")

# Download model from Hugging Face on first run
@st.cache_resource
def load_model():
    model_path = "models/hospital_readmission_model.pkl"
    
    # Only download if not already cached locally
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        print("Downloading model from Hugging Face...")
        
        url = "https://huggingface.co/Uthy4r/hospital_readmission_model/resolve/main/rf_readmission_smote.pkl"
        response = requests.get(url)
        
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    
    # Load the model
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Sidebar for input
st.sidebar.header("📋 Patient Information")

# Input fields (map to training features)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 0, 100, 50)
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 150, 40)

with col2:
    num_medications = st.slider("Number of Medications", 1, 81, 15)
    num_outpatient_visits = st.number_input("Outpatient Visits (past year)", 0, 100, 2)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 8)

st.sidebar.markdown("---")

# Create feature vector (must match training feature order)
input_data = pd.DataFrame({
    'age': [age],
    'time_in_hospital': [time_in_hospital],
    'num_lab_procedures': [num_lab_procedures],
    'num_medications': [num_medications],
    'num_outpatient_visits': [num_outpatient_visits],
    'number_diagnoses': [number_diagnoses]
})

# Make prediction
if st.sidebar.button("🔮 Predict Readmission Risk", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]
        # Check if the model supports predict_proba, otherwise mock it or handle gracefully
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            # Fallback if model doesn't support probability (though RF usually does)
            probability = 1.0 if prediction == 1 else 0.0
        
        # Display results
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", "High ⚠️" if probability > 0.5 else "Low ✅", 
                     f"{probability:.1%}")
        
        with col2:
            st.metric("Readmission Probability", f"{probability:.1%}")
        
        with col3:
            st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
        
        # Risk gauge
        st.progress(probability)
        
        # Interpretation
        st.markdown("---")
        st.subheader("💡 Interpretation")
        
        if probability > 0.7:
            st.warning("🚨 **Very High Risk**: This patient has a high likelihood of 30-day readmission. "
                      "Consider enhanced discharge planning and follow-up care.")
        elif probability > 0.5:
            st.info("⚠️ **Moderate-High Risk**: This patient shows increased readmission risk. "
                   "Standard discharge protocols with close monitoring recommended.")
        elif probability > 0.3:
            st.success("✅ **Low-Moderate Risk**: Patient has manageable readmission risk. "
                      "Standard care protocols appropriate.")
        else:
            st.success("✅ **Low Risk**: Patient shows minimal readmission risk. "
                      "Standard discharge procedures sufficient.")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Model info sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Model Information")
st.sidebar.markdown("""
- **Algorithm**: Random Forest (200 trees)
- **Balancing**: SMOTE (Synthetic Minority Oversampling)
- **Target**: 30-day hospital readmission
- **Data**: 130 US hospitals discharge records
- **Evaluation**: Confusion matrix, classification report available in notebook
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Model trained on publicly available hospital discharge data. Use for educational/research purposes only.</p>
    <p>For production deployment, validate against current clinical guidelines and institutional data.</p>
</div>
""", unsafe_allow_html=True)
