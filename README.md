🏥 Hospital Readmission Risk PredictionA machine learning model that predicts 30-day hospital readmission risk using patient discharge data from 130 US hospitals. This project combines Random Forest classification with SMOTE balancing to handle class imbalance in healthcare data.📊 Project OverviewProblem: Hospital readmissions increase costs and indicate gaps in discharge planning. Early identification of high-risk patients enables targeted interventions.Solution: A predictive model that estimates 30-day readmission probability using discharge data. The system features clinically calibrated risk thresholds to ensure high sensitivity for at-risk patients, addressing the class imbalance inherent in medical datasets.Results:Balanced Accuracy: 72% (after SMOTE)Precision (High-Risk): 89%Recall: Optimized to avoid missing critical cases (false negatives)📸 Interface📁 Repository Structurehospital-readmission/
├── notebooks/
│   └── Readmission_model.ipynb          # Full ML pipeline & analysis
├── models/
│   └── README.md                        # Model documentation (artifact hosted on Hugging Face)
├── app.py                               # Streamlit web interface
├── requirements.txt                     # Python dependencies
├── MODEL_CARD.md                        # Detailed model documentation & ethics
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
🚀 Quick StartLocal Setup (Development)# Clone repository
git clone [https://github.com/uthy4r/Readmission-model.git](https://github.com/uthy4r/Readmission-model.git)
cd Readmission-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
The app opens at http://localhost:8501.Note: The app will automatically download the model from Hugging Face on the first run.Run Notebook (Training)jupyter notebook notebooks/Readmission_model.ipynb
This executes the full pipeline:Data loading from UCI ML RepositoryExploratory data analysis (EDA)Feature engineering & preprocessingModel training (Logistic Regression baseline + Random Forest)SMOTE oversampling for class balanceModel evaluation & feature importance📦 Model File (Important)This repository uses a Hybrid Architecture to handle the large model file (~240MB) without bloating the Git repository.How it works:Frontend: Streamlit Cloud handles the UI.Model Registry: Hugging Face Hub hosts the trained .pkl artifact.Dynamic Loading: When the app starts, it fetches the model securely from:https://huggingface.co/Uthy4r/hospital_readmission_model/resolve/main/rf_readmission_smote.pklTo generate the model yourself:# Navigate to project folder
cd hospital-readmission

# Open and run the notebook
jupyter notebook notebooks/Readmission_model.ipynb

# In the notebook:
# - Click Kernel → Restart & Run All
# - Wait for completion (15-20 minutes)
# - The file rf_readmission_smote.pkl will be created
📝 Model DetailsAlgorithms UsedModelPurposeResultLogistic RegressionBaseline72% accuracyRandom Forest (200 trees)Primary72% accuracy, better feature insightsSMOTEClass balancingImproved recall for minority classKey Featuresage: Patient age at dischargetime_in_hospital: Number of days hospitalizednum_lab_procedures: Laboratory tests performednum_medications: Count of discharge medicationsnumber_outpatient: Visits in past yearnumber_emergency: Emergency visits in past yearnumber_inpatient: Inpatient visits in past yearClinical Logic & ThresholdsUnlike standard ML models that use a 50% cutoff, this system uses medically grounded thresholds:High Risk (> 30%): Patient has >3x the baseline risk. Action: Intensive intervention.Moderate Risk (15-30%): Patient has elevated risk. Action: Standard follow-up.Low Risk (< 15%): Risk is at or below baseline.🎯 Model EvaluationView detailed metrics in the notebook:Confusion matrices (Logistic Regression vs Random Forest)Classification reports (precision, recall, F1-score)Feature importance rankingsProbability distributions🌐 DeploymentLive DemoStreamlit Community Cloud: Click to Launch AppDeployment ArchitectureThis app is deployed using:Streamlit Cloud: For the Python web interface.Hugging Face: For large model object storage.📚 Data SourceUCI ML Repository: 130 US Hospitals for DiabetesRecords: ~101k hospital staysFeatures: 55 clinical variablesTarget: Readmitted within 30 days (binary)🔐 Security Notes✅ No hardcoded secrets (all auth tokens, API keys removed)✅ Large files ignored (models via .gitignore)✅ Data privacy: No patient identifiers in repo⚠️ Model for research only: Not validated for clinical use📦 RequirementsSee requirements.txt:streamlit — Web interfacescikit-learn — ML models & evaluationpandas — Data manipulationimbalanced-learn — SMOTE balancingucimlrepo — Dataset fetchingrequests — Model downloading🛠️ Troubleshooting"Model not found" errorThe app attempts to auto-download the model. If this fails, check your internet connection or manually download the .pkl from the Hugging Face link above and place it in the models/ directory.Port 8501 already in usestreamlit run app.py --server.port 8502
📊 Next Steps & Improvements[ ] Cross-validation with k-folds[ ] Hyperparameter tuning (GridSearchCV)[ ] Feature selection (RFE, SelectKBest)[ ] Production monitoring & model drift detection[ ] User authentication for Streamlit app[ ] Integration with hospital EHR systems[ ] Calibration for clinical decision thresholds📄 LicenseMIT License — Feel free to use for research/education👤 AuthorDr. Uthman Babatunde | Medical Doctor & Applied AI Engineer📧 Email🔗 LinkedIn | GitHub📖 ReferencesStrack, B., et al. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates" BioMed Research InternationalScikit-learn Documentation: SMOTEUCI ML Repository Diabetes DatasetLast Updated: January 2026Model Version: 1.0 (Random Forest + SMOTE)
