🏥 Hospital Readmission Risk Prediction
A machine learning model that predicts 30-day hospital readmission risk using patient discharge data from 130 US hospitals. This project combines Random Forest classification with SMOTE balancing to handle class imbalance in healthcare data.
📊 Project Overview
Problem: Hospital readmissions increase costs and indicate gaps in discharge planning. Early identification of high-risk patients enables targeted interventions.
Solution: A predictive model that estimates 30-day readmission probability using discharge data. The system features clinically calibrated risk thresholds (Low/Moderate/High) to ensure high sensitivity for at-risk patients, addressing the class imbalance inherent in medical datasets.
Results:
Balanced Accuracy: 72% (after SMOTE)
Precision (High-Risk): 89%
Recall: Optimized to avoid missing critical cases (false negatives)
📁 Repository Structure
hospital-readmission/
├── notebooks/
│   └── Readmission_model.ipynb          # Full ML pipeline & analysis
├── models/
│   └── README.md                        # Model documentation (artifact hosted on Hugging Face)
├── app.py                               # Streamlit web interface
├── requirements.txt                     # Python dependencies
├── MODEL_CARD.md                        # Detailed model documentation & ethics
├── .gitignore                           # Git ignore rules
└── README.md                            # This file



🚀 Quick Start
Local Setup (Development)
# Clone repository
git clone [https://github.com/uthy4r/Readmission-model.git](https://github.com/uthy4r/Readmission-model.git)
cd Readmission-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py



The app opens at http://localhost:8501.
Run Notebook (Training)
jupyter notebook notebooks/Readmission_model.ipynb



This executes the full pipeline:
Data loading from UCI ML Repository
Exploratory data analysis (EDA)
Feature engineering & preprocessing
Model training (Logistic Regression baseline + Random Forest)
SMOTE oversampling for class balance
Model evaluation & feature importance
📦 Model File (Important)
This repository uses a Hybrid Architecture to handle the large model file (~240MB) without bloating the Git repository. The app automatically downloads the model from Hugging Face on the first run.
To run the app locally:
Run streamlit run app.py.
The script will automatically fetch the model from https://huggingface.co/Uthy4r/hospital_readmission_model and cache it in the models/ directory.
To generate the model yourself:
# Navigate to project folder
cd hospital-readmission

# Open and run the notebook
jupyter notebook notebooks/Readmission_model.ipynb

# In the notebook:
# - Click Kernel → Restart & Run All
# - Wait for completion (15-20 minutes)
# - The file rf_readmission_smote.pkl will be created

# Move the model to the correct folder
move rf_readmission_smote.pkl models/



📝 Model Details
Algorithms Used
| Model | Purpose | Result |
| Logistic Regression | Baseline | 72% accuracy |
| Random Forest (200 trees) | Primary | 72% accuracy, better feature insights |
| SMOTE | Class balancing | Improved recall for minority class |
Key Features
age: Patient age at discharge
time_in_hospital: Number of days hospitalized
num_lab_procedures: Laboratory tests performed
num_medications: Count of discharge medications
number_outpatient: Visits in past year
number_emergency: Emergency visits in past year
number_inpatient: Inpatient visits in past year
Class Balance
Before SMOTE: 90% non-readmitted, 10% readmitted
After SMOTE: 50/50 balanced
🎯 Model Evaluation
View detailed metrics in the notebook:
Confusion matrices (Logistic Regression vs Random Forest)
Classification reports (precision, recall, F1-score)
Feature importance rankings
Probability distributions
🌐 Deployment
Option 1: Streamlit Community Cloud (Free, No Code)
Push code to GitHub
Go to share.streamlit.io
Connect your GitHub repo → Deploy
Share public URL: Click to Launch App
Option 2: Heroku / Railway / Render
# Add Procfile (Heroku)
echo "web: streamlit run app.py --server.port \$PORT" > Procfile

# Deploy
git push heroku main



Option 3: Docker (Production)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]



📚 Data Source
UCI ML Repository: 130 US Hospitals for Diabetes
Records: ~101k hospital stays
Features: 55 clinical variables
Target: Readmitted within 30 days (binary)
🔐 Security Notes
✅ No hardcoded secrets (all auth tokens, API keys removed)
✅ Large files ignored (models via .gitignore)
✅ Data privacy: No patient identifiers in repo
⚠️ Model for research only: Not validated for clinical use
📦 Requirements
See requirements.txt:
streamlit — Web interface
scikit-learn — ML models & evaluation
pandas — Data manipulation
imbalanced-learn — SMOTE balancing
ucimlrepo — Dataset fetching
requests — Model downloading
🛠️ Troubleshooting
"Model not found" error
Ensure internet connection (for auto-download)
Or manually download from Hugging Face and place in models/
Port 8501 already in use
streamlit run app.py --server.port 8502



Import errors
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall



📊 Next Steps & Improvements
$$ $$
Cross-validation with k-folds
$$ $$
Hyperparameter tuning (GridSearchCV)
$$ $$
Feature selection (RFE, SelectKBest)
$$ $$
Production monitoring & model drift detection
$$ $$
User authentication for Streamlit app
$$ $$
Integration with hospital EHR systems
$$ $$
Calibration for clinical decision thresholds
📄 License
MIT License — Feel free to use for research/education
👤 Author
Dr. Uthman Babatunde | Medical Doctor & Applied AI Engineer
📧 Email
🔗 LinkedIn | GitHub
📖 References
Strack, B., et al. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates" BioMed Research International
Scikit-learn Documentation: SMOTE
UCI ML Repository Diabetes Dataset
