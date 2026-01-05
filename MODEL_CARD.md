# Model Card: Hospital Readmission Risk Predictor

## 🧠 Model Details
* **Developer:** Uthman Babatunde (MD, AI Engineer)
* **Model Architecture:** Random Forest Classifier (200 estimators).
* **Input Data:** UCI Diabetes 130-US Hospitals Dataset (1999-2008).
* **Target:** 30-day all-cause hospital readmission.

## ⚕️ Intended Use
This model is designed as a **clinical decision support tool** to stratify patients by readmission risk at the time of discharge. It is intended to flag high-risk patients who require enhanced discharge planning (e.g., home health coordination, early follow-up).

## ⚖️ Risk Calibration & Thresholds
Unlike standard ML classification (which uses a 0.5 threshold), this model uses **clinically calibrated thresholds** based on the dataset's baseline readmission rate (~11%).
* **High Risk (>30%):** Patient has >3x the baseline risk. Action: Intensive intervention.
* **Moderate Risk (15-30%):** Patient has elevated risk. Action: Standard follow-up.
* **Low Risk (<15%):** Risk is at or below baseline.

## ⚠️ Limitations
* **Temporal Drift:** Data is from 1999-2008; clinical practices have evolved.
* **Feature Scope:** Limited to 7 key administrative features; does not include unstructured clinical notes or social determinants of health (SDOH).
* **Demographics:** Trained on US hospital data; external validation required for non-US populations.
