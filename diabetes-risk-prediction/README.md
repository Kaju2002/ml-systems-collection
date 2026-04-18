# Diabetes Risk Screener — Streamlit Web Application

Professional ML-powered web app for diabetes risk prediction with clinical measurements.

**🌐 Live:** https://ml-systems-collection-dqvmbvew53o6jdegnkapp5g.streamlit.app/

---

## 📊 Quick Start

### Run Locally

```bash
cd diabetes-risk-prediction
pip install -r requirements.txt
streamlit run app.py
```

### Access

- Local: `http://localhost:8501`
- Cloud: https://ml-systems-collection-dqvmbvew53o6jdegnkapp5g.streamlit.app/

---

## 🎯 Features

✅ **8 Clinical Input Fields** — Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Pedigree, Age  
✅ **Risk Prediction** — Logistic Regression model (78% accuracy, 0.83 ROC-AUC)  
✅ **Probability Visualization** — Risk score with progress bar  
✅ **Personalized Recommendations** — Health actions based on risk level  
✅ **Input Summary** — Grid display of all entered measurements  
✅ **Medical Disclaimer** — Educational purposes notice

---

## 📁 Files

| File                | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `app.py`            | Streamlit web interface (production-ready) |
| `model.pkl`         | Trained Logistic Regression model          |
| `scaler.pkl`        | StandardScaler for feature normalization   |
| `diabetes.csv`      | Original dataset (768 rows, 8 features)    |
| `requirements.txt`  | Python dependencies                        |
| `data_analysis.py`  | EDA & preprocessing pipeline               |
| `model_training.py` | Model training & evaluation                |
| `plot_*.png`        | Visualization outputs                      |

---

## 🤖 Model Info

**Algorithm:** Logistic Regression  
**Training Data:** 614 samples (80% split)  
**Test Data:** 154 samples (20% split)  
**Accuracy:** 78%  
**ROC-AUC:** 0.83  
**Threshold:** 0.4 (optimized for medical sensitivity)

### Class Distribution

- Non-Diabetic (0): 65%
- Diabetic (1): 35%

---

## 🎨 Design & UX

- **Theme:** Light, professional (no icons)
- **Styling:** Custom CSS with Inter & JetBrains Mono fonts
- **Layout:** Centered, responsive design
- **Colors:**
  - High Risk (Red): #DC2626
  - Low Risk (Green): #16A34A
  - Neutral (Blue): #2563EB

---

## 🔧 Technical Stack

```
Frontend: Streamlit
ML Library: scikit-learn
Data Processing: pandas, numpy
Visualization: matplotlib, seaborn
Caching: @st.cache_resource
Deployment: Streamlit Cloud
```

---

## ⚠️ Medical Disclaimer

**This is an educational tool only.** Not for medical diagnosis. Always consult a qualified physician.

---

## 📈 Model Development

See parent `README.md` for full ML pipeline details including:

- Data preprocessing steps
- Feature scaling methodology
- Threshold optimization results
- Training/evaluation metrics

---

## 🚀 Deployment Notes

✅ Uses absolute file paths (`os.path`) for cloud compatibility  
✅ Cached model/scaler loading with `@st.cache_resource`  
✅ Auto-deployed on GitHub push via Streamlit Cloud  
✅ CORS-enabled FastAPI backend for future React integration

---

**Status:** ✅ Production Ready | **Updated:** April 2026
