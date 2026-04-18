# ============================================================
#   DIABETES RISK SCREENER — STREAMLIT WEB APP (Proposal UI)
#   Run command : streamlit run app.py
# ============================================================


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 1 — IMPORT LIBRARIES                           │
# └─────────────────────────────────────────────────────────┘

import streamlit as st
import numpy as np
import pickle
import os


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 2 — PAGE CONFIG (Must be FIRST streamlit call) │
# └─────────────────────────────────────────────────────────┘

st.set_page_config(
    page_title="Diabetes Risk Screener",
    page_icon=None,  # Icons removed for clean proposal UI
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 3 — CUSTOM CSS STYLING (Clean Proposal UI)     │
# └─────────────────────────────────────────────────────────┘

st.markdown("""
<style>
/* ─── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Global Reset & Light Theme Proposal ──────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #F8FAFC;
    color: #1E293B;
}

/* ─── Hide Streamlit Branding ───────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ─── Hero Banner (Clean) ───────────────────────────────── */
.hero-banner {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 32px 32px 28px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 8px 0;
    letter-spacing: -0.3px;
}
.hero-title span { color: #2563EB; }
.hero-subtitle {
    font-size: 0.9rem;
    color: #475569;
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
}
.hero-badge {
    display: inline-block;
    background: #EEF2FF;
    color: #1E40AF;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}

/* ─── Section Card (Clean White Cards) ─────────────────── */
.section-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #475569;
    margin: 0 0 20px 0;
}

/* ─── Number Inputs (Clean) ────────────────────────────── */
[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    color: #0F172A !important;
    font-size: 0.9rem !important;
    padding: 8px 12px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
[data-testid="stNumberInput"] label {
    color: #334155 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ─── Predict Button (Solid Professional) ──────────────── */
[data-testid="stButton"] > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px) !important;
}

/* ─── Result Cards (Flat, Informative) ─────────────────── */
.result-high {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
}
.result-low {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin: 0 0 6px 0;
}
.result-label.high { color: #DC2626; }
.result-label.low  { color: #16A34A; }
.result-heading {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 6px 0;
}
.result-sub {
    font-size: 0.85rem;
    color: #475569;
    margin: 0 0 20px 0;
}
.risk-score-wrap {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 12px;
}
.risk-score-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    line-height: 1;
}
.risk-score-number.high { color: #DC2626; }
.risk-score-number.low  { color: #16A34A; }
.risk-score-label {
    font-size: 0.9rem;
    color: #475569;
    font-weight: 400;
}
.risk-bar-bg {
    background: #E2E8F0;
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 20px;
}
.risk-bar-fill-high {
    height: 100%;
    border-radius: 8px;
    background: #DC2626;
}
.risk-bar-fill-low {
    height: 100%;
    border-radius: 8px;
    background: #16A34A;
}
.actions-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}
.action-pill {
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.75rem;
    color: #1E293B;
}

/* ─── Summary Grid (Clean Metrics) ─────────────────────── */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 8px;
}
.summary-item {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 8px;
    text-align: center;
}
.summary-item-label {
    font-size: 0.6rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.summary-item-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: #0F172A;
}

/* ─── Model Info Chips (Neutral) ───────────────────────── */
.info-chips {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.info-chip {
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.75rem;
    color: #334155;
}
.info-chip span {
    color: #0F172A;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Disclaimer (Subtle) ──────────────────────────────── */
.disclaimer {
    background: #FEFCE8;
    border: 1px solid #FDE047;
    border-radius: 10px;
    padding: 12px 18px;
    margin-top: 24px;
    font-size: 0.75rem;
    color: #713F12;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 4 — LOAD MODEL & SCALER                        │
# └─────────────────────────────────────────────────────────┘

@st.cache_resource
def load_artifacts():
    """
    Loads model.pkl and scaler.pkl only ONCE and caches them.
    Without @st.cache_resource, files reload on every interaction.
    Uses absolute path relative to script location for cloud deployment.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    scaler_path = os.path.join(script_dir, "scaler.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 5 — HERO HEADER (No Icons)                     │
# └─────────────────────────────────────────────────────────┘

st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">Clinical ML Tool</div>
    <h1 class="hero-title">Diabetes <span>Risk</span> Screener</h1>
    <p class="hero-subtitle">
        Enter patient clinical measurements below. The model analyses the inputs using 
        Logistic Regression and returns a personalised diabetes risk score.
    </p>
</div>
""", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 6 — MODEL INFO STRIP (Clean)                   │
# └─────────────────────────────────────────────────────────┘

st.markdown("""
<div class="section-card">
    <p class="section-title">Model Details</p>
    <div class="info-chips">
        <div class="info-chip">Algorithm &nbsp;<span>Logistic Regression</span></div>
        <div class="info-chip">Accuracy &nbsp;<span>~78%</span></div>
        <div class="info-chip">ROC-AUC &nbsp;<span>~0.83</span></div>
        <div class="info-chip">Dataset &nbsp;<span>Pima Indians (768 rows)</span></div>
        <div class="info-chip">Features &nbsp;<span>8 clinical inputs</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 7 — PATIENT INPUT FORM (No Icons)              │
# └─────────────────────────────────────────────────────────┘

st.markdown('<div class="section-card"><p class="section-title">Patient Clinical Data</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0, max_value=20, value=1, step=1,
        help="Number of times the patient has been pregnant"
    )
    glucose = st.number_input(
        "Glucose  (mg/dL)",
        min_value=0, max_value=300, value=120,
        help="Plasma glucose (2-hr oral glucose tolerance test). Normal: 70–99"
    )
    blood_pressure = st.number_input(
        "Blood Pressure  (mm Hg)",
        min_value=0, max_value=200, value=70,
        help="Diastolic blood pressure. Normal: 60–80"
    )
    skin_thickness = st.number_input(
        "Skin Thickness  (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skinfold thickness. Normal: 10–40"
    )

with col2:
    insulin = st.number_input(
        "Insulin  (μU/mL)",
        min_value=0, max_value=1000, value=80,
        help="2-hour serum insulin. Normal: 16–166"
    )
    bmi = st.number_input(
        "BMI  (kg/m²)",
        min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index. Normal: 18.5–24.9"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.47, step=0.01,
        help="Likelihood of diabetes based on family history. Range: 0.08–2.42"
    )
    age = st.number_input(
        "Age  (years)",
        min_value=1, max_value=120, value=33, step=1
    )

st.markdown('</div>', unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 8 — PREDICTION FUNCTION                        │
# └─────────────────────────────────────────────────────────┘

def predict(pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age):
    """
    Takes 8 clinical inputs → scales them → runs logistic regression.
    Returns:
        pred (int)  : 0 = Non-Diabetic, 1 = Diabetic
        prob (float): Probability of diabetes as a percentage (0–100)
    """
    raw    = np.array([[pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]])
    scaled = scaler.transform(raw)
    pred   = model.predict(scaled)[0]
    prob   = model.predict_proba(scaled)[0][1] * 100
    return pred, prob


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 9 — PREDICT BUTTON (No Icon)                   │
# └─────────────────────────────────────────────────────────┘

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
predict_clicked = st.button("Analyse Risk Now", use_container_width=True)


# ┌─────────────────────────────────────────────────────────┐
# │  SECTION 10 — RESULTS DISPLAY (No Icons, Clean)         │
# └─────────────────────────────────────────────────────────┘

if predict_clicked:

    pred, prob = predict(
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    )

    prob_int = int(prob)

    # ── High Risk Result (No Icons) ───────────────────────
    if pred == 1:
        actions = [
            "Consult an endocrinologist",
            "Check fasting blood glucose",
            "Reduce carbohydrate intake",
            "Increase physical activity",
            "Discuss medication options"
        ]
        st.markdown(f"""
        <div class="result-high">
            <p class="result-label high">High Risk Detected</p>
            <h2 class="result-heading">Likely Diabetic</h2>
            <p class="result-sub">The model detected elevated diabetes indicators in this patient's profile.</p>
            <div class="risk-score-wrap">
                <span class="risk-score-number high">{prob:.1f}</span>
                <span class="risk-score-label">% probability of diabetes</span>
            </div>
            <div class="risk-bar-bg">
                <div class="risk-bar-fill-high" style="width:{prob_int}%"></div>
            </div>
            <p style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin:0 0 8px">
                Recommended Actions
            </p>
            <div class="actions-wrap">
                {''.join(f'<span class="action-pill">{a}</span>' for a in actions)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Low Risk Result (No Icons) ────────────────────────
    else:
        actions = [
            "Maintain a balanced diet",
            "Stay physically active",
            "Annual blood sugar screening",
            "Prioritise quality sleep",
            "Stay well hydrated"
        ]
        st.markdown(f"""
        <div class="result-low">
            <p class="result-label low">Low Risk</p>
            <h2 class="result-heading">Likely Non-Diabetic</h2>
            <p class="result-sub">No strong diabetes indicators were found in this patient's clinical profile.</p>
            <div class="risk-score-wrap">
                <span class="risk-score-number low">{prob:.1f}</span>
                <span class="risk-score-label">% probability of diabetes</span>
            </div>
            <div class="risk-bar-bg">
                <div class="risk-bar-fill-low" style="width:{prob_int}%"></div>
            </div>
            <p style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin:0 0 8px">
                Preventive Recommendations
            </p>
            <div class="actions-wrap">
                {''.join(f'<span class="action-pill">{a}</span>' for a in actions)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input Summary Grid (No Icons) ─────────────────────
    fields = [
        ("Pregnancies",    pregnancies,    ""),
        ("Glucose",        glucose,        "mg/dL"),
        ("Blood Press.",   blood_pressure, "mmHg"),
        ("Skin Thick.",    skin_thickness, "mm"),
        ("Insulin",        insulin,        "μU/mL"),
        ("BMI",            bmi,            "kg/m²"),
        ("Pedigree",       dpf,            ""),
        ("Age",            age,            "yrs"),
    ]

    grid_items = ""
    for label, val, unit in fields:
        display = f"{val:.2f}" if isinstance(val, float) else str(val)
        grid_items += f"""
        <div class="summary-item">
            <div class="summary-item-label">{label}</div>
            <div class="summary-item-value">
                {display}
                <span style="font-size:0.6rem;color:#64748B"> {unit}</span>
            </div>
        </div>"""

    st.markdown(f"""
    <div class="section-card">
        <p class="section-title">Input Summary</p>
        <div class="summary-grid">{grid_items}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer (No Icon) ──────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        <strong>Medical Disclaimer:</strong> This tool is built for educational and portfolio purposes only.
        It is <strong>not</strong> a certified diagnostic tool. Predictions are based on a statistical model 
        and should never replace professional medical advice. Always consult a qualified physician for 
        any health-related concerns.
    </div>
    """, unsafe_allow_html=True)