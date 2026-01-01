import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ================== CONFIG ==================
CSV_PATH = r"D:\Downloads\framingham.csv"   # update if needed
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

FEATURE_COLUMNS = [
    "age",
    "male",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "glucose"
]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

st.set_page_config(
    page_title="Heart Disease Risk Meter",
    page_icon="❤️",
    layout="centered"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title-text {
    font-size: 38px;
    font-weight: 800;
    color: #1f2c56;
    text-align: center;
}

.subtitle-text {
    font-size: 18px;
    color: #4f5d75;
    text-align: center;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

.result {
    font-size: 22px;
    font-weight: 700;
    text-align: center;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 12px;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #1e40af;
}
</style>
""", unsafe_allow_html=True)

# ================== TRAIN MODEL ==================
def train_model():
    st.info("Training model for the first time...")

    data = pd.read_csv(CSV_PATH)

    for col in data.columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    X = data[FEATURE_COLUMNS]
    y = data["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.success(f"Model trained successfully | ROC-AUC: {auc:.3f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

# ================== LOAD OR TRAIN ==================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return train_model()
    else:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_model()

# ================== RISK METER ==================
def show_risk_meter(prob):
    risk_percent = prob * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': '#00c853'},     # Very Low
                {'range': [25, 50], 'color': '#ffd600'},   # Low
                {'range': [50, 75], 'color': '#ff9100'},   # Moderate
                {'range': [75, 100], 'color': '#d50000'}   # High
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': risk_percent
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# ================== UI ==================
st.markdown('<div class="title-text">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">10-Year Coronary Heart Disease (CHD) Risk Assessment</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 50)
    male = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cigs = st.number_input("Cigarettes per Day", 0, 100, 0)
    bmi = st.number_input("BMI", value=25.0)

with col2:
    chol = st.number_input("Total Cholesterol (mg/dL)", value=200)
    sysBP = st.number_input("Systolic BP (mmHg)", value=120)
    diaBP = st.number_input("Diastolic BP (mmHg)", value=80)
    glucose = st.number_input("Glucose (mg/dL)", value=80)

st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("🔍 Predict Risk")

# ================== PREDICTION ==================
if predict_btn:
    features = np.array([[age, male, cigs, chol, sysBP, diaBP, bmi, glucose]])
    features = scaler.transform(features)
    prob = model.predict_proba(features)[0][1]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="result">Risk Assessment</div>', unsafe_allow_html=True)

    show_risk_meter(prob)

    if prob < 0.25:
        st.success("🟢 Very Low Risk of CHD")
    elif prob < 0.50:
        st.info("🟡 Low Risk of CHD")
    elif prob < 0.75:
        st.warning("🟠 Moderate Risk of CHD")
    else:
        st.error("🔴 High Risk of CHD")

    st.markdown(
        f"<p style='text-align:center; font-size:16px;'>Predicted Probability: <b>{prob:.2f}</b></p>",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:14px;'>
Built with ❤️ using Machine Learning & Streamlit<br>
For educational purposes only – not a medical diagnosis
</div>
""", unsafe_allow_html=True)
