
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- CUSTOM CSS ----------------


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 5%, #172a52 0%, transparent 40%),
        radial-gradient(ellipse at 90% 90%, #30194d 0%, transparent 38%),
        #080d19;
    font-family: 'Inter', sans-serif;
    color: #ffffff;
}

.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #142b48, #202044, #291c43);
    border: 1px solid rgba(135, 155, 255, 0.25);
    border-radius: 25px;
    padding: 38px;
    text-align: center;
    box-shadow: 0 15px 50px rgba(0,0,0,.3);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff, #6be7ff, #b69cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #b7c6df;
    font-size: 15px;
}

/* Form card */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(23, 34, 57, 0.78);
    border: 1px solid rgba(126, 156, 215, 0.22);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 10px 35px rgba(0,0,0,.2);
}

/* Section headings */
h2, h3 {
    color: #82dcff !important;
    font-weight: 700 !important;
}

/* Labels */
label, .stMarkdown p {
    color: #dce7f7 !important;
}

/* Input fields */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {
    background: #101b30 !important;
    border: 1px solid #354b70 !important;
    border-radius: 10px !important;
    color: white !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

/* Slider */
div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 10px;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    border: 1px solid #65e3ff;
    background: linear-gradient(100deg, #087fca, #7153dc);
    color: white;
    font-size: 18px;
    font-weight: 800;
    box-shadow: 0 5px 25px rgba(65, 119, 255, .3);
    transition: all .3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(100, 140, 255, .5);
    border: 1px solid #ffffff;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0c1425;
    border-right: 1px solid #293958;
}

/* Result panels */
.result-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 25px;
    font-size: 23px;
    font-weight: 800;
}

.low-risk {
    background: linear-gradient(120deg, #103d3b, #123e56);
    border: 1px solid #35e5c3;
    color: #74ffe0;
    box-shadow: 0 0 25px rgba(53,229,195,.12);
}

.high-risk {
    background: linear-gradient(120deg, #501d39, #3b244e);
    border: 1px solid #ff6c9c;
    color: #ffabc7;
    box-shadow: 0 0 25px rgba(255,108,156,.12);
}

hr {
    border-color: #30415e;
}
/* FIX TOP WHITE SPACE */

header[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stAppViewContainer"] {
    background: #080d19 !important;
}

[data-testid="stMain"] {
    background: #080d19 !important;
}

.block-container {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ❤️ HeartCare")
    st.markdown("---")
    st.markdown("### About Project")
    st.write(
        "A Machine Learning application "
        "that predicts heart disease risk "
        "using the K-Nearest Neighbors algorithm."
    )

    st.markdown("---")
    st.markdown("### Model Information")
    st.write("**Algorithm:** KNN")
    st.write("**Preprocessing:** StandardScaler")
    st.write("**Framework:** Streamlit")

    st.markdown("---")
    st.caption("Developed by Darshan")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>♥ Heart Intelligence</h1>
    <p>ADVANCED AI-POWERED HEALTH RISK ASSESSMENT</p>
    <p>Smart insights. Better awareness. Healthier decisions.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI-powered heart health risk assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">📋 Patient Information</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT FORM ----------------
with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Details")

        age = st.slider("Age", 18, 100, 40)

        sex = st.selectbox("Sex", ["M", "F"])

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 100, 600, 200
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure", 80, 200, 120
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", [0, 1]
        )

    with col2:
        st.subheader("Heart Test Details")

        chest_pain = st.selectbox(
            "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"]
        )

        resting_ecg = st.selectbox(
            "Resting ECG", ["Normal", "ST", "LVH"]
        )

        max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

        exercise_angina = st.selectbox(
            "Exercise Angina", ["Y", "N"]
        )

        oldpeak = st.slider(
            "Oldpeak", 0.0, 6.0, 1.0
        )

        st_slope = st.selectbox(
            "ST Slope", ["Up", "Flat", "Down"]
        )

    st.markdown("---")

    predict_button = st.button("🔍 Predict Heart Disease Risk")

# ---------------- PREDICTION ----------------
if predict_button:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing dummy columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Correct feature order
    input_df = input_df[expected_columns]

    # Scale only features used when fitting scaler
    scale_columns = list(scaler.feature_names_in_)

    input_df[scale_columns] = scaler.transform(
        input_df[scale_columns]
    )

    # Final prediction
    prediction = model.predict(
        input_df[expected_columns]
    )[0]

    st.markdown("---")
    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.markdown("""
        <div class="result-box high-risk">
            ⚠️ Higher Predicted Risk of Heart Disease
            <p style="font-size:14px;">
                Please consult a qualified healthcare professional.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-box low-risk">
            ✅ Lower Predicted Risk of Heart Disease
            <p style="font-size:14px;">
                This prediction does not guarantee absence of disease.
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "⚕️ Educational ML project only. "
    "Not a substitute for professional medical diagnosis."
)
