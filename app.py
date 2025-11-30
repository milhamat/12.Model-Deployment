import streamlit as st
import pandas as pd
from joblib import load

# ==============================
# Load model (cached)
# ==============================
@st.cache_resource
def load_model():
    return load("./Experiments/model.joblib")   # pastikan nama file benar

model = load_model()

# ==============================
# Column order (harus sama seperti saat training)
# ==============================
COLUMNS = [
    'Unnamed: 0',
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'smoking_history',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]

# ==============================
# Streamlit UI
# ==============================
st.title("Diabetes Prediction (Demo)")

st.markdown("### Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    # unnamed_0 = st.number_input("ID / Unnamed: 0", min_value=0, value=1, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])

with col2:
    heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
    smoking_history = st.selectbox(
        "Smoking History",
        ["non-smoker", "past_smoker", "current"]
    )
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.5, step=0.1)
    hba1c = st.number_input("HbA1c level (glycated Hemoglobin)", min_value=0.0, max_value=20.0, value=5.7, step=0.1)
    glucose = st.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=140.0, step=1.0)

# Tombol prediksi
if st.button("Predict"):
    # Buat DataFrame satu baris
    data = {
        # 'Unnamed: 0': unnamed_0,
        'Unnamed: 0': 1,  # Dummy value for Unnamed: 0
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }

    df = pd.DataFrame([data])[COLUMNS]  # pastikan urutan kolom sesuai

    # Prediksi
    prediction = model.predict(df)[0]

    st.markdown("### Hasil Prediksi")
    st.write("Raw prediction:", int(prediction))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        st.write("Probabilities (class 0, class 1):", proba)
        st.progress(float(proba[1]))

    # Optional: interpretasi label
    label = "DIABETES" if int(prediction) == 1 else "TIDAK DIABETES"
    st.success(f"Model memprediksi: **{label}**")
