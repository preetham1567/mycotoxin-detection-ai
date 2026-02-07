import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Mycotoxin Detection AI", layout="centered")

st.title("🌾 Mycotoxin Contamination Prediction")

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "model_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("❌ model_pipeline.joblib not found")
    st.stop()

model = load_model()
st.success("✅ Model loaded successfully")

# ===============================
# USER INPUT
# ===============================
temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
storage_days = st.number_input("Storage Days", 0, 365, 30)
moisture_content = st.number_input("Moisture Content (%)", 0.0, 100.0, 12.0)
crop_type = st.selectbox("Crop Type", ["maize", "rice", "wheat", "groundnut"])

# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "storage_days": storage_days,
        "moisture_content": moisture_content,
        "crop_type": crop_type
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Contaminated (Risk: {probability:.2f})")
    else:
        st.success(f"✅ Safe (Risk: {probability:.2f})")
