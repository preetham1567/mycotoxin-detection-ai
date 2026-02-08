import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Mycotoxin Detection AI",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Mycotoxin Contamination Prediction")
st.write("Predict mycotoxin risk based on environmental and storage conditions.")

# ===============================
# MODEL LOADING
# ===============================
MODEL_PATH = "model_pipeline.joblib"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error("❌ model_pipeline.joblib not found in repository root")
    st.stop()

model = load_model(MODEL_PATH)
st.sidebar.success("✅ Model loaded")

# ===============================
# USER INPUT
# ===============================
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
    moisture_content = st.number_input("Moisture Content (%)", 0.0, 100.0, 12.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    storage_days = st.number_input("Storage Days", 0, 365, 30)
    crop_type = st.selectbox("Crop Type", ["maize", "rice", "wheat", "groundnut"])

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "storage_days": storage_days,
        "moisture_content": moisture_content,
        "crop_type": crop_type
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"🚨 Contaminated (Risk: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Safe (Risk: {prob*100:.2f}%)")
