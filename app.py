
import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Mycotoxin Detection", layout="centered")

st.title("🌾 AI-Based Mycotoxin Risk Detection")
st.write("Enter environmental and storage conditions")

# Inputs
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 28.0)
humidity = st.slider("Humidity (%)", 30.0, 100.0, 75.0)
rainfall = st.slider("Rainfall (mm)", 0.0, 400.0, 150.0)
storage_days = st.slider("Storage Days", 1, 365, 120)
moisture = st.slider("Moisture Content (%)", 8.0, 25.0, 15.0)

crop = st.selectbox("Crop Type", ["Maize", "Rice", "Sorghum", "Wheat"])

crop_rice = 1 if crop == "Rice" else 0
crop_sorghum = 1 if crop == "Sorghum" else 0
crop_wheat = 1 if crop == "Wheat" else 0

if st.button("🔍 Predict Risk"):
    user_input = np.array([[temperature, humidity, rainfall,
                            storage_days, moisture,
                            crop_rice, crop_sorghum, crop_wheat]])

    user_input_scaled = scaler.transform(user_input)
    risk_prob = model.predict_proba(user_input_scaled)[0][1] * 100

    if risk_prob > 60:
        st.error(f"🚨 HIGH RISK ({risk_prob:.2f}%)")
    elif risk_prob > 30:
        st.warning(f"⚠️ MEDIUM RISK ({risk_prob:.2f}%)")
    else:
        st.success(f"✅ LOW RISK ({risk_prob:.2f}%)")
