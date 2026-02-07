import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Mycotoxin Detection AI",
    layout="centered"
)

st.title("🌾 Mycotoxin Contamination Prediction")
st.write("Predict the risk of mycotoxin contamination based on environmental and storage conditions.")

# ===============================
# MODEL PATH
# ===============================
MODEL_PATH = "model_pipeline.joblib"

# ===============================
# LOAD MODEL SAFELY
# ===============================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.isfile(MODEL_PATH):
    st.error("❌ Model file `model_pipeline.joblib` not found in repository root.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ===============================
# USER INPUTS
# ===============================
st.subheader("📥 Enter Input Parameters")

temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
storage_days = st.number_input("Storage Days", min_value=0, max_value=365, value=30)
moisture_content = st.number_input("Moisture Content (%)", min_value=0.0, max_value=100.0, value=12.0)
crop_type = st.selectbox(
    "Crop Type",
    ["maize", "rice", "wheat", "groundnut"]
)

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

    try:
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = None

        if prediction == 1:
            if probability is not None:
                st.error(f"⚠️ Contaminated — Risk Score: {probability:.2f}")
            else:
                st.error("⚠️ Contaminated")
        else:
            if probability is not None:
                st.success(f"✅ Safe — Risk Score: {probability:.2f}")
            else:
                st.success("✅ Safe")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
