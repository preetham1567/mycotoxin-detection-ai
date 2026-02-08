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
st.write(
    "Predict the risk of mycotoxin contamination based on environmental "
    "and storage conditions."
)

# ===============================
# MODEL PATH & LOADING
# ===============================
MODEL_PATH = "model_pipeline.joblib"  # must exist in repo root

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.isfile(MODEL_PATH):
    st.error(f"❌ Model file `{MODEL_PATH}` not found.")
    st.info("Upload the model file to the GitHub repo root.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model Loaded")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ===============================
# USER INPUTS
# ===============================
st.subheader("📥 Enter Input Parameters")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
    moisture_content = st.number_input("Moisture Content (%)", 0.0, 100.0, 12.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    storage_days = st.number_input("Storage Days", 0, 365, 30)
    crop_type = st.selectbox(
        "Crop Type",
        ["maize", "rice", "wheat", "groundnut"]
    )

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Run Prediction Analysis", use_container_width=True):
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

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("🧪 Result")

        if prediction == 1:
            if probability is not None:
                st.error(f"🚨 HIGH RISK ({probability * 100:.2f}%)")
            else:
                st.error("🚨 HIGH RISK")
            st.write(
                "⚠️ Recommendation: Improve ventilation and reduce "
                "moisture immediately."
            )
        else:
            if probability is not None:
                st.success(f"✅ SAFE ({probability * 100:.2f}%)")
            else:
                st.success("✅ SAFE")
            st.write(
                "✅ Current conditions are within safe thresholds."
            )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info(
            "Ensure feature names match exactly those used during training."
        )
