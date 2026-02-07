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
st.write("Predict the risk of mycotoxin contamination based on environmental and storage conditions.")

# ===============================
# MODEL PATH & LOADING
# ===============================
# Ensure this matches your filename on GitHub exactly
MODEL_PATH = "model_pipeline.joblib"

@st.cache_resource
def load_model(path):
    """Loads the model pipeline once and caches it to improve performance."""
    return joblib.load(path)

# Check if file exists before trying to load
if not os.path.isfile(MODEL_PATH):
    st.error(f"❌ Model file `{MODEL_PATH}` not found in the repository.")
    st.info("Please ensure you have uploaded your trained model file to the root of your GitHub repo.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    # Optional: Sidebar status
    st.sidebar.success("✅ AI Model Loaded")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ===============================
# USER INPUTS
# ===============================
st.subheader("📥 Enter Input Parameters")

# Use columns to make the UI look cleaner
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    moisture_content = st.number_input("Moisture Content (%)", min_value=0.0, max_value=100.0, value=12.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    storage_days = st.number_input("Storage Days", min_value=0, max_value=365, value=30)
    crop_type = st.selectbox(
        "Crop Type",
        ["maize", "rice", "wheat", "groundnut"]
    )

# ===============================
# PREDICTION LOGIC
# ===============================
if st.button("🔍 Run Prediction Analysis", use_container_width=True):
    # Prepare the data in the exact format the Pipeline expects
    input_df = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "storage_days": storage_days,
        "moisture_content": moisture_content,
        "crop_type": crop_type
    }])

    try:
        # Get class prediction (0 or 1)
        prediction = model.predict(input_df)[0]
        
        # Get probability if the model supports it
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = None

        st.markdown("---")
        st.subheader("Results")

        if prediction == 1:
            if probability is not None:
                st.error(f"### 🚨 Result: HIGH RISK ({probability * 100:.2f}%)")
            else:
                st.error("### 🚨 Result: Contaminated")
            st.write("Recommendations: Check storage ventilation and reduce moisture content immediately.")
        else:
            if probability is not None:
                st.success(f"### ✅ Result: SAFE ({probability * 100:.2f}%)")
            else:
                st.success("### ✅ Result: Safe")
            st.write("The current conditions are within safe thresholds for mycotoxin prevention.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Check if your input feature names match those used during model training.")
