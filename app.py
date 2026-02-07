import streamlit as st
import pandas as pd
import pickle

# ===============================
# LOAD MODEL
# ===============================
model = pickle.load(open("model_pipeline.pkl", "rb"))

st.set_page_config(page_title="Mycotoxin Detection AI", layout="centered")

st.title("🧪 Mycotoxin Contamination Detection")
st.write("Enter environmental and storage parameters to assess contamination risk.")

# ===============================
# USER INPUTS
# ===============================
temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
storage_days = st.number_input("Storage Duration (days)", 0, 365, 30)
moisture_content = st.number_input("Moisture Content (%)", 0.0, 30.0, 12.5)
crop_type = st.selectbox(
    "Crop Type",
    ["maize", "wheat", "rice", "groundnut"]
)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Contamination Risk"):
    input_data = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "storage_days": storage_days,
        "moisture_content": moisture_content,
        "crop_type": crop_type
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Contaminated\n\nRisk Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Safe\n\nRisk Probability: {probability*100:.2f}%")
