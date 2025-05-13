import streamlit as st
import requests

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("Breast Cancer Predictor")
st.write("Enter histological measurements to predict tumor malignancy and understand feature impact with SHAP.")

# Input fields
area_error = st.number_input("Area Error", min_value=0.0, format="%.4f")
worst_area = st.number_input("Worst Area", min_value=0.0, format="%.4f")
worst_texture = st.number_input("Worst Texture", min_value=0.0, format="%.4f")
worst_radius = st.number_input("Worst Radius", min_value=0.0, format="%.4f")
worst_concavity = st.number_input("Worst Concavity", min_value=0.0, format="%.4f")

if st.button("Predict"):
    payload = {
        "area_error": area_error,
        "worst_area": worst_area,
        "worst_texture": worst_texture,
        "worst_radius": worst_radius,
        "worst_concavity": worst_concavity
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()

            st.success(f"Prediction: **{result['prediction_label']}**")
            st.write("### 🧪 Probabilities")
            st.write(result["probabilities"])

            st.write("### 🧠 SHAP Feature Importance")
            shap_values = result["shap_values"]
            st.bar_chart(shap_values)

        else:
            st.error("Prediction failed. Check backend logs.")

    except Exception as e:
        st.error(f"Error connecting to FastAPI backend: {e}")
