# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

# Load your pipeline
model = joblib.load("models/logreg_pipeline.pkl")

# Replace with your actual final features
final_features = [
    'area_error', 
    'worst_area', 
    'worst_texture', 
    'worst-radius', 
    'worst_concavity'
]

# Create FastAPI app
app = FastAPI(
    title="Breast Cancer Predictor",
    description="Predicts if a tumor is malignant or benign based on histological measurements.",
    version="1.0.0"
)

# Init SHAP explainer
explainer = shap.Explainer(model.named_steps['classifier'])
shap_input = model.named_steps['preprocessor'].transform(input_df)


# Define the data model
class PatientData(BaseModel):
    area_error: float
    worst_area: float
    worst_texture: float
    worst_radius: float
    worst_concavity: float

# Predict endpoint
@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])[final_features]
    
    # Make prediction
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0].tolist()

    # SHAP values
    shap_values = explainer(input_df)
    feature_importance = dict(zip(final_features, shap_values.values[0].tolist()))
    
    return {
        "prediction": int(pred),
        "prediction_label": "Malignant" if pred == 1 else "Benign",
        "probabilities": {
            "Benign": round(proba[0], 4),
            "Malignant": round(proba[1], 4)
        },
        "shap_values": feature_importance
    }
