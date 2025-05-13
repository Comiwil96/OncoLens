# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load your pipeline
model = joblib.load("models/logreg_pipeline.pkl")

# Replace with your actual final features
final_features = [
    'mean_radius', 
    'mean_texture', 
    'mean_smoothness', 
    'mean_symmetry', 
    'mean_fractal_dimension'
]

# Create FastAPI app
app = FastAPI(
    title="Breast Cancer Predictor",
    description="Predicts if a tumor is malignant or benign based on diagnostic features.",
    version="1.0.0"
)

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
    proba = model.predict_proba(input_df)[0]
    
    return {
        "prediction": int(pred),
        "prediction_label": "Malignant" if pred == 1 else "Benign",
        "probabilities": {
            "Benign": round(proba[0], 4),
            "Malignant": round(proba[1], 4)
        }
    }
