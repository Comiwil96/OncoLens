from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

# Load your pipeline
model = joblib.load("models/logreg_pipeline.pkl")

# Replace with your actual final features
final_features = [
    'area error', 
    'worst area', 
    'worst texture', 
    'worst radius', 
    'worst concavity'
]

# Create FastAPI app
app = FastAPI(
    title="Breast Cancer Predictor",
    description="Predicts if a tumor is malignant or benign based on histological measurements.",
    version="1.0.0"
)

# Define the data model
class PatientData(BaseModel):
    area_error: float
    worst_area: float
    worst_texture: float
    worst_radius: float
    worst_concavity: float

# Initialize SHAP explainer — done lazily on first request
explainer = None

@app.post("/predict")
def predict(data: PatientData):
    global explainer
    
    # Mapping dictionary to convert API parameter names to model feature names
    feature_map = {
        "area_error": "area error",
        "worst_area": "worst area",
        "worst_texture": "worst texture",
        "worst_radius": "worst radius",
        "worst_concavity": "worst concavity"
    }
    
    # Convert input to DataFrame with proper feature names
    input_data = data.model_dump()
    
    # Create DataFrame with mapped feature names
    input_df = pd.DataFrame([{feature_map[key]: value for key, value in input_data.items()}])
    
    # Make prediction
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0].tolist()

    # SHAP values calculation
    if explainer is None:
        # Initialize the explainer using a sample from the preprocessed data
        preprocessed_input = model.named_steps['preprocessor'].transform(input_df)
        explainer = shap.Explainer(model.named_steps['classifier'], preprocessed_input)

    # Get SHAP values for this prediction
    preprocessed_input = model.named_steps['preprocessor'].transform(input_df)
    shap_values = explainer(preprocessed_input)
    
    # Map the SHAP values to the original feature names
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