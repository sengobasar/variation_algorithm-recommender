from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load model and feature names
model = joblib.load("breast_cancer_tree.pkl")
feature_names = joblib.load("feature_names.pkl")

app = FastAPI(title="Breast Cancer Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input structure
class PatientInput(BaseModel):
    features: list[float]  # List of 30 numbers (must match training features)

@app.post("/predict")
def predict(data: PatientInput):
    try:
        # Ensure we have exactly 30 features
        if len(data.features) != 30:
            return {"error": f"Expected 30 features, got {len(data.features)}"}
            
        df = pd.DataFrame([data.features], columns=feature_names)
        prediction = model.predict(df)[0]
        result = "Malignant" if prediction == 0 else "Benign"
        return {"prediction": result, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
