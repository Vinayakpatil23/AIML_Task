"""
Heart Disease Classification - FastAPI Backend
API endpoints for model predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ==================== INITIALIZE APP ====================
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease risk using ML models",
    version="1.0"
)

# ==================== LOAD MODELS AND DATA ====================
# Load best model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load feature names
with open('data/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load model name
with open('models/best_model_name.txt', 'r') as f:
    model_name = f.read().strip()

print(f"✓ Model loaded: {model_name}")
print(f"✓ Features: {feature_names}")

# ==================== PYDANTIC MODELS ====================
class PatientData(BaseModel):
    """Input patient data for prediction"""
    age: float
    sex: int  # 1=male, 0=female
    cp: int  # chest pain type (0-3)
    trestbps: float  # resting blood pressure
    chol: float  # serum cholesterol
    fbs: int  # fasting blood sugar > 120
    restecg: int  # resting ECG (0-2)
    thalach: float  # max heart rate achieved
    exang: int  # exercise induced angina
    oldpeak: float  # ST depression
    slope: int  # slope of ST segment (0-2)
    ca: int  # number of vessels (0-4)
    thal: int  # thalassemia (0-3)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1,
                "slope": 2,
                "ca": 2,
                "thal": 3
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int  # 0=no disease, 1=disease present
    confidence: float  # probability of prediction
    model: str  # model name used
    risk_level: str  # low, moderate, high

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData]

# ==================== HEALTH CHECK ====================
@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": model_name,
        "features": feature_names,
        "version": "1.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True
    }

# ==================== SINGLE PREDICTION ====================
@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Predict heart disease risk for a single patient
    
    Returns:
    - prediction: 0 = No disease, 1 = Disease present
    - confidence: Probability of the prediction
    - risk_level: Low/Moderate/High based on confidence
    """
    try:
        # Convert to DataFrame with feature names in correct order
        data_dict = patient.dict()
        X = pd.DataFrame([data_dict], columns=feature_names)
        
        # Standardize using loaded scaler
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = int(model.predict(X_scaled)[0])
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            confidence = float(model.predict_proba(X_scaled)[0][prediction])
        else:
            # For SVM without probability
            confidence = float(abs(model.decision_function(X_scaled)[0]))
        
        # Determine risk level
        if confidence < 0.6:
            risk_level = "low"
        elif confidence < 0.8:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model=model_name,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ==================== BATCH PREDICTION ====================
@app.post("/predict-batch")
def predict_batch(request: BatchPredictionRequest):
    """
    Predict heart disease risk for multiple patients
    
    Returns: List of predictions
    """
    try:
        results = []
        
        for patient in request.patients:
            # Convert to DataFrame
            data_dict = patient.dict()
            X = pd.DataFrame([data_dict], columns=feature_names)
            
            # Standardize
            X_scaled = scaler.transform(X)
            
            # Predict
            prediction = int(model.predict(X_scaled)[0])
            
            # Get probability
            if hasattr(model, 'predict_proba'):
                confidence = float(model.predict_proba(X_scaled)[0][prediction])
            else:
                confidence = float(abs(model.decision_function(X_scaled)[0]))
            
            # Risk level
            if confidence < 0.6:
                risk_level = "low"
            elif confidence < 0.8:
                risk_level = "moderate"
            else:
                risk_level = "high"
            
            results.append({
                "patient_data": data_dict,
                "prediction": prediction,
                "confidence": confidence,
                "risk_level": risk_level
            })
        
        return {
            "total_patients": len(request.patients),
            "model": model_name,
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ==================== MODEL INFO ====================
@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    return {
        "model_name": model_name,
        "features": feature_names,
        "num_features": len(feature_names),
        "scaler_type": type(scaler).__name__,
        "prediction_classes": ["No Heart Disease (0)", "Heart Disease Present (1)"]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting API server with {model_name}...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
