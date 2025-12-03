# 🚀 How to Use the Heart Disease Prediction System

## Quick Access Guide

### **For First-Time Users** 👈 START HERE
1. Read: `QUICKSTART.md` (5 minutes)
2. Read: This file (you are here!)
3. Run: Follow "Getting Started" section below

### **For Understanding the Concepts**
1. Read: `README.md` (deep technical explanation)
2. Read: `GUIDE.md` (step-by-step breakdown)

### **For Integration/Development**
1. Read: `README.md` → System Architecture section
2. Review: `3_fastapi_app.py` → API endpoints
3. Review: `4_streamlit_app.py` → UI implementation

---

## Getting Started (3 Steps)

### **Step 1: Verify Setup** (2 minutes)
```bash
cd /workspaces/AIML_Task
python verify_system.py
# Should show: "🎉 ALL CHECKS PASSED!"
```

### **Step 2: Start FastAPI Server** (Terminal 1)
```bash
python 3_fastapi_app.py
# Should show: "Uvicorn running on http://0.0.0.0:8000"
```

### **Step 3: Start Streamlit UI** (Terminal 2)
```bash
streamlit run 4_streamlit_app.py
# Should show: "You can now view your Streamlit app in your browser at http://localhost:8501"
```

---

## Using the System

### **Method 1: Streamlit Web UI (Easiest) ✅**

**Open Browser:**
```
http://localhost:8501
```

**Mode 1: Single Prediction**
1. Enter patient data using form:
   - Slider for Age (29-77)
   - Radio buttons for Sex, Chest Pain type, etc.
2. Click "🔍 Predict" button
3. See prediction results with confidence score

**Mode 2: Batch Upload**
1. Prepare CSV file with columns:
   ```
   age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
   52,1,0,125,212,0,1,168,0,1,2,2,3
   60,1,1,140,298,0,1,122,1,4.2,1,3,3
   ```
2. Click "Choose CSV file" button
3. Click "🔍 Predict All"
4. Download results CSV

**Mode 3: Sample Test**
1. Expand predefined examples
2. Click "Predict" on each example
3. See expected outputs

### **Method 2: FastAPI Web Interface**

**Open Browser:**
```
http://localhost:8000/docs
```

**Test API:**
1. Expand the `/predict` endpoint
2. Click "Try it out"
3. Edit request body with patient data
4. Click "Execute"
5. See response with prediction

**Alternative: Use `/predict-batch`**
1. For multiple patients at once

### **Method 3: Python Code**

**Single Prediction:**
```python
import requests

url = "http://localhost:8000/predict"

patient_data = {
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

response = requests.post(url, json=patient_data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

**Batch Predictions:**
```python
import requests
import pandas as pd

url = "http://localhost:8000/predict-batch"

# Read CSV file
df = pd.read_csv('patients.csv')

# Convert to list of dicts
patients = df.to_dict('records')

# Send request
response = requests.post(url, json={"patients": patients})
results = response.json()

# Process results
for result in results['results']:
    print(f"Patient age {result['patient_data']['age']}: "
          f"Prediction = {result['prediction']}, "
          f"Confidence = {result['confidence']:.2%}")
```

### **Method 4: cURL Commands**

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | json_pp
```

**Health Check:**
```bash
curl http://localhost:8000/health | json_pp
```

**Model Info:**
```bash
curl http://localhost:8000/model-info | json_pp
```

---

## Understanding Predictions

### **Response Structure**
```json
{
  "prediction": 0,              // 0 = No disease, 1 = Disease
  "confidence": 0.8142,         // Probability (0-1)
  "model": "Logistic Regression",
  "risk_level": "moderate"      // low / moderate / high
}
```

### **Risk Levels Explained**
- 🟢 **Low** (<60% confidence): Minimal disease risk
- 🟡 **Moderate** (60-80%): Moderate disease risk
- 🔴 **High** (>80%): High disease risk

### **Interpretation**
```
Prediction = 0, Confidence = 0.85, Risk = moderate
→ Model predicts NO disease with 85% confidence
→ But 15% chance of disease (moderate risk)

Prediction = 1, Confidence = 0.92, Risk = high
→ Model predicts DISEASE with 92% confidence
→ High risk - should seek medical attention
```

---

## Example Use Cases

### **Case 1: Health Checkup**
A 55-year-old patient gets a checkup:

**Input:**
```json
{
  "age": 55, "sex": 1, "cp": 1, "trestbps": 135,
  "chol": 240, "fbs": 0, "restecg": 1, "thalach": 110,
  "exang": 1, "oldpeak": 2.5, "slope": 1, "ca": 1, "thal": 2
}
```

**Output:**
```json
{
  "prediction": 1,
  "confidence": 0.78,
  "model": "Logistic Regression",
  "risk_level": "moderate"
}
```

**Interpretation:** Moderate disease risk. Recommend further evaluation with cardiologist.

---

### **Case 2: Healthy Individual**
Young, healthy person for preventive screening:

**Input:**
```json
{
  "age": 35, "sex": 0, "cp": 0, "trestbps": 110,
  "chol": 180, "fbs": 0, "restecg": 0, "thalach": 130,
  "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 1
}
```

**Output:**
```json
{
  "prediction": 0,
  "confidence": 0.91,
  "model": "Logistic Regression",
  "risk_level": "low"
}
```

**Interpretation:** Low disease risk. Continue regular exercise and healthy diet.

---

### **Case 3: High-Risk Patient**
Senior patient with multiple risk factors:

**Input:**
```json
{
  "age": 70, "sex": 1, "cp": 1, "trestbps": 160,
  "chol": 300, "fbs": 1, "restecg": 2, "thalach": 85,
  "exang": 1, "oldpeak": 4.0, "slope": 0, "ca": 3, "thal": 3
}
```

**Output:**
```json
{
  "prediction": 1,
  "confidence": 0.95,
  "model": "Logistic Regression",
  "risk_level": "high"
}
```

**Interpretation:** High disease risk. Immediate medical intervention recommended.

---

## Testing Workflows

### **Workflow 1: Single Patient Testing**
```
1. Open Streamlit UI (http://localhost:8501)
2. Select "Single Prediction" mode
3. Fill in patient data via form
4. Click "Predict"
5. Review results with confidence and risk level
6. Compare with medical opinion
```

### **Workflow 2: Bulk Patient Screening**
```
1. Prepare CSV with patient data (13 columns)
2. Open Streamlit UI
3. Select "Batch Upload" mode
4. Upload CSV file
5. Click "Predict All"
6. Download results CSV
7. Analyze predictions in bulk
```

### **Workflow 3: API Integration Testing**
```
1. Open FastAPI docs (http://localhost:8000/docs)
2. Test /predict endpoint with sample patient
3. Test /predict-batch with multiple patients
4. Test /health endpoint
5. Test /model-info endpoint
6. Review response format and status codes
```

### **Workflow 4: Sample Data Testing**
```
1. Open Streamlit UI
2. Select "Sample Test" mode
3. Expand "Healthy Person" example
4. Click "Predict" → See low risk
5. Expand "At-Risk Patient" example
6. Click "Predict" → See moderate risk
7. Expand "High-Risk Patient" example
8. Click "Predict" → See high risk
```

---

## Troubleshooting

### **Problem: Connection Refused**
```
Error: ConnectionRefusedError at http://localhost:8000
```
**Solution:**
- Ensure FastAPI server is running: `python 3_fastapi_app.py`
- Check port 8000 is available: `lsof -ti:8000`

### **Problem: Module Not Found**
```
Error: ModuleNotFoundError: No module named 'fastapi'
```
**Solution:**
```bash
pip install -r requirements.txt
```

### **Problem: File Not Found**
```
Error: FileNotFoundError: data/scaler.pkl
```
**Solution:**
```bash
python 1_preprocessing.py
python 2_train_model.py
```

### **Problem: Port Already in Use**
```
Error: Address already in use: ('0.0.0.0', 8000)
```
**Solution:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port (edit 3_fastapi_app.py)
python 3_fastapi_app.py --port 8001
```

### **Problem: Streamlit Won't Connect to API**
```
Error: Connection refused when making predictions
```
**Solution:**
1. Check FastAPI is running on port 8000
2. Check API URL in streamlit code (should be http://localhost:8000)
3. In 4_streamlit_app.py, change: `API_URL = "http://localhost:8000"`

---

## Performance Tips

### **For Better Predictions**
1. Ensure input data is accurate and complete
2. All 13 features must be provided
3. Values should be in realistic ranges
4. Use consistent units (mmHg for pressure, mg/dl for cholesterol)

### **For Production Deployment**
1. Use Docker container for consistency
2. Add authentication to API
3. Implement rate limiting
4. Add monitoring and logging
5. Use load balancer for multiple instances
6. Store predictions in database
7. Set up continuous model monitoring

### **For Model Improvement**
1. Collect more training data
2. Tune hyperparameters (GridSearchCV)
3. Try ensemble methods (Random Forest, XGBoost)
4. Implement cross-validation
5. Handle class imbalance
6. Add feature engineering

---

## Sample Data for Testing

### **Test Data 1: Normal Person**
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
40,0,0,120,200,0,0,120,0,0.0,1,0,1
```

### **Test Data 2: At-Risk Person**
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
60,1,1,150,300,1,2,100,1,3.5,0,3,2
```

### **Test Data 3: High-Risk Person**
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
70,1,0,160,350,1,2,90,1,4.0,0,4,3
```

---

## API Endpoints Reference

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/` | GET | Health check | No |
| `/health` | GET | Health status | No |
| `/model-info` | GET | Model details | No |
| `/predict` | POST | Single prediction | No |
| `/predict-batch` | POST | Batch predictions | No |
| `/docs` | GET | Swagger UI | No |
| `/redoc` | GET | ReDoc UI | No |

---

## Feature Reference

| Feature | Range | Unit | Description |
|---------|-------|------|-------------|
| age | 29-77 | years | Patient age |
| sex | 0-1 | binary | 0=female, 1=male |
| cp | 0-3 | type | Chest pain type |
| trestbps | 90-200 | mmHg | Resting blood pressure |
| chol | 126-564 | mg/dl | Serum cholesterol |
| fbs | 0-1 | binary | Fasting blood sugar > 120 |
| restecg | 0-2 | type | Resting ECG |
| thalach | 60-202 | bpm | Max heart rate achieved |
| exang | 0-1 | binary | Exercise-induced angina |
| oldpeak | 0-6.2 | mm | ST depression |
| slope | 0-2 | type | ST segment slope |
| ca | 0-4 | count | Major vessels |
| thal | 0-3 | type | Thalassemia |

---

## Next Steps

✅ **You're ready to use the system!**

1. Run the servers (Terminal 1 & 2)
2. Access Streamlit UI (http://localhost:8501)
3. Make predictions with real patient data
4. Explore API documentation (http://localhost:8000/docs)
5. Review model performance in visualizations

**For deployment:** See GUIDE.md → "Deployment Guide"

---

**Happy Predicting! ❤️**
