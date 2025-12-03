# ❤️ Heart Disease Classification - Quick Start Guide

## 🎯 What You Have

Your complete ML system is ready! Here's what's been created:

### ✅ Completed Tasks

1. **Data Preprocessing** ✓
   - Loaded `heart(in).csv` (302 valid records after cleaning)
   - Removed missing values and duplicates
   - Standardized 13 numerical features
   - Split into 80% training (241 samples) and 20% testing (61 samples)

2. **Model Training** ✓
   - **Logistic Regression**: 80.33% accuracy ⭐
   - **SVM**: 77.05% accuracy
   - Best model saved and ready for predictions

3. **Backend API** ✓
   - FastAPI with `/predict` endpoint
   - Supports single and batch predictions
   - Auto-scaling with saved StandardScaler
   - Swagger documentation included

4. **Frontend UI** ✓
   - Streamlit with beautiful interface
   - Single prediction mode
   - Batch upload mode
   - Sample testing mode

---

## 🚀 How to Run

### **Option 1: Quick Start (Easiest)**

```bash
cd /workspaces/AIML_Task

# Terminal 1: Start API server
python 3_fastapi_app.py

# Terminal 2: Start Streamlit UI
streamlit run 4_streamlit_app.py
```

### **Option 2: Using Setup Script**

```bash
chmod +x setup.sh
./setup.sh
```

Then run the API and Streamlit commands above.

---

## 🌐 Access Points

Once running, access from:

| Component | URL | Purpose |
|-----------|-----|---------|
| **Streamlit UI** | http://localhost:8501 | Interactive prediction interface |
| **FastAPI Docs** | http://localhost:8000/docs | API testing & documentation |
| **API Health** | http://localhost:8000/health | Health check |
| **API Info** | http://localhost:8000/model-info | Model details |

---

## 💡 How to Use

### **Method 1: Streamlit UI (Recommended for Testing)**

1. Open http://localhost:8501
2. Select mode from sidebar:
   - **Single Prediction**: Enter patient data via form
   - **Batch Upload**: Upload CSV with multiple patients
   - **Sample Test**: Test with pre-configured examples

### **Method 2: FastAPI (Recommended for Integration)**

1. Open http://localhost:8000/docs
2. Expand `/predict` endpoint
3. Click "Try it out"
4. Enter patient data JSON
5. Click "Execute"

### **Method 3: Python/cURL**

**Python:**
```python
import requests

url = "http://localhost:8000/predict"
patient = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
    "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
    "exang": 0, "oldpeak": 1, "slope": 2, "ca": 2, "thal": 3
}
response = requests.post(url, json=patient)
print(response.json())
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1,"slope":2,"ca":2,"thal":3}'
```

---

## 📊 Model Performance

| Metric | Logistic Regression | SVM |
|--------|-------------------|-----|
| **Accuracy** | **80.33%** ⭐ | 77.05% |
| **Precision** | 81.11% | 77.14% |
| **Recall** | 79.39% | 81.82% |
| **F1-Score** | 80.24% | 79.41% |
| **ROC-AUC** | 88.87% | 84.52% |

✨ **Best Model**: Logistic Regression (Selected for production)

---

## 📁 Files Created

```
/workspaces/AIML_Task/
├── 1_preprocessing.py          # Data cleaning & standardization
├── 2_train_model.py            # Model training
├── 3_fastapi_app.py            # Backend API
├── 4_streamlit_app.py          # Frontend UI
├── GUIDE.md                    # Detailed documentation
├── requirements.txt            # Python dependencies
│
├── data/                       # Preprocessed datasets
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── scaler.pkl              # StandardScaler (for normalizing input)
│   └── feature_names.pkl       # Feature order
│
├── models/                     # Trained ML models
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   ├── best_model.pkl          # ⭐ Main model used by API
│   └── best_model_name.txt
│
└── plots/                      # Performance visualizations
    ├── confusion_matrices.png
    └── roc_curves.png
```

---

## 🎓 Understanding the Prediction

### **Response Format**
```json
{
  "prediction": 0,              // 0 = No disease, 1 = Disease present
  "confidence": 0.85,           // Probability (0-1)
  "model": "Logistic Regression",
  "risk_level": "moderate"      // low, moderate, or high
}
```

### **Risk Levels**
- 🟢 **Low** (< 60% confidence): Minimal disease risk
- 🟡 **Moderate** (60-80%): Moderate disease risk  
- 🔴 **High** (> 80%): High disease risk

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| **API not connecting** | Ensure `python 3_fastapi_app.py` is running in terminal 1 |
| **Port 8000 already in use** | `lsof -ti:8000 \| xargs kill -9` |
| **Streamlit won't connect to API** | Check FastAPI logs, ensure it's on `localhost:8000` |
| **Module not found error** | Run: `pip install -r requirements.txt` |
| **Data/Models missing** | Run: `python 1_preprocessing.py && python 2_train_model.py` |

---

## 🧪 Test Cases

### **Healthy Person** (Should predict: No Disease)
```json
{
  "age": 40, "sex": 0, "cp": 0, "trestbps": 120,
  "chol": 200, "fbs": 0, "restecg": 0, "thalach": 120,
  "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 1
}
```

### **At-Risk Patient** (May predict: Disease)
```json
{
  "age": 60, "sex": 1, "cp": 3, "trestbps": 150,
  "chol": 300, "fbs": 1, "restecg": 2, "thalach": 100,
  "exang": 1, "oldpeak": 3.5, "slope": 0, "ca": 3, "thal": 2
}
```

### **High-Risk Patient** (Should predict: Disease)
```json
{
  "age": 70, "sex": 1, "cp": 1, "trestbps": 160,
  "chol": 350, "fbs": 1, "restecg": 2, "thalach": 90,
  "exang": 1, "oldpeak": 4.0, "slope": 0, "ca": 4, "thal": 3
}
```

---

## 📈 Feature Descriptions

| Feature | Range | Description |
|---------|-------|-------------|
| age | 29-77 | Patient age |
| sex | 0-1 | 0=female, 1=male |
| cp | 0-3 | Chest pain type |
| trestbps | 90-200 | Resting blood pressure (mmHg) |
| chol | 126-564 | Serum cholesterol (mg/dl) |
| fbs | 0-1 | Fasting blood sugar > 120 (0=no, 1=yes) |
| restecg | 0-2 | Resting ECG results |
| thalach | 60-202 | Maximum heart rate achieved |
| exang | 0-1 | Exercise-induced angina (0=no, 1=yes) |
| oldpeak | 0-6.2 | ST depression |
| slope | 0-2 | ST segment slope |
| ca | 0-4 | Major vessels count |
| thal | 0-3 | Thalassemia type |

---

## 🎯 Workflow Diagram

```
Dataset (heart(in).csv)
    ↓
1_preprocessing.py
├─ Load data
├─ Clean (remove nulls/duplicates)
├─ Standardize features
└─ Save: data/X_train, X_test, scaler
    ↓
2_train_model.py
├─ Train Logistic Regression
├─ Train SVM
├─ Compare metrics
└─ Save: models/best_model.pkl (80.33% accuracy)
    ↓
System Ready!
├─ FastAPI: 3_fastapi_app.py
└─ Streamlit: 4_streamlit_app.py
    ↓
User Makes Prediction
├─ Input patient data
├─ API scales with scaler.pkl
├─ Predict with best_model.pkl
└─ Return: prediction + confidence + risk_level
```

---

## ✨ Best Practices Used

✓ **Data Preprocessing**: Standardization for model convergence
✓ **Train-Test Split**: 80-20 with stratification
✓ **Model Comparison**: Evaluated both SVM and Logistic Regression
✓ **Serialization**: Models saved as `.pkl` for reproducibility
✓ **API Design**: RESTful endpoints with clear response format
✓ **Frontend**: User-friendly UI with multiple input methods
✓ **Validation**: Pydantic models for input validation
✓ **Documentation**: Swagger docs auto-generated from FastAPI

---

## 📝 Next Steps (Optional Enhancements)

- [ ] Deploy API to cloud (AWS, Azure, GCP)
- [ ] Add authentication to API
- [ ] Create database to store predictions
- [ ] Add data validation rules
- [ ] Implement model versioning
- [ ] Add monitoring and logging
- [ ] Create CI/CD pipeline
- [ ] Add unit tests

---

## 🆘 Need Help?

1. Check **GUIDE.md** for detailed documentation
2. Review **FastAPI docs** at http://localhost:8000/docs
3. Check terminal logs for error messages
4. Verify all files exist in `data/` and `models/` directories

---

## 🎉 Summary

**You now have a complete ML system!**

- ✅ Data preprocessed and cleaned
- ✅ Models trained (80.33% accuracy)
- ✅ API running with endpoints
- ✅ Interactive Streamlit UI
- ✅ Ready for deployment

**Start with:** 
```bash
python 3_fastapi_app.py  # Terminal 1
streamlit run 4_streamlit_app.py  # Terminal 2
```

Then visit: http://localhost:8501

**Happy Predicting! ❤️**
