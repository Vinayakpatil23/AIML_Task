# 📑 Heart Disease Classification - Complete Index

## 🎯 Quick Navigation

### **I'm in a hurry** ⏱️
→ Read: `QUICKSTART.md` (5 min)
→ Run: Step 1-3 in this file
→ Done!

### **I want to understand everything** 📚
→ Read: `README.md` (20 min)
→ Read: `GUIDE.md` (30 min)
→ Read: `USAGE.md` (15 min)
→ Explore: Visualizations in `plots/`

### **I want to integrate this** 🔧
→ Read: `README.md` → "System Architecture"
→ Review: `3_fastapi_app.py` (API code)
→ Review: `USAGE.md` → "Method 3: Python Code"

### **I want to improve the models** 🚀
→ Read: `README.md` → "Key Takeaways"
→ Explore: `2_train_model.py` (model training)
→ See: `plots/` for current performance

---

## 📂 File Structure & Purpose

### **Documentation Files** 📖
| File | Size | Duration | Purpose |
|------|------|----------|---------|
| **QUICKSTART.md** | 8.5 KB | 5 min | Get started quickly |
| **GUIDE.md** | 13 KB | 30 min | Understand each step |
| **README.md** | 19 KB | 20 min | Complete technical explanation |
| **USAGE.md** | 9 KB | 15 min | How to use the system |
| **INDEX.md** | This file | 2 min | Navigation guide |

### **Python Scripts** 🐍
| Script | Size | Purpose |
|--------|------|---------|
| **1_preprocessing.py** | 2.9 KB | Load, clean, and standardize data |
| **2_train_model.py** | 5.5 KB | Train models and evaluate |
| **3_fastapi_app.py** | 6.5 KB | REST API server |
| **4_streamlit_app.py** | 11 KB | Web interface |
| **verify_system.py** | 6.8 KB | System verification (22 checks) |

### **Data Files** 📊 (in `data/` directory)
| File | Size | Content |
|------|------|---------|
| **X_train.csv** | 61 KB | Training features (241 samples) |
| **X_test.csv** | 15.5 KB | Test features (61 samples) |
| **y_train.csv** | 489 B | Training labels |
| **y_test.csv** | 129 B | Test labels |
| **scaler.pkl** | 948 B | StandardScaler (data normalization) |
| **feature_names.pkl** | 115 B | Feature ordering |

### **Model Files** 🤖 (in `models/` directory)
| File | Size | Model |
|------|------|-------|
| **best_model.pkl** | 1 KB | ⭐ Logistic Regression (80.33%) |
| **logistic_regression.pkl** | 1 KB | Backup model |
| **svm_model.pkl** | 18 KB | Alternative model (77.05%) |
| **best_model_name.txt** | 19 B | Model type reference |

### **Visualizations** 📈 (in `plots/` directory)
| File | Size | Content |
|------|------|---------|
| **confusion_matrices.png** | 27 KB | Model performance comparison |
| **roc_curves.png** | 40 KB | ROC curve analysis |

### **Configuration** ⚙️
| File | Purpose |
|------|---------|
| **requirements.txt** | Python package dependencies |
| **setup.sh** | Automated setup script |
| **heart(in).csv** | Original dataset (1,026 records) |

---

## 🚀 How to Run

### **Full Pipeline (Recommended)**

```bash
# Step 1: Navigate to project
cd /workspaces/AIML_Task

# Step 2: Verify system (optional but recommended)
python verify_system.py

# Step 3: Start API Server (Terminal 1)
python 3_fastapi_app.py

# Step 4: Start Web UI (Terminal 2)
streamlit run 4_streamlit_app.py

# Step 5: Access in browser
# UI:      http://localhost:8501
# API:     http://localhost:8000/docs
```

### **Quick Setup**

```bash
./setup.sh  # Runs preprocessing and training
# Then follow steps 3-5 above
```

---

## 📊 System Overview

```
DATASET (1,026 records)
    ↓
1_preprocessing.py
├─ Remove nulls & duplicates (→ 302 records)
├─ Standardize 13 features
├─ 80-20 train-test split
└─ Save: data/*.csv, data/*.pkl
    ↓
2_train_model.py
├─ Train Logistic Regression
├─ Train SVM
├─ Compare metrics
└─ Save best model + visualizations
    ↓
3_fastapi_app.py (Backend)
├─ Load model & scaler
├─ Serve 5 endpoints
└─ Listen on :8000
    ↓
4_streamlit_app.py (Frontend)
├─ Single prediction mode
├─ Batch upload mode
├─ Sample test mode
└─ Listen on :8501
```

---

## 🎓 What You'll Learn

### **Data Science**
- ✓ Data cleaning & validation
- ✓ Feature standardization (StandardScaler)
- ✓ Train-test splitting with stratification
- ✓ Model training & evaluation
- ✓ Classification metrics (Accuracy, Precision, Recall, F1, AUC)

### **Machine Learning**
- ✓ Logistic Regression
- ✓ Support Vector Machines (SVM)
- ✓ Model comparison & selection
- ✓ Confusion matrices & ROC curves
- ✓ Binary classification

### **Backend Development**
- ✓ FastAPI framework
- ✓ REST API design
- ✓ Data validation (Pydantic)
- ✓ Automatic API documentation
- ✓ Error handling

### **Frontend Development**
- ✓ Streamlit framework
- ✓ Interactive UI components
- ✓ File upload handling
- ✓ Data visualization
- ✓ API integration

### **DevOps & Deployment**
- ✓ Model serialization (pickle)
- ✓ Dependency management
- ✓ System verification
- ✓ Production-ready code

---

## 📈 Performance Results

| Metric | Logistic Regression ⭐ | SVM |
|--------|----------------------|-----|
| Accuracy | 80.33% | 77.05% |
| Precision | 81.11% | 77.14% |
| Recall | 79.39% | 81.82% |
| F1-Score | 80.24% | 79.41% |
| ROC-AUC | 88.87% | 84.52% |

**Selected Model**: Logistic Regression (Best overall performance)

---

## 🧪 Testing the System

### **Method 1: Streamlit UI** (Easiest)
1. Open http://localhost:8501
2. Select "Single Prediction" mode
3. Fill in patient data
4. Click "Predict"
5. See results with confidence & risk level

### **Method 2: FastAPI Swagger**
1. Open http://localhost:8000/docs
2. Expand `/predict` endpoint
3. Click "Try it out"
4. Enter sample data
5. Click "Execute"

### **Method 3: Python Script**
```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
        "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
        "exang": 0, "oldpeak": 1, "slope": 2, "ca": 2, "thal": 3
    }
)
print(response.json())
```

### **Method 4: Sample Data**
All sample test patients are included in Streamlit UI ("Sample Test" mode)

---

## 🔍 Feature Descriptions

13 input features for prediction:

1. **age** (29-77) - Patient age in years
2. **sex** (0-1) - 0=female, 1=male
3. **cp** (0-3) - Chest pain type
4. **trestbps** (90-200 mmHg) - Resting blood pressure
5. **chol** (126-564 mg/dl) - Serum cholesterol
6. **fbs** (0-1) - Fasting blood sugar > 120
7. **restecg** (0-2) - Resting ECG results
8. **thalach** (60-202 bpm) - Max heart rate achieved
9. **exang** (0-1) - Exercise-induced angina
10. **oldpeak** (0-6.2 mm) - ST depression
11. **slope** (0-2) - ST segment slope
12. **ca** (0-4) - Major vessels
13. **thal** (0-3) - Thalassemia type

**Output**: 0 = No disease, 1 = Disease present

---

## ❓ FAQ

**Q: What if I get a "Connection refused" error?**
A: Make sure FastAPI server is running in Terminal 1

**Q: Can I modify the models?**
A: Yes! Edit `2_train_model.py` to try different algorithms or parameters

**Q: How do I deploy this to production?**
A: See README.md → "Next Steps (Optional Enhancements)"

**Q: Can I use this with real patient data?**
A: Yes, but ensure data quality and validate with medical professionals

**Q: What's the difference between the two models?**
A: Logistic Regression is simpler & faster (80.33% accuracy)
   SVM captures non-linear patterns (77.05% accuracy)

---

## 🎯 Verification Checklist

Before using the system, verify:

- [ ] Run `python verify_system.py` (should pass 22/22 checks)
- [ ] Check `data/` directory has 6 files
- [ ] Check `models/` directory has 4 files
- [ ] Check `plots/` directory has 2 visualizations
- [ ] All Python scripts are present
- [ ] Requirements installed: `pip install -r requirements.txt`

---

## 📞 Getting Help

| Issue | Solution | File |
|-------|----------|------|
| "How do I get started?" | Read QUICKSTART.md | QUICKSTART.md |
| "How does this work?" | Read GUIDE.md | GUIDE.md |
| "I want full details" | Read README.md | README.md |
| "How do I use it?" | Read USAGE.md | USAGE.md |
| "Connection errors?" | Check USAGE.md → Troubleshooting | USAGE.md |
| "Code not working?" | Run verify_system.py | verify_system.py |

---

## 🎉 What You Now Have

✅ **Complete ML Pipeline**
- Data preprocessing
- Model training (2 algorithms)
- Model evaluation

✅ **REST API**
- 5 endpoints
- Auto-generated documentation
- Input validation

✅ **Web Interface**
- Single predictions
- Batch processing
- Sample testing

✅ **Documentation**
- 5 comprehensive guides
- Code examples
- Troubleshooting

✅ **Production Ready**
- Trained models saved
- Reproducible pipeline
- System verification

---

## 🚀 Next Steps

1. **Start the system** (run API + Streamlit)
2. **Test with sample data** (use Streamlit UI)
3. **Understand the code** (read documentation)
4. **Make predictions** (use your own data)
5. **Integrate if needed** (use FastAPI endpoints)

---

## 📝 Summary

You have a **complete, production-ready** Heart Disease Classification system with:

- ✓ Clean, preprocessed data (302 valid records)
- ✓ Trained models (Logistic Regression 80.33% accuracy)
- ✓ REST API (FastAPI with Swagger docs)
- ✓ Web UI (Streamlit with 3 modes)
- ✓ Full documentation (5 guides)
- ✓ System verification (22 checks passed)

**Everything is ready. Start by running the API and Streamlit servers!**

---

**Happy Learning & Predicting! ❤️**
