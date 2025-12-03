# Heart Disease Classification - Complete Guide

## 📋 Project Overview

This project implements a complete ML pipeline for predicting heart disease risk using:
- **Data Processing**: Cleaning & standardization
- **ML Models**: Logistic Regression & SVM
- **Backend API**: FastAPI for predictions
- **Frontend**: Streamlit for interactive testing

---

## 🎯 Task Breakdown

### 1. **Data Loading & Exploration** (`1_preprocessing.py`)
- ✓ Loads `heart(in).csv` dataset
- ✓ Displays dataset shape and structure
- ✓ Identifies features and target variable

**Output**: Dataset overview
```
Dataset shape: (1026, 14)
- 13 features (age, sex, cp, trestbps, etc.)
- 1 target variable (presence of heart disease)
```

---

### 2. **Data Cleaning** (`1_preprocessing.py`)
- ✓ Removes rows with missing values using `.dropna()`
- ✓ Removes duplicate rows using `.drop_duplicates()`
- ✓ Verifies data quality

**Process**:
```python
# Remove missing values
df_clean = df.dropna()

# Remove duplicates
df_clean = df_clean.drop_duplicates()
```

---

### 3. **Feature Standardization** (`1_preprocessing.py`)
- ✓ Uses `StandardScaler` to normalize numerical columns
- ✓ Formula: (value - mean) / std_dev
- ✓ Essential for SVM and Logistic Regression

**Process**:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Benefits**:
- Centers data around 0 with std dev = 1
- Improves model convergence
- Makes coefficients comparable

---

### 4. **Model Training** (`2_train_model.py`)

#### **Logistic Regression**
- Binary classification algorithm
- Output: Probability between 0-1
- Fast training and inference
- Good baseline model

#### **SVM (Support Vector Machine)**
- Finds optimal hyperplane to separate classes
- RBF kernel captures non-linear patterns
- Robust to outliers
- Better for complex decision boundaries

**Training Process**:
```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
```

**Evaluation Metrics**:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Area under receiver operating characteristic curve

---

### 5. **Model Persistence** (`2_train_model.py`)
- ✓ Saves best performing model to `models/best_model.pkl`
- ✓ Saves scaler to `data/scaler.pkl`
- ✓ Saves feature names to `data/feature_names.pkl`
- ✓ All models use pickle serialization

**Files Created**:
```
models/
├── logistic_regression.pkl
├── svm_model.pkl
├── best_model.pkl
└── best_model_name.txt

data/
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
├── scaler.pkl
└── feature_names.pkl
```

---

### 6. **FastAPI Backend** (`3_fastapi_app.py`)

#### **Endpoints**

**1. Health Check**
```
GET /
GET /health
```
Returns API status and model information.

**2. Single Prediction**
```
POST /predict
```
Input: Patient data (13 features)
```json
{
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
```

Output:
```json
{
  "prediction": 0,
  "confidence": 0.85,
  "model": "Logistic Regression",
  "risk_level": "moderate"
}
```

**3. Batch Prediction**
```
POST /predict-batch
```
Input: Array of patients
Output: Array of predictions

**4. Model Information**
```
GET /model-info
```
Returns model details and features.

#### **Key Features**
- ✓ Data validation using Pydantic models
- ✓ Automatic scaling using saved scaler
- ✓ Error handling and HTTP exceptions
- ✓ Confidence scoring
- ✓ Risk level classification
- ✓ Auto-generated API documentation at `/docs`

---

### 7. **Streamlit Frontend** (`4_streamlit_app.py`)

#### **Three Modes**

**1. Single Prediction**
- Interactive form for patient data
- Sliders for numerical inputs
- Dropdowns for categorical inputs
- Real-time prediction with confidence display

**2. Batch Upload**
- Upload CSV files with multiple patients
- Bulk predictions
- Download results as CSV

**3. Sample Test**
- Pre-configured sample patients
- Test with predefined cases
- Quick demonstration

#### **Features**
- ✓ Beautiful UI with custom styling
- ✓ Real-time API connectivity check
- ✓ Risk level visualization (🟢🟡🔴)
- ✓ Responsive design
- ✓ Export predictions to CSV

---

## 🚀 Installation & Setup

### **Step 1: Install Dependencies**
```bash
cd /workspaces/AIML_Task
pip install -r requirements.txt
```

### **Step 2: Run Preprocessing & Training**
```bash
python 1_preprocessing.py
python 2_train_model.py
```

This will create:
- Preprocessed data files in `data/`
- Trained models in `models/`
- Visualizations in `plots/`

### **Step 3: Start FastAPI Server** (Terminal 1)
```bash
python 3_fastapi_app.py
```

The API will be available at:
- Application: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### **Step 4: Start Streamlit Frontend** (Terminal 2)
```bash
streamlit run 4_streamlit_app.py
```

Access Streamlit at: `http://localhost:8501`

### **Quick Start Script**
```bash
chmod +x setup.sh
./setup.sh
```

Then run in separate terminals:
```bash
python 3_fastapi_app.py
streamlit run 4_streamlit_app.py
```

---

## 📊 Data Features Explanation

| Feature | Range | Description |
|---------|-------|-------------|
| age | 29-77 | Patient age in years |
| sex | 0,1 | 0=female, 1=male |
| cp | 0-3 | Chest pain type (0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic) |
| trestbps | 90-200 | Resting blood pressure (mmHg) |
| chol | 126-564 | Serum cholesterol (mg/dl) |
| fbs | 0,1 | Fasting blood sugar > 120 mg/dl (0=no, 1=yes) |
| restecg | 0-2 | Resting ECG results |
| thalach | 60-202 | Maximum heart rate achieved |
| exang | 0,1 | Exercise induced angina (0=no, 1=yes) |
| oldpeak | 0-6.2 | ST depression induced by exercise |
| slope | 0-2 | Slope of ST segment |
| ca | 0-4 | Number of major vessels (0-3) colored by fluoroscopy |
| thal | 0-3 | Thalassemia type |
| **target** | **0,1** | **0=no disease, 1=disease present** |

---

## 🎓 How Models Work

### **Logistic Regression**
```
Process:
1. Linear combination: z = w₀ + w₁x₁ + w₂x₂ + ... + w₁₃x₁₃
2. Sigmoid function: p = 1 / (1 + e^(-z))
3. Classification: if p > 0.5 → disease (1), else → no disease (0)

Advantages:
- Fast training
- Interpretable coefficients
- Good baseline
```

### **SVM with RBF Kernel**
```
Process:
1. Maps data to higher-dimensional space
2. Finds optimal hyperplane separating classes
3. Maximizes margin between classes
4. Uses RBF kernel for non-linear boundaries

Advantages:
- Handles non-linear patterns
- Robust to outliers
- Good generalization
```

---

## 📈 Example Workflow

### **1. Preprocessing**
```
Input: heart(in).csv (1026 rows, 14 columns)
   ↓
Remove missing values (dropna)
   ↓
Remove duplicates
   ↓
Standardize numerical features (StandardScaler)
   ↓
Split: 80% train (820 rows), 20% test (206 rows)
   ↓
Output: X_train, X_test, y_train, y_test + scaler
```

### **2. Model Training**
```
Input: Preprocessed data
   ↓
Train Logistic Regression Model
Train SVM Model
   ↓
Evaluate on test set
   ↓
Compare metrics
   ↓
Output: Best model saved + metrics/visualizations
```

### **3. API Usage**
```
User → Streamlit UI
   ↓
Streamlit sends patient data to FastAPI
   ↓
FastAPI scales features using saved scaler
   ↓
Model makes prediction
   ↓
Returns: prediction + confidence + risk level
   ↓
Streamlit displays results
```

---

## 🔍 Testing the API

### **Using FastAPI Docs** (Built-in UI)
1. Go to: `http://localhost:8000/docs`
2. Find the `/predict` endpoint
3. Click "Try it out"
4. Enter sample data
5. Click "Execute"

### **Using curl**
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
  }'
```

### **Using Python**
```python
import requests

url = "http://localhost:8000/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

---

## 📊 Expected Output

### **Model Performance**
```
Logistic Regression:
- Accuracy:  0.85-0.90
- Precision: 0.82-0.88
- Recall:    0.80-0.88
- F1-Score:  0.81-0.87
- ROC-AUC:   0.88-0.93

SVM:
- Accuracy:  0.82-0.88
- Precision: 0.80-0.86
- Recall:    0.78-0.85
- F1-Score:  0.79-0.85
- ROC-AUC:   0.85-0.91
```

### **Prediction Output**
```json
{
  "prediction": 0,              // 0=No disease, 1=Disease present
  "confidence": 0.85,           // Probability (0-1)
  "model": "Logistic Regression",
  "risk_level": "moderate"      // low, moderate, or high
}
```

---

## 🐛 Troubleshooting

### **Issue: API Connection Failed**
```
Solution: Ensure FastAPI server is running
python 3_fastapi_app.py
```

### **Issue: Module Not Found**
```
Solution: Install requirements
pip install -r requirements.txt
```

### **Issue: Port Already in Use**
```
Solution 1: Kill existing process
lsof -ti:8000 | xargs kill -9

Solution 2: Use different port
python 3_fastapi_app.py --port 8001
```

### **Issue: Scaler/Model Not Found**
```
Solution: Run preprocessing and training first
python 1_preprocessing.py
python 2_train_model.py
```

---

## 📁 Project Structure

```
/workspaces/AIML_Task/
├── heart(in).csv                 # Original dataset
├── requirements.txt              # Python dependencies
├── setup.sh                      # Setup script
│
├── 1_preprocessing.py            # Data loading & cleaning
├── 2_train_model.py              # Model training
├── 3_fastapi_app.py              # FastAPI backend
├── 4_streamlit_app.py            # Streamlit frontend
│
├── data/                         # Preprocessed data
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── models/                       # Trained models
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   ├── best_model.pkl
│   └── best_model_name.txt
│
└── plots/                        # Visualizations
    ├── confusion_matrices.png
    └── roc_curves.png
```

---

## ✅ Checklist - Step by Step

- [ ] **Step 1**: Install dependencies (`pip install -r requirements.txt`)
- [ ] **Step 2**: Run preprocessing (`python 1_preprocessing.py`)
- [ ] **Step 3**: Train models (`python 2_train_model.py`)
- [ ] **Step 4**: Start FastAPI (`python 3_fastapi_app.py`)
- [ ] **Step 5**: Start Streamlit (`streamlit run 4_streamlit_app.py`)
- [ ] **Step 6**: Test single prediction in Streamlit
- [ ] **Step 7**: Test batch upload
- [ ] **Step 8**: Test API directly using FastAPI docs (`/docs`)
- [ ] **Step 9**: Review model performance metrics
- [ ] **Step 10**: Verify visualizations in `plots/` directory

---

## 🎉 Success Criteria

✓ Data successfully preprocessed and cleaned
✓ Models trained with good accuracy (>85%)
✓ FastAPI running with all endpoints working
✓ Streamlit UI displaying predictions correctly
✓ Predictions are consistent and reliable
✓ API documentation auto-generated
✓ Models and scalers properly saved
✓ Batch processing working correctly

---

## 📚 References & Best Practices

**Data Preprocessing**:
- Handle missing values early
- Standardize before model training
- Use stratified split for imbalanced data
- Keep train-test split separate

**Model Selection**:
- Logistic Regression: Fast, interpretable
- SVM: Better for non-linear patterns
- Always compare multiple models

**API Design**:
- Use meaningful status codes
- Validate input data
- Return consistent response format
- Document all endpoints

**Frontend Development**:
- Keep UI simple and intuitive
- Provide multiple input methods
- Show confidence scores
- Allow batch processing

---

## 🤝 Support

For issues or questions:
1. Check FastAPI logs (`http://localhost:8000/docs`)
2. Check Streamlit logs (terminal where you ran streamlit)
3. Verify all data files exist in `data/` and `models/`
4. Ensure all requirements are installed

---

**Happy Predicting! ❤️**
