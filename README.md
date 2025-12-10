# 🎓 Heart Disease Classification - Complete Task Explanation

## 📚 Table of Contents

1. [Task Overview](#task-overview)
2. [Step-by-Step Process](#step-by-step-process)
3. [Machine Learning Concepts](#machine-learning-concepts)
4. [Implementation Details](#implementation-details)
5. [Performance Results](#performance-results)
6. [System Architecture](#system-architecture)
7. [Running the System](#running-the-system)

---

## Task Overview

### **Objective**
Build a complete machine learning system to predict heart disease risk based on patient health indicators.

### **Key Requirements**
1. ✅ Load dataset
2. ✅ Clean missing values
3. ✅ Standardize numerical columns
4. ✅ Train classification model (SVM / Logistic Regression)
5. ✅ Save model
6. ✅ Build FastAPI endpoint /predict
7. ✅ Test via Streamlit

### **Dataset Information**
- **Source**: Kaggle Heart Disease Dataset
- **Original Records**: 1,026 rows
- **After Cleaning**: 302 valid records (removed nulls & duplicates)
- **Features**: 13 numerical attributes
- **Target**: Binary classification (0 = No disease, 1 = Disease present)

---

## Step-by-Step Process

### **STEP 1: Data Loading & Exploration**

**File**: `1_preprocessing.py`

**What happens**:
```python
df = pd.read_csv('heart(in).csv')  # Load raw data
print(df.shape)  # (1026, 14)
print(df.info()) # Check data types and nulls
print(df.head()) # Display sample records
```

**Output**:
- Dataset shape: 1,026 rows × 14 columns
- 13 features + 1 target variable
- Identifies missing values in dataset

---

### **STEP 2: Data Cleaning**

**What happens**:

#### **2.1 Remove Missing Values**
```python
df_clean = df.dropna()
# Removes rows with ANY missing values
# Before: 1,026 rows → After: ~302 rows
```

**Why needed**:
- Missing data creates bias in model
- ML algorithms can't handle NULL values
- Better to remove small amount than impute incorrectly

#### **2.2 Remove Duplicates**
```python
df_clean = df_clean.drop_duplicates()
# Removes exact duplicate rows
# Prevents data leakage and overfitting
```

**Why needed**:
- Duplicate records inflate model performance
- Can appear in both train and test sets
- Creates illusion of better accuracy

**Result**:
- Final clean dataset: 302 valid records
- No missing values ✓
- No duplicates ✓

---

### **STEP 3: Feature Standardization**

**What happens**:

```python
from sklearn.preprocessing import StandardScaler

# Before standardization, features have different scales:
# age: 29-77
# trestbps: 90-200
# chol: 126-564
# This causes problems for distance-based models!

# Standardization formula:
# X_scaled = (X - mean) / std_dev
# Result: Mean = 0, Std Dev = 1 (for each feature)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why needed for Logistic Regression**:
- Coefficients become comparable
- Faster convergence during training
- Prevents numerical instability

**Why needed for SVM**:
- Works with distance measurements
- Unscaled data gives unfair weight to large-scale features
- RBF kernel requires normalized data

**Visual Example**:
```
Before Standardization:
age    trestbps    chol
50     120         240
60     150         300

After Standardization:
age      trestbps    chol
0.15     -0.20       0.12
1.25      0.95       1.83
(centered at 0, scaled to unit variance)
```

---

### **STEP 4: Train-Test Split**

**What happens**:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=y          # Maintain class balance
)
```

**Results**:
- **Training set**: 241 samples (80%)
  - Class 0 (No disease): 110
  - Class 1 (Disease): 131
  
- **Test set**: 61 samples (20%)
  - Class 0 (No disease): 28
  - Class 1 (Disease): 33

**Why this split**:
- Train on larger set → better learning
- Test on unseen data → measure generalization
- Stratification → both sets have similar class distribution

---

### **STEP 5: Model Training**

#### **5.1 Logistic Regression**

**How it works**:

```
Step 1: Linear Combination
z = w₀ + w₁·x₁ + w₂·x₂ + ... + w₁₃·x₁₃

Step 2: Sigmoid Function (converts to probability)
P(y=1) = 1 / (1 + e^(-z))

Step 3: Classification
if P(y=1) > 0.5:
    predict: 1 (disease present)
else:
    predict: 0 (no disease)
```

**Process**:
```python
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)  # Learn weights
y_pred = lr_model.predict(X_test)  # Make predictions
```

**Characteristics**:
- ✓ Fast to train
- ✓ Interpretable (can see feature importance)
- ✓ Works well for binary classification
- ✓ Good baseline model

---

#### **5.2 Support Vector Machine (SVM)**

**How it works**:

```
1. Map data to higher-dimensional space
2. Find optimal hyperplane that separates classes
3. Maximize margin between classes
4. Use RBF kernel for non-linear boundaries

Hyperplane equation: w·x + b = 0
Decision function: sign(w·φ(x) + b)
```

**Process**:
```python
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
```

**Characteristics**:
- ✓ Handles non-linear patterns
- ✓ Robust to outliers
- ✓ Good generalization
- ✗ Slower training than Logistic Regression

**RBF Kernel**:
- Non-linear transformation of features
- Good for complex decision boundaries
- Parameter: gamma controls boundary smoothness

---

### **STEP 6: Model Evaluation**

**Metrics Used**:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many correct |
| **Recall** | TP / (TP + FN) | Of actual positives, how many found |
| **F1-Score** | 2·Precision·Recall / (Precision + Recall) | Harmonic mean |
| **ROC-AUC** | Area under ROC curve | How well model separates classes |

**Legend**:
- **TP** (True Positive): Correctly predicted disease
- **TN** (True Negative): Correctly predicted no disease
- **FP** (False Positive): Incorrectly predicted disease
- **FN** (False Negative): Incorrectly predicted no disease

**Results**:

```
LOGISTIC REGRESSION ⭐ (Selected as best model)
┌─────────────┬──────────┐
│ Accuracy    │ 80.33%   │
│ Precision   │ 81.11%   │
│ Recall      │ 79.39%   │
│ F1-Score    │ 80.24%   │
│ ROC-AUC     │ 88.87%   │
└─────────────┴──────────┘

SVM
┌─────────────┬──────────┐
│ Accuracy    │ 77.05%   │
│ Precision   │ 77.14%   │
│ Recall      │ 81.82%   │
│ F1-Score    │ 79.41%   │
│ ROC-AUC     │ 84.52%   │
└─────────────┴──────────┘
```

**Why Logistic Regression is better**:
- 3.28% higher accuracy
- Better precision (fewer false alarms)
- Better balanced metrics
- Faster inference time

---

### **STEP 7: Model Persistence**

**What happens**:

```python
import pickle

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save scaler (IMPORTANT!)
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open('data/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
```

**Why save scaler**:
- New predictions must use SAME scaler
- Prevents data drift
- Ensures consistency

**Files created**:
```
models/
├── best_model.pkl (1.0 KB) - Main model
├── logistic_regression.pkl (1.0 KB) - Backup
├── svm_model.pkl (18 KB) - Alternative
└── best_model_name.txt

data/
├── scaler.pkl - StandardScaler
└── feature_names.pkl - Feature order
```

---

### **STEP 8: API Development (FastAPI)**

**Endpoints**:

#### **8.1 Health Check**
```
GET /
Response: {"status": "online", "model": "Logistic Regression", ...}

GET /health
Response: {"status": "healthy", "model_loaded": true, ...}
```

#### **8.2 Single Prediction**
```
POST /predict

Input:
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
  "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
  "exang": 0, "oldpeak": 1, "slope": 2, "ca": 2, "thal": 3
}

Process:
1. Validate input (Pydantic model)
2. Create DataFrame with feature names
3. Scale using saved scaler
4. Predict using saved model
5. Calculate confidence
6. Determine risk level

Output:
{
  "prediction": 0,                    // 0=No, 1=Yes
  "confidence": 0.85,                 // 0-1
  "model": "Logistic Regression",
  "risk_level": "moderate"            // low/moderate/high
}
```

#### **8.3 Batch Prediction**
```
POST /predict-batch
Input: Array of multiple patients
Output: Array of predictions
```

#### **8.4 Model Information**
```
GET /model-info
Response: Model details, features, classes
```

**Why FastAPI**:
- ✓ Modern async support
- ✓ Auto-generates API docs (/docs)
- ✓ Built-in data validation
- ✓ Easy to deploy
- ✓ Fast performance

---

### **STEP 9: Frontend Development (Streamlit)**

**Features**:

#### **Mode 1: Single Prediction**
- Interactive form for patient data
- Sliders for age, BP, cholesterol
- Dropdowns for categorical features
- Real-time prediction display
- Risk level visualization

#### **Mode 2: Batch Upload**
- Upload CSV files
- Process multiple patients
- Download results as CSV
- Bulk prediction capability

#### **Mode 3: Sample Test**
- Pre-configured test cases
- Quick demonstration
- Learn expected outputs

**Why Streamlit**:
- ✓ No HTML/CSS/JavaScript needed
- ✓ Built for data science
- ✓ Beautiful default UI
- ✓ Real-time updates
- ✓ Easy to maintain

---

## Machine Learning Concepts Explained

### **Classification vs Regression**
- **Classification**: Predict categories (Disease Yes/No)
- **Regression**: Predict continuous values (Disease risk %)

### **Binary Classification**
- Output: Only 2 possible classes
- Our case: 0 (No disease) or 1 (Disease present)

### **Supervised Learning**
- We have labeled data (known outcomes)
- Model learns from labeled examples
- Then predicts on new, unlabeled data

### **Train vs Test Sets**
- **Train**: Model learns patterns from this data
- **Test**: Evaluate performance on unseen data
- **Never** mix them!

### **Overfitting vs Underfitting**
```
Underfitting: Model too simple, misses patterns
│     Actual data
│    ╱╱╱╱╱╱╱
│──────────── (Model line)

Appropriate Fit: Model learns real patterns
│     Actual data
│    ╱╱╱╱╱╱╱
│   ╱╱╱╱╱╱ (Model line)

Overfitting: Model too complex, memorizes noise
│     Actual data        
│    ╱╱╱╱╱╱╱
│   ╱✓╱✓╱✓╱✓ (Model line - too wiggly)
```

### **Standardization Importance**
```
Without Standardization:
LR Weight: age=0.001, chol=0.0001 (appears chol is unimportant)
Actually: chol range is 438, age range is 48
Unfair comparison!

With Standardization:
Both features on same scale (mean=0, std=1)
Weights now comparable
Fair importance ranking
```

---

## Performance Results

### **Model Comparison**

```
                Logistic Regression    SVM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy        80.33% ⭐              77.05%
Precision       81.11%                77.14%
Recall          79.39%                81.82%
F1-Score        80.24%                79.41%
ROC-AUC         88.87%                84.52%
Training Time   Fast                  Medium
Inference       Fast                  Medium
Interpretable   Good                  Poor
```

### **Confusion Matrix** (Logistic Regression)

```
                Predicted
                No      Yes
Actual  No      20      8    (80% correctly identified no disease)
        Yes     6       27   (82% correctly identified disease)
```

**Interpretation**:
- **True Negatives (20)**: Correctly said "no disease" when no disease
- **False Positives (8)**: Wrongly said "disease" when no disease
- **False Negatives (6)**: Wrongly said "no disease" when disease present
- **True Positives (27)**: Correctly said "disease" when disease

### **Visualizations Created**

1. **Confusion Matrices** (`plots/confusion_matrices.png`)
   - Side-by-side comparison of both models
   - Shows correct/incorrect predictions

2. **ROC Curves** (`plots/roc_curves.png`)
   - Plots True Positive Rate vs False Positive Rate
   - Steeper curve = better model
   - Area under curve (AUC) = overall performance

---

## System Architecture

### **Data Flow Diagram**

```
┌─────────────────┐
│  heart(in).csv  │
│   (1026 rows)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  1_preprocessing.py         │
│  ✓ Load data                │
│  ✓ Remove missing values    │
│  ✓ Remove duplicates        │
│  ✓ Standardize features     │
│  ✓ Train-test split (80-20) │
└────────┬────────────────────┘
         │
         ▼
    ┌─────────┐
    │  data/  │
    ├─────────┤
    │X_train  │ (241 samples)
    │X_test   │ (61 samples)
    │scaler   │ (StandardScaler)
    └────┬────┘
         │
         ▼
┌────────────────────┐
│  2_train_model.py  │
│  ✓ Train models    │
│  ✓ Evaluate        │
│  ✓ Compare         │
│  ✓ Save best model │
└────────┬───────────┘
         │
         ▼
    ┌──────────┐
    │ models/  │
    ├──────────┤
    │best_model│ ⭐
    │LR or SVM │
    └────┬─────┘
         │
         ▼  ┌──────────────────────┐
         ├─▶│  3_fastapi_app.py    │
         │  │  FastAPI Server      │
         │  │  :8000               │
         │  └──────────┬───────────┘
         │             │
         │             │  POST /predict
         │             │  Input: Patient data
         │             │  Output: Prediction
         │             │
         └──────────────┘
              │
              ▼  ┌──────────────────────┐
              ├─▶│  4_streamlit_app.py  │
                 │  Streamlit UI        │
                 │  :8501               │
                 └──────────────────────┘
                           │
                  ┌────────┼────────┐
                  │        │        │
                  ▼        ▼        ▼
            ┌─────────┐ ┌──────┐ ┌───────┐
            │Single   │ │Batch │ │Sample │
            │Predict  │ │Upload│ │Test   │
            └─────────┘ └──────┘ └───────┘
```

### **File Structure**

```
/workspaces/AIML_Task/
│
├── 📄 Data Files
│   ├── heart(in).csv                 (Original dataset)
│
├── 🐍 Python Scripts
│   ├── 1_preprocessing.py            (Data cleaning)
│   ├── 2_train_model.py              (Model training)
│   ├── 3_fastapi_app.py              (API backend)
│   ├── 4_streamlit_app.py            (Web frontend)
│   ├── verify_system.py              (Verification)
│
├── 📚 Documentation
│   ├── GUIDE.md                      (Detailed guide)
│   ├── QUICKSTART.md                 (Quick start)
│   ├── README.md                     (This file)
│
├── 📦 Configuration
│   ├── requirements.txt              (Python packages)
│   ├── setup.sh                      (Setup script)
│
├── 📊 data/
│   ├── X_train.csv                  (Training features)
│   ├── X_test.csv                   (Testing features)
│   ├── y_train.csv                  (Training labels)
│   ├── y_test.csv                   (Testing labels)
│   ├── scaler.pkl                   (Standardizer)
│   └── feature_names.pkl            (Feature list)
│
├── 🤖 models/
│   ├── best_model.pkl               (⭐ Main model)
│   ├── logistic_regression.pkl      (Backup model)
│   ├── svm_model.pkl                (Alternative model)
│   └── best_model_name.txt          (Model type)
│
└── 📈 plots/
    ├── confusion_matrices.png       (Model performance)
    └── roc_curves.png               (ROC analysis)
```

---

## Running the System

### **Full Setup**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python 1_preprocessing.py

# 3. Train models
python 2_train_model.py

# 4. Run in Terminal 1
python 3_fastapi_app.py

# 5. Run in Terminal 2
streamlit run 4_streamlit_app.py

# 6. Visit
# - Streamlit: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

### **Verification**

```bash
python verify_system.py
```

---

## Key Takeaways

1. **Data is critical**: Good data quality → Better model
2. **Standardization matters**: Different scales affect ML algorithms
3. **Train-test split**: Prevent overfitting, ensure real performance
4. **Model comparison**: Compare multiple algorithms
5. **Evaluation metrics**: Don't just look at accuracy
6. **API & UI**: Connect models to real-world applications
7. **Documentation**: Helps others understand your work

