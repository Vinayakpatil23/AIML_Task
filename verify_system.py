#!/usr/bin/env python3
"""
Heart Disease Classification - Verification Script
Tests all components are working correctly
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("HEART DISEASE CLASSIFICATION - SYSTEM VERIFICATION")
print("=" * 70)

checks_passed = 0
checks_failed = 0

# ==================== CHECK 1: Data Files ====================
print("\n[CHECK 1] Verifying data files...")
data_files = [
    'data/X_train.csv',
    'data/X_test.csv',
    'data/y_train.csv',
    'data/y_test.csv',
    'data/scaler.pkl',
    'data/feature_names.pkl'
]

for file in data_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✓ {file} ({size:,} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ MISSING: {file}")
        checks_failed += 1

# ==================== CHECK 2: Model Files ====================
print("\n[CHECK 2] Verifying model files...")
model_files = [
    'models/best_model.pkl',
    'models/logistic_regression.pkl',
    'models/svm_model.pkl',
    'models/best_model_name.txt'
]

for file in model_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✓ {file} ({size:,} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ MISSING: {file}")
        checks_failed += 1

# ==================== CHECK 3: Visualization Files ====================
print("\n[CHECK 3] Verifying visualization files...")
plot_files = [
    'plots/confusion_matrices.png',
    'plots/roc_curves.png'
]

for file in plot_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✓ {file} ({size:,} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ MISSING: {file}")
        checks_failed += 1

# ==================== CHECK 4: Load and Test Scaler ====================
print("\n[CHECK 4] Testing scaler...")
try:
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✓ Scaler loaded successfully")
    print(f"    - Type: {type(scaler).__name__}")
    print(f"    - Mean shape: {scaler.mean_.shape}")
    print(f"    - Scale shape: {scaler.scale_.shape}")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Failed to load scaler: {e}")
    checks_failed += 1

# ==================== CHECK 5: Load and Test Model ====================
print("\n[CHECK 5] Testing model...")
try:
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"  ✓ Model loaded successfully")
    print(f"    - Type: {type(model).__name__}")
    
    # Check model has necessary methods
    if hasattr(model, 'predict'):
        print(f"    - Has predict() method: ✓")
        checks_passed += 1
    else:
        print(f"    - Has predict() method: ✗")
        checks_failed += 1
        
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    checks_failed += 1

# ==================== CHECK 6: Test Data Loading ====================
print("\n[CHECK 6] Testing data loading...")
try:
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_test = pd.read_csv('data/y_test.csv')
    
    print(f"  ✓ Data loaded successfully")
    print(f"    - X_train shape: {X_train.shape}")
    print(f"    - X_test shape: {X_test.shape}")
    print(f"    - y_train shape: {y_train.shape}")
    print(f"    - y_test shape: {y_test.shape}")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Failed to load data: {e}")
    checks_failed += 1

# ==================== CHECK 7: Feature Names ====================
print("\n[CHECK 7] Verifying feature names...")
try:
    with open('data/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"  ✓ Feature names loaded successfully")
    print(f"    - Number of features: {len(feature_names)}")
    print(f"    - Features: {', '.join(feature_names)}")
    
    if len(feature_names) == 13:
        print(f"    - Feature count correct: ✓")
        checks_passed += 1
    else:
        print(f"    - Feature count INCORRECT (expected 13, got {len(feature_names)})")
        checks_failed += 1
except Exception as e:
    print(f"  ✗ Failed to load feature names: {e}")
    checks_failed += 1

# ==================== CHECK 8: Model Name ====================
print("\n[CHECK 8] Verifying best model name...")
try:
    with open('models/best_model_name.txt', 'r') as f:
        model_name = f.read().strip()
    print(f"  ✓ Model name: {model_name}")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Failed to load model name: {e}")
    checks_failed += 1

# ==================== CHECK 9: Test Prediction ====================
print("\n[CHECK 9] Testing model prediction...")
try:
    # Test data point
    test_sample = X_test.iloc[0:1]
    
    # Make prediction
    prediction = model.predict(test_sample)[0]
    
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(test_sample)[0][prediction]
    else:
        confidence = abs(model.decision_function(test_sample)[0])
    
    print(f"  ✓ Prediction successful")
    print(f"    - Sample prediction: {prediction} (0=No disease, 1=Disease)")
    print(f"    - Confidence: {confidence:.4f}")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Prediction failed: {e}")
    checks_failed += 1

# ==================== CHECK 10: Python Scripts ====================
print("\n[CHECK 10] Verifying Python scripts...")
scripts = [
    '1_preprocessing.py',
    '2_train_model.py',
    '3_fastapi_app.py',
    '4_streamlit_app.py'
]

for script in scripts:
    if os.path.exists(script):
        size = os.path.getsize(script)
        print(f"  ✓ {script} ({size:,} bytes)")
        checks_passed += 1
    else:
        print(f"  ✗ MISSING: {script}")
        checks_failed += 1

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

total_checks = checks_passed + checks_failed
print(f"\nTotal Checks: {total_checks}")
print(f"✓ Passed: {checks_passed}")
print(f"✗ Failed: {checks_failed}")

if checks_failed == 0:
    print("\n🎉 ALL CHECKS PASSED! System is ready to use!")
    print("\nNext steps:")
    print("  1. python 3_fastapi_app.py          (Terminal 1)")
    print("  2. streamlit run 4_streamlit_app.py (Terminal 2)")
    print("\nThen visit: http://localhost:8501")
    sys.exit(0)
else:
    print(f"\n⚠️  {checks_failed} check(s) failed. Please fix issues above.")
    print("\nTroubleshooting:")
    print("  - Ensure you ran: python 1_preprocessing.py")
    print("  - Ensure you ran: python 2_train_model.py")
    print("  - Check that all files are in correct directories")
    sys.exit(1)
