"""
Heart Disease Classification - Data Preprocessing
Load, clean, and standardize the dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dataset
print("Loading dataset...")
df = pd.read_csv('heart(in).csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

print(f"\nMissing values before cleaning:")
print(df.isnull().sum())

# ==================== CLEANING ====================
# Remove rows with missing values
df_clean = df.dropna()
print(f"\nDataset shape after removing missing values: {df_clean.shape}")

print(f"\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# Check for duplicates
print(f"\nDuplicate rows: {df_clean.duplicated().sum()}")

# Remove duplicates if any
df_clean = df_clean.drop_duplicates()

# ==================== FEATURE ENGINEERING ====================
# Separate features and target
X = df_clean.drop('target', axis=1)
y = df_clean['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts())

# ==================== STANDARDIZATION ====================
print("\nStandardizing numerical columns...")

# Identify numerical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns: {numerical_cols}")

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
X_scaled = pd.DataFrame(
    scaler.fit_transform(X[numerical_cols]),
    columns=numerical_cols,
    index=X.index
)

print(f"\nScaled data statistics:")
print(X_scaled.describe())

# ==================== TRAIN-TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
print(f"\nTraining set target distribution:")
print(y_train.value_counts())
print(f"\nTesting set target distribution:")
print(y_test.value_counts())

# ==================== SAVE PREPROCESSED DATA ====================
# Create data directory
os.makedirs('data', exist_ok=True)

# Save train and test data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Save scaler for later use
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open('data/feature_names.pkl', 'wb') as f:
    pickle.dump(numerical_cols, f)

print("\n✓ Preprocessing complete!")
print("✓ Files saved in 'data/' directory:")
print("  - X_train.csv, X_test.csv")
print("  - y_train.csv, y_test.csv")
print("  - scaler.pkl")
print("  - feature_names.pkl")
