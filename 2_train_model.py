"""
Heart Disease Classification - Model Training
Train both SVM and Logistic Regression models
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("HEART DISEASE CLASSIFICATION - MODEL TRAINING")
print("=" * 60)

# Load preprocessed data
print("\nLoading preprocessed data...")
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ==================== LOGISTIC REGRESSION ====================
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\nLogistic Regression Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Disease']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# ==================== SUPPORT VECTOR MACHINE ====================
print("\n" + "=" * 60)
print("TRAINING SVM MODEL")
print("=" * 60)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\nSVM Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_svm):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_svm):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['No Disease', 'Disease']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# ==================== MODEL COMPARISON ====================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

lr_acc = accuracy_score(y_test, y_pred_lr)
svm_acc = accuracy_score(y_test, y_pred_svm)

print(f"\nLogistic Regression Accuracy: {lr_acc:.4f}")
print(f"SVM Accuracy:                 {svm_acc:.4f}")

# Choose best model
best_model = lr_model if lr_acc >= svm_acc else svm_model
best_model_name = "Logistic Regression" if lr_acc >= svm_acc else "SVM"

print(f"\n✓ Best Model: {best_model_name} (Accuracy: {max(lr_acc, svm_acc):.4f})")

# ==================== SAVE MODELS ====================
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

os.makedirs('models', exist_ok=True)

# Save both models
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('models/best_model_name.txt', 'w') as f:
    f.write(best_model_name)

print("\n✓ Models saved successfully!")
print("  - models/logistic_regression.pkl")
print("  - models/svm_model.pkl")
print("  - models/best_model.pkl (Best performing model)")
print("  - models/best_model_name.txt")

# ==================== VISUALIZATION ====================
print("\nGenerating visualizations...")
os.makedirs('plots', exist_ok=True)

# Confusion Matrix for SVM
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression - Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('SVM - Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=100, bbox_inches='tight')
print("✓ Saved: plots/confusion_matrices.png")

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_proba_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC: {roc_auc_score(y_test, y_pred_proba_lr):.3f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC: {roc_auc_score(y_test, y_pred_proba_svm):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('plots/roc_curves.png', dpi=100, bbox_inches='tight')
print("✓ Saved: plots/roc_curves.png")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
