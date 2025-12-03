"""
Heart Disease Classification - Feature Importance Analysis
Identify which features are most important for predictions
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import os

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# ==================== LOAD DATA & MODEL ====================
print("\nLoading data and model...")

# Load data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load best model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('data/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"✓ Data loaded: {X_test.shape}")
print(f"✓ Model loaded")
print(f"✓ Features: {feature_names}")

# ==================== METHOD 1: MODEL COEFFICIENTS ====================
print("\n" + "=" * 70)
print("METHOD 1: MODEL COEFFICIENTS (Weight-based Importance)")
print("=" * 70)

# Get coefficients from Logistic Regression
if hasattr(model, 'coef_'):
    coefficients = model.coef_[0]
    
    # Create DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Coefficients (sorted by importance):")
    print(coef_df.to_string(index=False))
    
    print("\n💡 Interpretation:")
    print("  - Larger absolute values = more important")
    print("  - Positive coefficient = increases disease risk")
    print("  - Negative coefficient = decreases disease risk")
    
    # Visualize coefficients
    plt.figure(figsize=(10, 6))
    colors = ['red' if x > 0 else 'blue' for x in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance - Logistic Regression Coefficients')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/feature_coefficients.png', dpi=100, bbox_inches='tight')
    print("\n✓ Saved: plots/feature_coefficients.png")

# ==================== METHOD 2: PERMUTATION IMPORTANCE ====================
print("\n" + "=" * 70)
print("METHOD 2: PERMUTATION IMPORTANCE (Model-agnostic)")
print("=" * 70)

print("\nCalculating permutation importance...")
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\nPermutation Importance (sorted by importance):")
print(perm_df.to_string(index=False))

print("\n💡 Interpretation:")
print("  - Measures how much model performance drops when feature is shuffled")
print("  - Higher value = feature is more important")
print("  - Works with any model type")
print("  - Std Dev shows variability across shuffles")

# Visualize permutation importance
plt.figure(figsize=(10, 6))
plt.barh(perm_df['Feature'], perm_df['Importance'], 
         xerr=perm_df['Std'], color='steelblue', capsize=3)
plt.xlabel('Importance Score')
plt.title('Feature Importance - Permutation Method')
plt.tight_layout()
plt.savefig('plots/feature_permutation_importance.png', dpi=100, bbox_inches='tight')
print("✓ Saved: plots/feature_permutation_importance.png")

# ==================== METHOD 3: CORRELATION WITH TARGET ====================
print("\n" + "=" * 70)
print("METHOD 3: CORRELATION WITH TARGET VARIABLE")
print("=" * 70)

# Load training data with target
y_train = pd.read_csv('data/y_train.csv')
X_train_with_target = X_train.copy()
X_train_with_target['target'] = y_train.values

# Calculate correlation
correlations = X_train_with_target.corr()['target'].drop('target').sort_values(
    ascending=False, key=abs
)

corr_df = pd.DataFrame({
    'Feature': correlations.index,
    'Correlation': correlations.values,
    'Abs_Correlation': np.abs(correlations.values)
}).sort_values('Abs_Correlation', ascending=False)

print("\nFeature Correlation with Target:")
print(corr_df.to_string(index=False))

print("\n💡 Interpretation:")
print("  - Measures linear relationship with disease presence")
print("  - Range: -1 to +1")
print("  - Positive = feature increases with disease")
print("  - Negative = feature decreases with disease")
print("  - Larger absolute value = stronger relationship")

# Visualize correlations
plt.figure(figsize=(10, 6))
colors = ['red' if x > 0 else 'blue' for x in corr_df['Correlation']]
plt.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
plt.xlabel('Correlation with Target')
plt.title('Feature Correlation with Heart Disease (Target)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots/feature_correlation.png', dpi=100, bbox_inches='tight')
print("✓ Saved: plots/feature_correlation.png")

# ==================== METHOD 4: COMBINED RANKING ====================
print("\n" + "=" * 70)
print("METHOD 4: COMBINED IMPORTANCE RANKING")
print("=" * 70)

# Normalize scores to 0-1 range
coef_scores = (coef_df.set_index('Feature')['Abs_Coefficient'] / 
               coef_df['Abs_Coefficient'].max())
perm_scores = (perm_df.set_index('Feature')['Importance'] / 
               perm_df['Importance'].max())
corr_scores = (corr_df.set_index('Feature')['Abs_Correlation'] / 
               corr_df['Abs_Correlation'].max())

# Combine all methods (average)
combined_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient_Score': [coef_scores.get(f, 0) for f in feature_names],
    'Permutation_Score': [perm_scores.get(f, 0) for f in feature_names],
    'Correlation_Score': [corr_scores.get(f, 0) for f in feature_names]
})

combined_importance['Combined_Score'] = combined_importance[[
    'Coefficient_Score', 'Permutation_Score', 'Correlation_Score'
]].mean(axis=1)

combined_importance = combined_importance.sort_values('Combined_Score', ascending=False)

print("\nCombined Importance Ranking:")
print(combined_importance.to_string(index=False))

# Visualize combined ranking
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Combined score
axes[0, 0].barh(combined_importance['Feature'], 
                combined_importance['Combined_Score'], color='purple')
axes[0, 0].set_xlabel('Combined Score')
axes[0, 0].set_title('Combined Importance Score')

# Coefficient score
axes[0, 1].barh(combined_importance['Feature'], 
                combined_importance['Coefficient_Score'], color='red')
axes[0, 1].set_xlabel('Score')
axes[0, 1].set_title('Coefficient-based Importance')

# Permutation score
axes[1, 0].barh(combined_importance['Feature'], 
                combined_importance['Permutation_Score'], color='steelblue')
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_title('Permutation Importance')

# Correlation score
axes[1, 1].barh(combined_importance['Feature'], 
                combined_importance['Correlation_Score'], color='green')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_title('Correlation-based Importance')

plt.tight_layout()
plt.savefig('plots/feature_importance_combined.png', dpi=100, bbox_inches='tight')
print("✓ Saved: plots/feature_importance_combined.png")

# ==================== TOP FEATURES ANALYSIS ====================
print("\n" + "=" * 70)
print("TOP 5 MOST IMPORTANT FEATURES")
print("=" * 70)

top_5 = combined_importance.head(5)
print("\nRanking:")
for idx, (_, row) in enumerate(top_5.iterrows(), 1):
    print(f"{idx}. {row['Feature']:12} → Combined Score: {row['Combined_Score']:.3f}")

print("\n" + "=" * 70)
print("SUMMARY OF IMPORTANCE METHODS")
print("=" * 70)

summary = f"""
📊 THREE WAYS TO MEASURE FEATURE IMPORTANCE:

1️⃣  COEFFICIENT-BASED (Weight Analysis)
   ├─ How: Look at model weights/coefficients
   ├─ When: Works with linear models (Logistic Regression)
   ├─ Pros: Fast, interpretable, shows direction (+/-)
   └─ Cons: Only works for linear models

2️⃣  PERMUTATION IMPORTANCE (Shuffling Test)
   ├─ How: Shuffle each feature and measure performance drop
   ├─ When: Works with ANY model type
   ├─ Pros: Model-agnostic, robust, realistic
   └─ Cons: Slower to calculate

3️⃣  CORRELATION ANALYSIS (Relationship Strength)
   ├─ How: Measure correlation with target variable
   ├─ When: Quick exploratory analysis
   ├─ Pros: Fast, easy to interpret, statistical
   └─ Cons: Only captures linear relationships

📈 RECOMMENDED APPROACH:
   → Use COMBINED ranking (all 3 methods averaged)
   → Focuses on most consensus-backed important features
   → Most reliable for decision making

🎯 HOW TO USE THIS INFORMATION:
   ✓ Focus data collection on top features
   ✓ Engineer new features from important ones
   ✓ Prioritize monitoring important features in production
   ✓ Investigate why top features are important
   ✓ Use feature selection to simplify model
"""

print(summary)

# ==================== SAVE RESULTS ====================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save to CSV
combined_importance.to_csv('plots/feature_importance_ranking.csv', index=False)
print("✓ Saved: plots/feature_importance_ranking.csv")

print("\n✅ Feature Importance Analysis Complete!")
print("\n📁 Generated Files:")
print("   - plots/feature_coefficients.png")
print("   - plots/feature_permutation_importance.png")
print("   - plots/feature_correlation.png")
print("   - plots/feature_importance_combined.png")
print("   - plots/feature_importance_ranking.csv")

print("\n" + "=" * 70)
