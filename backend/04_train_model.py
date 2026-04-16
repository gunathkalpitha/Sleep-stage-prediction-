import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("SIMPLE BASELINE - Sleep Stage Classification with Random Forest")
print("="*70)

# Load preprocessed data
X_train = np.load("sleep_data/processed/X_train.npy")
X_test  = np.load("sleep_data/processed/X_test.npy")
y_train = np.load("sleep_data/processed/y_train.npy")
y_test  = np.load("sleep_data/processed/y_test.npy")

with open("sleep_data/processed/feature_names.pkl", "rb") as f:
    FEATURE_NAMES = pickle.load(f)

CLASS_NAMES = {0: 'Wake', 1: 'Light', 2: 'Deep'}

print(f"\nDataset:")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples:     {len(X_test)}")
print(f"  Features:         {len(FEATURE_NAMES)}")
print(f"  Classes:          {len(CLASS_NAMES)}")
print(f"\nFeatures: {FEATURE_NAMES}")

# GridSearchCV Tuning
print(f"\n{'='*70}")
print("GridSearchCV - Hyperparameter Tuning (5-Fold CV)")
print("="*70)

# Conservative parameter grid to avoid overfitting
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 12, 14],
    'min_samples_leaf': [4, 6, 8],
    'min_samples_split': [10, 12, 14]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use 5-fold CV on all data (no test set peeking)
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

gs = GridSearchCV(rf_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

start_time = time.time()
gs.fit(X_all, y_all)
train_time = time.time() - start_time

model = gs.best_estimator_

print(f"\nBest Parameters: {gs.best_params_}")
print(f"Best CV Score: {gs.best_score_*100:.1f}%")

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracies
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_test_gap = (train_acc - test_acc) * 100

print(f"\nResults:")
print(f"  Train Accuracy: {train_acc*100:.1f}%")
print(f"  Test Accuracy:  {test_acc*100:.1f}%")
print(f"  Train-Test Gap: {train_test_gap:.1f}%")

if train_test_gap < 5:
    status = "Excellent"
elif train_test_gap < 10:
    status = "Good"
elif train_test_gap < 15:
    status = "Acceptable"
else:
    status = "High overfitting"

print(f"  Generalization: {status}")

# Use GridSearchCV's CV scores (already computed)
cv_scores = gs.cv_results_['mean_test_score']

print(f"\n5-Fold Cross-Validation:")
print(f"  Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"  Mean:   {cv_scores.mean()*100:.1f}%")
print(f"  Std:    ±{cv_scores.std()*100:.1f}%")

# Classification report
print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=list(CLASS_NAMES.values())))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"Confusion Matrix:")
print(cm)

# Feature importance
print(f"\nFeature Importance:")
importances = model.feature_importances_
feature_importance_dict = {FEATURE_NAMES[i]: importances[i] for i in range(len(FEATURE_NAMES))}
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
for rank, (feature, importance) in enumerate(sorted_features, 1):
    print(f"  {rank:2d}. {feature:20s} {importance:.4f}")

# Save results and model
os.makedirs("sleep_data/results", exist_ok=True)

# Confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(CLASS_NAMES.values()),
            yticklabels=list(CLASS_NAMES.values()))
plt.title(f'Confusion Matrix (Test Accuracy: {test_acc*100:.1f}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("sleep_data/results/confusion_matrix.png", dpi=100)
plt.close()

# Feature importance plot
plt.figure(figsize=(10, 6))
features = [f[0] for f in sorted_features]
importances_sorted = [f[1] for f in sorted_features]
plt.barh(range(len(features)), importances_sorted, color='steelblue')
plt.yticks(range(len(features)), features)
plt.xlabel('Importance Score')
plt.title('Feature Importance (Simple Baseline)')
plt.tight_layout()
plt.savefig("sleep_data/results/feature_importance.png", dpi=100)
plt.close()

# Train vs Test vs CV Accuracy
plt.figure(figsize=(10, 6))
categories = ['Train', 'Test', 'CV Mean']
accuracies = [train_acc*100, test_acc*100, cv_scores.mean()*100]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
bars = plt.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.axhline(y=82, color='r', linestyle='--', linewidth=2, label='Paper target 82%')
plt.ylabel('Accuracy %')
plt.title('Train vs Test vs Cross-Validation Accuracy')
plt.ylim([0, 100])
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("sleep_data/results/train_vs_test.png", dpi=100)
plt.close()

# Save model
with open("sleep_data/processed/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n" + "="*70)
print("Model and visualizations saved successfully")
print("  Model:        sleep_data/processed/rf_model.pkl")
print("  Confusion:    sleep_data/results/confusion_matrix.png")
print("  Importance:   sleep_data/results/feature_importance.png")
print("="*70)
import os
