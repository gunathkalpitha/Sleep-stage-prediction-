import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = {0: 'Wake', 1: 'Light', 2: 'Deep'}

X_train = np.load("sleep_data/processed/X_train.npy")
X_test  = np.load("sleep_data/processed/X_test.npy")
y_train = np.load("sleep_data/processed/y_train.npy")
y_test  = np.load("sleep_data/processed/y_test.npy")

with open("sleep_data/processed/feature_names.pkl", "rb") as f:
    FEATURE_NAMES = pickle.load(f)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ── Regularised model ─────────────────────────────────────
# max_depth        limits how deep each tree grows
# min_samples_leaf forces each leaf to have enough samples
# max_features     limits features per split
# These prevent the model from memorising training data

model = RandomForestClassifier(
    n_estimators      = 100,
    max_depth         = 15,
    min_samples_leaf  = 5,
    min_samples_split = 10,
    max_features      = 'sqrt',
    random_state      = 42,
    n_jobs            = -1
)

print("\nTraining regularised Random Forest...")
start = time.time()
model.fit(X_train, y_train)
print(f"Done in {time.time() - start:.1f}s")

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
y_pred    = model.predict(X_test)
gap       = (train_acc - test_acc) * 100

print(f"\nTrain accuracy : {train_acc * 100:.1f}%")
print(f"Test accuracy  : {test_acc  * 100:.1f}%")
print(f"Gap            : {gap:.1f}%")

if gap < 3:
    print("Status         : ✅ Excellent generalisation")
elif gap < 7:
    print("Status         : ✅ Acceptable generalisation")
else:
    print("Status         : ⚠️  Some overfitting remains")

print(f"\nTarget         : 82.0%")
print(f"Result         : {'✅ Exceeded' if test_acc >= 0.82 else '⚠️ Below'} target\n")

print(classification_report(
    y_test, y_pred,
    target_names=list(CLASS_NAMES.values())
))

# ── 5-fold cross validation ───────────────────────────────
# Tests the model on 5 different splits of data
# Gives a more honest accuracy estimate

print("Running 5-fold cross validation...")
import numpy as np
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

cv_scores = cross_val_score(
    model, X_all, y_all,
    cv=5, scoring='accuracy', n_jobs=-1
)
print(f"CV scores      : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"CV mean        : {cv_scores.mean()*100:.1f}%")
print(f"CV std         : ±{cv_scores.std()*100:.1f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

importances = model.feature_importances_
indices     = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for rank, idx in enumerate(indices):
    print(f"  #{rank+1:<2} {FEATURE_NAMES[idx]:<20} {importances[idx]:.4f}")

os.makedirs("sleep_data/results", exist_ok=True)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=list(CLASS_NAMES.values()),
    yticklabels=list(CLASS_NAMES.values())
)
plt.title(f'Confusion Matrix  (Test: {test_acc*100:.1f}%  Train: {train_acc*100:.1f}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("sleep_data/results/confusion_matrix.png", dpi=100)
plt.close()

plt.figure(figsize=(10, 6))
plt.barh(
    [FEATURE_NAMES[i] for i in indices[::-1]],
    importances[indices[::-1]],
    color='steelblue'
)
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig("sleep_data/results/feature_importance.png", dpi=100)
plt.close()

# ── Train vs Test accuracy comparison plot ────────────────
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ['Train Accuracy', 'Test Accuracy', 'CV Mean'],
    [train_acc * 100, test_acc * 100, cv_scores.mean() * 100],
    color=['#3498db', '#2ecc71', '#e67e22'],
    width=0.5
)
ax.set_ylim(0, 110)
ax.axhline(y=82, color='red', linestyle='--',
           alpha=0.7, label='Paper target 82%')
ax.set_ylabel('Accuracy %')
ax.set_title('Train vs Test vs Cross-Validation Accuracy')
ax.legend()
for bar, val in zip(bars, [train_acc*100, test_acc*100, cv_scores.mean()*100]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("sleep_data/results/train_vs_test.png", dpi=100)
plt.close()

with open("sleep_data/processed/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved → sleep_data/processed/rf_model.pkl")
print(f"\nSummary:")
print(f"  Train  : {train_acc*100:.1f}%")
print(f"  Test   : {test_acc*100:.1f}%")
print(f"  CV     : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"  Gap    : {gap:.1f}%")