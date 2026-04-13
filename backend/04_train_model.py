import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

model = RandomForestClassifier(
    n_estimators      = 100,
    max_features      = 'sqrt',
    random_state      = 42,
    n_jobs            = -1
)

print("\nTraining Random Forest...")
start = time.time()
model.fit(X_train, y_train)
print(f"Done in {time.time() - start:.1f}s")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy * 100:.1f}%")
print(f"Target   : 82.0%")
print(f"Status   : {'Exceeded' if accuracy >= 0.82 else 'Below'} target\n")

print(classification_report(
    y_test, y_pred,
    target_names=list(CLASS_NAMES.values())
))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
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
plt.title(f'Confusion Matrix  (Accuracy: {accuracy*100:.1f}%)')
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

with open("sleep_data/processed/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to sleep_data/processed/rf_model.pkl")