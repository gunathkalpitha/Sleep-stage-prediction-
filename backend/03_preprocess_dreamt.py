"""
Preprocess DREAMT features - create train/test splits
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

print("=" * 55)
print("  Preprocessing DREAMT Features")
print("=" * 55)

# Load balanced dataset
features_df = pd.read_csv("sleep_data/processed/features_dreamt.csv")
print(f"\n[1] Loaded balanced dataset")
print(f"    Shape: {features_df.shape}")

# Feature names
FEATURE_NAMES = [
    'mean_hr', 'std_hr', 'min_hr', 'hr_range',
    'rmssd', 'pnn50',
    'ppg_snr',
    'accel_mean', 'spectral_power',
    'accel_var', 'zcr', 'spectral_entropy'
]

# Extract features and labels
X = features_df[FEATURE_NAMES].values
y = features_df['sleep_stage'].values

print(f"\n[2] Features extracted")
print(f"    X shape: {X.shape}")
print(f"    y shape: {y.shape}")

# Check class distribution
print(f"\n[3] Class distribution:")
unique, counts = np.unique(y, return_counts=True)
CLASS_NAMES = {0: 'Wake', 1: 'Light', 2: 'Deep'}
for cls, count in zip(unique, counts):
    pct = count / len(y) * 100
    bar = '█' * int(pct / 2)
    print(f"    {CLASS_NAMES[cls]:6s}: {count:5d}  {bar} {pct:5.1f}%")

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[4] Train/test split (80/20 stratified)")
print(f"    Train: {len(X_train)} samples")
print(f"    Test:  {len(X_test)} samples")

# Save numpy arrays
os.makedirs("sleep_data/processed", exist_ok=True)
np.save("sleep_data/processed/X_train.npy", X_train)
np.save("sleep_data/processed/X_test.npy", X_test)
np.save("sleep_data/processed/y_train.npy", y_train)
np.save("sleep_data/processed/y_test.npy", y_test)

# Save feature names
with open("sleep_data/processed/feature_names.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)

print(f"\n✅ Preprocessing complete!")
print(f"   Ready for model training (04_train_model.py)")
print("=" * 55)
