"""
Preprocess DREAMT features - create train/test splits with IQR outlier removal & scaling
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 55)
print("  Preprocessing DREAMT Features (IQR Optimized)")
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

# ══════════════════════════════════════════════════════════
# [3] IQR-based outlier removal
# ══════════════════════════════════════════════════════════
print(f"\n[3] IQR outlier detection & removal")
X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
mask = pd.Series([True] * len(X_df))

outliers_removed = 0
for col in FEATURE_NAMES:
    Q1 = X_df[col].quantile(0.25)
    Q3 = X_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    col_mask = (X_df[col] >= lower_bound) & (X_df[col] <= upper_bound)
    removed = (~col_mask).sum()
    if removed > 0:
        print(f"    {col:20s}: {removed:4d} outliers removed")
        outliers_removed += removed
    mask = mask & col_mask

X = X[mask.values]
y = y[mask.values]

print(f"    Total rows removed: {outliers_removed}")
print(f"    Remaining samples : {len(X)}")

# Check class distribution after outlier removal
print(f"\n[4] Class distribution after cleanup:")
unique, counts = np.unique(y, return_counts=True)
CLASS_NAMES = {0: 'Wake', 1: 'Light', 2: 'Deep'}
for cls, count in zip(unique, counts):
    pct = count / len(y) * 100
    bar = '█' * int(pct / 2)
    print(f"    {CLASS_NAMES[cls]:6s}: {count:5d}  {bar} {pct:5.1f}%")

# Train/test split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[5] Train/test split (80/20 stratified)")
print(f"    Train: {len(X_train)} samples")
print(f"    Test:  {len(X_test)} samples")

# ══════════════════════════════════════════════════════════
# [6] Feature scaling (StandardScaler)
# ══════════════════════════════════════════════════════════
print(f"\n[6] Feature scaling (StandardScaler)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    ✅ Training set scaled")
print(f"       Mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"       Std:  {X_train_scaled.std(axis=0).round(4)}")

# Save numpy arrays (NO SMOTE - using real data only)
os.makedirs("sleep_data/processed", exist_ok=True)
np.save("sleep_data/processed/X_train.npy", X_train_scaled)
np.save("sleep_data/processed/X_test.npy", X_test_scaled)
np.save("sleep_data/processed/y_train.npy", y_train)
np.save("sleep_data/processed/y_test.npy", y_test)

# Save feature names
with open("sleep_data/processed/feature_names.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)

# Save scaler for inference
with open("sleep_data/processed/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Calculate class weights for imbalanced classes
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(y_train), 
                                      y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print(f"\n[7] Class weights (Proper weights for imbalanced data)")
print(f"    Calculated from real data distribution (no synthetic samples)")
for cls, weight in class_weight_dict.items():
    print(f"    {CLASS_NAMES[cls]:6s}: {weight:.3f}x (higher weight for minority classes)")

with open("sleep_data/processed/class_weights.pkl", "wb") as f:
    pickle.dump(class_weight_dict, f)

print(f"\n✅ Preprocessing complete (Real data + Class weights)!")
print(f"   [Train samples: {len(y_train)} - NO synthetic data]")
print(f"   [Deep sleep weight: {class_weight_dict[2]:.3f}x | Wake weight: {class_weight_dict[0]:.3f}x]")
print(f"   Ready for model training (04_train_model.py)")
print("=" * 55)
