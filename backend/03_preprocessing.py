import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, pickle, warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  Step 3 — Data Preprocessing (Fixed)")
print("=" * 55)

# ══════════════════════════════════════════════════════════
# STEP 3a — Load and inspect
# ══════════════════════════════════════════════════════════
print("\n[3a] Loading dataset...")

df = pd.read_csv("sleep_data/processed/features_12.csv")

FEATURE_NAMES = [
    'mean_hr', 'std_hr', 'min_hr', 'hr_range',
    'rmssd', 'pnn50', 'ppg_snr',
    'accel_mean', 'spectral_power',
    'accel_var', 'zcr', 'spectral_entropy'
]
CLASS_NAMES = {0: 'Wake', 1: 'Light', 2: 'Deep'}

print(f"    Shape       : {df.shape}")
print(f"    Total epochs: {len(df)}")

print(f"\n    Class distribution (BEFORE balancing):")
for cls, name in CLASS_NAMES.items():
    cnt = (df['sleep_stage'] == cls).sum()
    pct = cnt / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"      {name:6s}: {cnt:5d}  {bar} {pct:.1f}%")

print(f"\n    Actual feature ranges in your data:")
for feat in FEATURE_NAMES:
    print(f"      {feat:20s}: "
          f"min={df[feat].min():.3f}  "
          f"max={df[feat].max():.3f}  "
          f"mean={df[feat].mean():.3f}")

# ══════════════════════════════════════════════════════════
# STEP 3b — Remove outliers using IQR method
# (data-driven — works with ANY data values)
# ══════════════════════════════════════════════════════════
print("\n[3b] Removing outliers using IQR method...")

# IQR = Interquartile Range method
# Remove values that are more than 3×IQR away from median
# This is data-driven — no hardcoded ranges needed

before = len(df)
mask   = pd.Series([True] * len(df))

for feat in FEATURE_NAMES:
    Q1  = df[feat].quantile(0.25)
    Q3  = df[feat].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 3.0 * IQR
    upper = Q3 + 3.0 * IQR

    feat_mask = (df[feat] >= lower) & (df[feat] <= upper)
    removed   = (~feat_mask).sum()

    if removed > 0:
        print(f"    {feat:20s}: removed {removed:4d} rows "
              f"(outside {lower:.2f} – {upper:.2f})")
    mask = mask & feat_mask

df = df[mask].reset_index(drop=True)
print(f"\n    Rows before : {before}")
print(f"    Rows after  : {len(df)}")
print(f"    Removed     : {before - len(df)} outlier rows")

print(f"\n    Class distribution (AFTER outlier removal):")
for cls, name in CLASS_NAMES.items():
    cnt = (df['sleep_stage'] == cls).sum()
    pct = cnt / len(df) * 100 if len(df) > 0 else 0
    bar = '█' * int(pct / 2)
    print(f"      {name:6s}: {cnt:5d}  {bar} {pct:.1f}%")

# ══════════════════════════════════════════════════════════
# STEP 3c — Handle class imbalance using SMOTE
# ══════════════════════════════════════════════════════════
print("\n[3c] Balancing classes with SMOTE...")

X = df[FEATURE_NAMES].values
y = df['sleep_stage'].values

print(f"    Before SMOTE:")
for cls, name in CLASS_NAMES.items():
    cnt = int((y == cls).sum())
    pct = cnt / len(y) * 100
    bar = '█' * int(pct / 2)
    print(f"      {name:6s}: {cnt:5d}  {bar} {pct:.1f}%")

# Check minimum class size for SMOTE
min_class_size = min((y == cls).sum() for cls in CLASS_NAMES.keys())
k_neighbors    = min(3, min_class_size - 1)

print(f"\n    Minimum class size : {min_class_size}")
print(f"    Using k_neighbors  : {k_neighbors}")

try:
    smote        = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_bal, y_bal = smote.fit_resample(X, y)

    print(f"\n    After SMOTE:")
    for cls, name in CLASS_NAMES.items():
        cnt = int((y_bal == cls).sum())
        pct = cnt / len(y_bal) * 100
        bar = '█' * int(pct / 2)
        print(f"      {name:6s}: {cnt:5d}  {bar} {pct:.1f}%")
    print(f"\n    Total epochs after balancing: {len(y_bal)}")

except Exception as e:
    print(f"    ⚠️  SMOTE failed: {e}")
    print(f"    Using original unbalanced data")
    X_bal, y_bal = X, y

# ══════════════════════════════════════════════════════════
# STEP 3d — Feature scaling
# ══════════════════════════════════════════════════════════
print("\n[3d] Scaling features (StandardScaler)...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

print(f"    Sample feature (mean_hr):")
print(f"      Before: mean={X_bal[:,0].mean():.2f}  "
      f"std={X_bal[:,0].std():.2f}")
print(f"      After : mean={X_scaled[:,0].mean():.4f}  "
      f"std={X_scaled[:,0].std():.4f}")
print(f"    ✅ All 12 features scaled to mean≈0, std≈1")

# ══════════════════════════════════════════════════════════
# STEP 3e — Train / Test split (80/20)
# ══════════════════════════════════════════════════════════
print("\n[3e] Train / Test split (80% / 20%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_bal,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_bal     # keeps class ratio same in both sets
)

print(f"    Total  : {len(X_scaled):5d} samples")
print(f"    Train  : {len(X_train):5d} samples (80%)")
print(f"    Test   : {len(X_test):5d}  samples (20%)")

print(f"\n    Train distribution:")
for cls, name in CLASS_NAMES.items():
    cnt = int((y_train == cls).sum())
    pct = cnt / len(y_train) * 100
    print(f"      {name:6s}: {cnt:5d}  ({pct:.1f}%)")

print(f"\n    Test distribution:")
for cls, name in CLASS_NAMES.items():
    cnt = int((y_test == cls).sum())
    pct = cnt / len(y_test) * 100
    print(f"      {name:6s}: {cnt:5d}  ({pct:.1f}%)")

# ══════════════════════════════════════════════════════════
# STEP 3f — Save everything
# ══════════════════════════════════════════════════════════
print("\n[3f] Saving preprocessed data...")

os.makedirs("sleep_data/processed", exist_ok=True)

np.save("sleep_data/processed/X_train.npy", X_train)
np.save("sleep_data/processed/X_test.npy",  X_test)
np.save("sleep_data/processed/y_train.npy", y_train)
np.save("sleep_data/processed/y_test.npy",  y_test)

with open("sleep_data/processed/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("sleep_data/processed/feature_names.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)

print(f"    ✅ X_train.npy : {X_train.shape}")
print(f"    ✅ X_test.npy  : {X_test.shape}")
print(f"    ✅ y_train.npy : {y_train.shape}")
print(f"    ✅ y_test.npy  : {y_test.shape}")
print(f"    ✅ scaler.pkl  : saved (needed for real-time prediction)")

# ══════════════════════════════════════════════════════════
# STEP 3g — Feature distribution plot
# ══════════════════════════════════════════════════════════
print("\n[3g] Saving feature distribution plot...")

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("Feature Distributions by Sleep Stage", fontsize=14)
axes      = axes.flatten()
colors    = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71'}

for i, feat in enumerate(FEATURE_NAMES):
    ax       = axes[i]
    feat_idx = i
    for cls, name in CLASS_NAMES.items():
        vals = X_bal[y_bal == cls, feat_idx]
        ax.hist(vals, bins=30, alpha=0.6,
                color=colors[cls], label=name, density=True)
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel("Value", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "sleep_data/processed/feature_distributions.png"
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"    ✅ Plot saved: {plot_path}")

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  PREPROCESSING COMPLETE")
print("=" * 55)
print(f"""
  Steps completed:
    3a  Loaded {before} epochs from CSV
    3b  Removed outliers using IQR method
    3c  Balanced classes with SMOTE
    3d  Scaled all features (mean=0, std=1)
    3e  Split 80% train / 20% test
    3f  Saved arrays + scaler
    3g  Saved distribution plot

  Files saved:
    sleep_data/processed/X_train.npy  ← training features
    sleep_data/processed/X_test.npy   ← testing features
    sleep_data/processed/y_train.npy  ← training labels
    sleep_data/processed/y_test.npy   ← testing labels
    sleep_data/processed/scaler.pkl   ← for real-time use

  ✅ Ready for Step 4: Train Random Forest!
""")
print("=" * 55)