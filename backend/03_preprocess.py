import pickle, os, glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import kagglehub
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  Step 2 — Feature Extraction (Fixed v3)")
print("=" * 55)

# ══════════════════════════════════════════════════════════
# PART A — Load WESAD
# ══════════════════════════════════════════════════════════
print("\n[A] Loading WESAD dataset...")
wesad_path = kagglehub.dataset_download(
    "orvile/wesad-wearable-stress-affect-detection-dataset"
)
pkl_files = sorted(glob.glob(
    os.path.join(wesad_path, "**/*.pkl"), recursive=True
))
print(f"    Found {len(pkl_files)} subject files")

# ══════════════════════════════════════════════════════════
# PART B — Feature extraction functions (all 12)
# ══════════════════════════════════════════════════════════
FS_BVP = 64
FS_ACC = 32
EPOCH  = 30

FEATURE_NAMES = [
    'mean_hr', 'std_hr', 'min_hr', 'hr_range',
    'rmssd', 'pnn50',
    'ppg_snr',
    'accel_mean', 'spectral_power',
    'accel_var', 'zcr', 'spectral_entropy'
]

def hr_features(bvp, fs=FS_BVP):
    peaks, _ = signal.find_peaks(bvp, distance=int(fs*0.4),
                                  height=np.mean(bvp))
    if len(peaks) < 3:
        return [60.0, 5.0, 55.0, 10.0]
    rr = np.diff(peaks) / fs
    hr = 60.0 / rr
    hr = hr[(hr > 30) & (hr < 200)]
    if len(hr) < 2:
        return [60.0, 5.0, 55.0, 10.0]
    return [float(np.mean(hr)), float(np.std(hr)),
            float(np.min(hr)),  float(np.max(hr) - np.min(hr))]

def hrv_features(bvp, fs=FS_BVP):
    peaks, _ = signal.find_peaks(bvp, distance=int(fs*0.4),
                                  height=np.mean(bvp))
    if len(peaks) < 4:
        return [30.0, 15.0]
    rr_ms   = np.diff(peaks) / fs * 1000
    diff_rr = np.diff(rr_ms)
    rmssd   = float(np.clip(np.sqrt(np.mean(diff_rr**2)), 0, 300))
    pnn50   = float(np.clip(
        np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100, 0, 100))
    return [rmssd, pnn50]

def ppg_snr(bvp, fs=FS_BVP):
    if len(bvp) < fs * 2:
        return [10.0]
    freqs, psd = signal.welch(bvp, fs=fs,
                               nperseg=min(len(bvp), fs*4))
    sig   = np.mean(psd[(freqs >= 0.5) & (freqs <= 4.0)]) + 1e-10
    noise = np.mean(psd[freqs > 4.0]) + 1e-10
    return [float(np.clip(10 * np.log10(sig / noise), -20, 40))]

def accel_features(acc, fs=FS_ACC):
    mag = np.sqrt(np.sum(acc**2, axis=1))
    freqs, psd = signal.welch(mag, fs=fs,
                               nperseg=min(len(mag), fs*4))
    band = (freqs >= 0.1) & (freqs <= 1.0)
    sp   = float(np.trapz(psd[band], freqs[band])) if np.any(band) else 0.0
    psd_norm = psd / (np.sum(psd) + 1e-10)
    zc   = np.diff(np.sign(mag - np.mean(mag)))
    return [float(np.mean(mag)), sp, float(np.var(mag)),
            float(np.sum(zc != 0) / len(mag)),
            float(entropy(psd_norm + 1e-10))]

def extract_epoch(bvp_ep, acc_ep):
    return (hr_features(bvp_ep) + hrv_features(bvp_ep) +
            ppg_snr(bvp_ep)     + accel_features(acc_ep))

# ══════════════════════════════════════════════════════════
# PART C — Process WESAD subjects
# ══════════════════════════════════════════════════════════
print("\n[C] Extracting features from WESAD subjects...")

epoch_bvp = EPOCH * FS_BVP
epoch_acc = EPOCH * FS_ACC
all_rows  = []
subj_ids  = []

for pkl_path in pkl_files:
    subj = os.path.basename(pkl_path).replace(".pkl", "")
    print(f"\n    Processing {subj}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        BVP = data['signal']['wrist']['BVP'].flatten()
        ACC = data['signal']['wrist']['ACC']
        n   = min(len(BVP) // epoch_bvp, len(ACC) // epoch_acc)
        for i in range(n):
            bvp_ep = BVP[i*epoch_bvp:(i+1)*epoch_bvp]
            acc_ep = ACC[i*epoch_acc:(i+1)*epoch_acc]
            all_rows.append(extract_epoch(bvp_ep, acc_ep))
            subj_ids.append(subj)
        print(f"      ✅ {n} epochs  ({len(BVP)/FS_BVP/60:.1f} mins)")
    except Exception as e:
        print(f"      ❌ Error: {e}")

wesad_df            = pd.DataFrame(all_rows, columns=FEATURE_NAMES)
wesad_df['subject'] = subj_ids
print(f"\n    WESAD features: {wesad_df.shape}")

# ══════════════════════════════════════════════════════════
# PART D — Load Apple Watch labels (FIXED PARSER)
# ══════════════════════════════════════════════════════════
print("\n[D] Loading Apple Watch sleep labels...")

# ── THE FIX: file uses SPACE as separator, not comma ──────
# Each line looks like:  "540 0"  or  "30 -1"
# We must split on whitespace, not comma

def load_label_file(filepath):
    """
    Correctly parse Apple Watch label files.
    Format: each line = 'timestamp stage'
    e.g.  '540 0'  means timestamp=540, stage=0
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()          # split on whitespace
            if len(parts) >= 2:
                try:
                    timestamp = int(parts[0])
                    stage     = int(parts[1])
                    rows.append({'timestamp': timestamp, 'stage': stage})
                except ValueError:
                    continue             # skip header lines if any
    return pd.DataFrame(rows)

# ── Stage mapping (PhysioNet Apple Watch format) ──────────
# -1 = undefined/unscored → skip
#  0 = Wake
#  1 = N1 (Light sleep)
#  2 = N2 (Light sleep)
#  3 = N3 (Deep sleep)
#  5 = REM → skip for now

STAGE_MAP   = {-1: None, 0: 0, 1: 1, 2: 1, 3: 2, 5: None}
CLASS_NAMES = {0: 'Wake', 1: 'Light', 2: 'Deep'}

label_files = sorted(glob.glob("sleep_data/labels/*.txt"))
all_labels  = []

for lf in label_files:
    df = load_label_file(lf)

    print(f"\n    {os.path.basename(lf)}:")
    print(f"      Raw rows     : {len(df)}")
    print(f"      Stage values : {sorted(df['stage'].unique())}")

    # Apply stage map
    df['label'] = df['stage'].map(STAGE_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    wake  = (df['label'] == 0).sum()
    light = (df['label'] == 1).sum()
    deep  = (df['label'] == 2).sum()
    print(f"      Valid epochs : {len(df)}"
          f"  (W:{wake} L:{light} D:{deep})")

    all_labels.append(df)

labels_df = pd.concat(all_labels, ignore_index=True)

print(f"\n    ✅ Total valid labels: {len(labels_df)}")
print(f"\n    Overall distribution:")
for cls, name in CLASS_NAMES.items():
    cnt = (labels_df['label'] == cls).sum()
    pct = cnt / len(labels_df) * 100
    bar = '█' * int(pct / 2)
    print(f"      {name:6s}: {cnt:5d}  {bar} {pct:.1f}%")

# ══════════════════════════════════════════════════════════
# PART E — Combine features + labels
# ══════════════════════════════════════════════════════════
print("\n[E] Building combined dataset...")

n_use = min(len(wesad_df), len(labels_df))
print(f"    WESAD epochs  : {len(wesad_df)}")
print(f"    Label epochs  : {len(labels_df)}")
print(f"    Using         : {n_use} epochs")

final_df = wesad_df.iloc[:n_use].copy().reset_index(drop=True)
final_df['sleep_stage'] = labels_df['label'].values[:n_use]
final_df['stage_name']  = final_df['sleep_stage'].map(CLASS_NAMES)

# Drop rows with NaN in features only
before   = len(final_df)
final_df = final_df.dropna(subset=FEATURE_NAMES)
print(f"    Dropped NaN   : {before - len(final_df)} rows")
print(f"    Final shape   : {final_df.shape}")

# ══════════════════════════════════════════════════════════
# PART F — Save
# ══════════════════════════════════════════════════════════
os.makedirs("sleep_data/processed", exist_ok=True)
save_path = "sleep_data/processed/features_12.csv"
final_df.to_csv(save_path, index=False)

print("\n" + "=" * 55)
print("  FEATURE EXTRACTION COMPLETE")
print("=" * 55)
print(f"\n  Saved   : {save_path}")
print(f"  Shape   : {final_df.shape}")
print(f"  Columns : {list(final_df.columns)}")

print(f"\n  Class distribution:")
for cls, name in CLASS_NAMES.items():
    cnt = (final_df['sleep_stage'] == cls).sum()
    pct = cnt / len(final_df) * 100 if len(final_df) > 0 else 0
    bar = '█' * int(pct / 2)
    print(f"    {name:6s}: {cnt:4d} epochs  {bar} {pct:.1f}%")

print(f"\n  Feature means:")
print(final_df[FEATURE_NAMES].mean().round(3).to_string())
print(f"\n  ✅ Ready for Step 3: Model Training!")
print("=" * 55)