import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import entropy
import glob
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  DREAMT Feature Extraction")
print("=" * 55)

FS_BVP  = 64
FS_ACC  = 64   # DREAMT resampled everything to 64Hz
EPOCH   = 30   # 30 second windows

FEATURE_NAMES = [
    'mean_hr', 'std_hr', 'min_hr', 'hr_range',
    'rmssd', 'pnn50',
    'ppg_snr',
    'accel_mean', 'spectral_power',
    'accel_var', 'zcr', 'spectral_entropy'
]

# Sleep stage mapping
# DREAMT uses: W=Wake, N1=Light, N2=Light, N3=Deep, R=REM
STAGE_MAP   = {'W': 0, 'N1': 1, 'N2': 1, 'N3': 2, 'R': -1,
               'P': -1, 'Missing': -1}
CLASS_NAMES = {0: 'Wake', 1: 'Light', 2: 'Deep'}

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

def hrv_features(ibi_epoch):
    """Use IBI directly — much more accurate than deriving from BVP"""
    ibi = ibi_epoch.dropna().values
    ibi = ibi[(ibi > 300) & (ibi < 2000)]  # valid IBI range in ms
    if len(ibi) < 4:
        return [30.0, 15.0]
    diff_ibi = np.diff(ibi)
    rmssd    = float(np.clip(np.sqrt(np.mean(diff_ibi**2)), 0, 300))
    pnn50    = float(np.clip(
        np.sum(np.abs(diff_ibi) > 50) / len(diff_ibi) * 100, 0, 100))
    return [rmssd, pnn50]

def ppg_snr(bvp, fs=FS_BVP):
    if len(bvp) < fs * 2:
        return [10.0]
    freqs, psd = signal.welch(bvp, fs=fs,
                               nperseg=min(len(bvp), fs*4))
    sig   = np.mean(psd[(freqs >= 0.5) & (freqs <= 4.0)]) + 1e-10
    noise = np.mean(psd[freqs > 4.0]) + 1e-10
    return [float(np.clip(10 * np.log10(sig / noise), -20, 40))]

def accel_features(acc_x, acc_y, acc_z, fs=FS_ACC):
    mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    freqs, psd = signal.welch(mag, fs=fs,
                               nperseg=min(len(mag), fs*4))
    band     = (freqs >= 0.1) & (freqs <= 1.0)
    sp       = float(np.trapz(psd[band], freqs[band])) \
               if np.any(band) else 0.0
    psd_norm = psd / (np.sum(psd) + 1e-10)
    zc       = np.diff(np.sign(mag - np.mean(mag)))
    return [float(np.mean(mag)), sp,
            float(np.var(mag)),
            float(np.sum(zc != 0) / len(mag)),
            float(entropy(psd_norm + 1e-10))]

def extract_epoch(bvp_ep, ibi_ep, acc_x, acc_y, acc_z):
    return (hr_features(bvp_ep)             +
            hrv_features(ibi_ep)            +
            ppg_snr(bvp_ep)                 +
            accel_features(acc_x, acc_y, acc_z))

files = sorted(glob.glob("sleep_data/dreamt/S*.csv"))
print(f"\nFound {len(files)} subject files")

SAMPLES_PER_EPOCH = EPOCH * FS_BVP   # 1920 samples

all_rows    = []
all_labels  = []
subject_ids = []

for file_path in files:
    subj = os.path.basename(file_path).replace("_whole_df.csv", "")
    print(f"\nProcessing {subj}...")

    try:
        df = pd.read_csv(file_path)

        # Find sleep stage column
        stage_col = None
        for col in df.columns:
            if 'sleep' in col.lower() or 'stage' in col.lower():
                stage_col = col
                break

        if stage_col is None:
            print(f"  ❌ No sleep stage column found — skipping")
            continue

        print(f"  Rows       : {len(df)}")
        print(f"  Duration   : {len(df)/FS_BVP/3600:.2f} hours")
        print(f"  Stage col  : {stage_col}")
        print(f"  Stages     : {df[stage_col].unique()}")

        n_epochs = len(df) // SAMPLES_PER_EPOCH
        print(f"  Epochs     : {n_epochs} × 30s windows")

        epoch_ok = 0
        for i in range(n_epochs):
            start = i * SAMPLES_PER_EPOCH
            end   = start + SAMPLES_PER_EPOCH

            epoch_df   = df.iloc[start:end]

            # Get dominant sleep stage for this epoch
            stage_vals = epoch_df[stage_col].value_counts()
            if len(stage_vals) == 0:
                continue
            dominant_stage = stage_vals.index[0]
            label = STAGE_MAP.get(str(dominant_stage), -1)
            if label == -1:
                continue

            # Extract signals
            bvp_ep = epoch_df['BVP'].values.astype(float)
            ibi_ep = epoch_df['IBI'] if 'IBI' in epoch_df.columns \
                     else pd.Series(dtype=float)
            acc_x  = epoch_df['ACC_X'].values.astype(float)
            acc_y  = epoch_df['ACC_Y'].values.astype(float)
            acc_z  = epoch_df['ACC_Z'].values.astype(float)

            feats = extract_epoch(bvp_ep, ibi_ep, acc_x, acc_y, acc_z)

            all_rows.append(feats)
            all_labels.append(label)
            subject_ids.append(subj)
            epoch_ok += 1

        print(f"  ✅ {epoch_ok} valid epochs extracted")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

# Build dataframe
features_df                = pd.DataFrame(all_rows, columns=FEATURE_NAMES)
features_df['sleep_stage'] = all_labels
features_df['stage_name']  = features_df['sleep_stage'].map(CLASS_NAMES)
features_df['subject']     = subject_ids

features_df = features_df.dropna(subset=FEATURE_NAMES)

print(f"\n{'='*55}")
print(f"  Final dataset shape : {features_df.shape}")
print(f"\n  Class distribution:")
for cls, name in CLASS_NAMES.items():
    cnt = (features_df['sleep_stage'] == cls).sum()
    pct = cnt / len(features_df) * 100
    bar = '█' * int(pct / 2)
    print(f"    {name:6s}: {cnt:5d}  {bar} {pct:.1f}%")

os.makedirs("sleep_data/processed", exist_ok=True)
save_path = "sleep_data/processed/features_12.csv"
features_df.to_csv(save_path, index=False)
print(f"\n  ✅ Saved to {save_path}")
print(f"{'='*55}")