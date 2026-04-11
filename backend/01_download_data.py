import kagglehub
import urllib.request
import os
import pickle
import glob

print("=" * 55)
print("  Smart Alarm — Data Download Script")
print("  Dataset 1: WESAD      (Kaggle)    → 12 raw signals")
print("  Dataset 2: Apple Watch (PhysioNet) → Sleep labels")
print("=" * 55)

# ══════════════════════════════════════════════════════════
# DATASET 1 — WESAD from Kaggle
# (BVP/PPG + ACC + IBI → all 12 features)
# ══════════════════════════════════════════════════════════
print("\n[1/2] Downloading WESAD dataset from Kaggle...")
print("      (This may take a few minutes — ~1.5 GB)")

wesad_path = kagglehub.dataset_download(
    "orvile/wesad-wearable-stress-affect-detection-dataset"
)
print(f"      ✅ WESAD saved to: {wesad_path}")

# Verify pickle files exist
pkl_files = sorted(glob.glob(
    os.path.join(wesad_path, "**/*.pkl"), recursive=True
))
print(f"      Subjects found: {len(pkl_files)}")
for f in pkl_files:
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"        • {os.path.basename(f)}  ({size_mb:.1f} MB)")

# Quick sanity check on first subject
print("\n      Verifying signals inside first subject file...")
with open(pkl_files[0], 'rb') as f:
    sample = pickle.load(f, encoding='latin1')

wrist = sample['signal']['wrist']
print(f"        Wrist signals: {list(wrist.keys())}")
print(f"        BVP shape:  {wrist['BVP'].shape}  @ 64 Hz")
print(f"        ACC shape:  {wrist['ACC'].shape}  @ 32 Hz")
print(f"        TEMP shape: {wrist['TEMP'].shape} @ 4 Hz")
print(f"      ✅ WESAD verified — all signals present!")

# ══════════════════════════════════════════════════════════
# DATASET 2 — Apple Watch from PhysioNet
# (HR + Accelerometer + Sleep stage labels)
# ══════════════════════════════════════════════════════════
print("\n[2/2] Downloading Apple Watch dataset from PhysioNet...")
print("      (Heart rate + motion + PSG sleep labels)")

# All 31 subject IDs
SUBJECTS = [
    "1360686", "1449548", "1455390", "1818471", "2598705",
    "2638030", "3509524", "3997827", "4018081", "4314139",
    "4426783", "46343",   "5132496", "5209809", "5498603",
    "6220552", "7749105", "784472",  "8173033", "8258170",
    "8686948", "8692923", "9106476", "9618981", "9961348",
]

# We only need 5 subjects — enough to train a solid model
DOWNLOAD_SUBJECTS = SUBJECTS[:5]

base = "https://physionet.org/files/sleep-accel/1.0.0/"
folders = ["heart_rate", "motion", "labels"]
suffixes = {
    "heart_rate": "_heartrate.txt",
    "motion":     "_acceleration.txt",
    "labels":     "_labeled_sleep.txt",
}

for folder in folders:
    os.makedirs(f"sleep_data/{folder}", exist_ok=True)

def download_with_progress(url, save_path):
    """Download with live progress bar, skip if exists"""
    if os.path.exists(save_path):
        size = os.path.getsize(save_path)
        print(f"        Already exists ({size/1024:.1f} KB) — skipping")
        return True

    tmp_path = save_path + ".tmp"

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct  = min(int(count * block_size * 100 / total_size), 100)
            done = count * block_size / 1024 / 1024
            tot  = total_size / 1024 / 1024
            print(f"\r        Progress: {pct:3d}%  "
                  f"({done:.1f} / {tot:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook)
        os.rename(tmp_path, save_path)
        size = os.path.getsize(save_path) / 1024 / 1024
        print(f"\r        ✅ Done  ({size:.1f} MB)            ")
        return True
    except KeyboardInterrupt:
        print(f"\n        ⚠️  Interrupted — run again to resume")
        return False
    except Exception as e:
        print(f"\n        ❌ Error: {e}")
        return False

total    = len(DOWNLOAD_SUBJECTS) * 3
current  = 0
failed   = []

for subject in DOWNLOAD_SUBJECTS:
    print(f"\n      Subject {subject}:")
    for folder in folders:
        current += 1
        filename  = subject + suffixes[folder]
        url       = f"{base}{folder}/{filename}"
        save_path = f"sleep_data/{folder}/{filename}"
        print(f"        [{current}/{total}] {folder}/{filename}")
        ok = download_with_progress(url, save_path)
        if not ok:
            failed.append(filename)

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  DOWNLOAD SUMMARY")
print("=" * 55)

print(f"\n  WESAD (Kaggle):")
print(f"    Location : {wesad_path}")
print(f"    Subjects : {len(pkl_files)} pkl files")
print(f"    Signals  : BVP (PPG), ACC, TEMP, EDA")
print(f"    Purpose  : Extract all 12 features")

aw_files = glob.glob("sleep_data/**/*.txt", recursive=True)
print(f"\n  Apple Watch (PhysioNet):")
print(f"    Location : sleep_data/")
print(f"    Files    : {len(aw_files)} txt files")
print(f"    Subjects : {len(DOWNLOAD_SUBJECTS)}")
print(f"    Purpose  : Sleep stage labels")

if failed:
    print(f"\n  ⚠️  Failed downloads ({len(failed)}):")
    for f in failed:
        print(f"    • {f}")
    print(f"  → Run script again to retry these files")
else:
    print(f"\n  ✅ All files downloaded successfully!")

print(f"\n  Your project folder structure:")
print(f"    sleep_data/")
print(f"      heart_rate/   ← Apple Watch HR (BPM)")
print(f"      motion/       ← Apple Watch accelerometer")
print(f"      labels/       ← PSG sleep stage labels")
print(f"    {wesad_path}/")
print(f"      S2.pkl ... S17.pkl  ← WESAD raw signals")
print(f"\n  ✅ Ready for Step 2: Feature Extraction!")
print("=" * 55)