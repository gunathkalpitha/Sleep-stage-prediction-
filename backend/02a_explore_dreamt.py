import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import glob

print("=" * 55)
print("  DREAMT Dataset Explorer")
print("=" * 55)

files = sorted(glob.glob("sleep_data/dreamt/S*.csv"))
print(f"\nFound {len(files)} subject files\n")

# Load first file and inspect
df = pd.read_csv(files[0], nrows=5)

print(f"Columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")

# Load full first subject for stats
df_full = pd.read_csv(files[0])

print(f"\nShape       : {df_full.shape}")
print(f"Duration    : {len(df_full) / 64 / 3600:.2f} hours")

print(f"\nFirst 5 rows:")
print(df_full.head())

print(f"\nSleep stage distribution:")
if 'Sleep_Stage' in df_full.columns:
    print(df_full['Sleep_Stage'].value_counts())
else:
    # find the label column
    for col in df_full.columns:
        if 'sleep' in col.lower() or 'stage' in col.lower() or 'label' in col.lower():
            print(f"Label column found: {col}")
            print(df_full[col].value_counts())
            break

print(f"\nSample signal values:")
signal_cols = ['BVP', 'IBI', 'HR', 'ACC_X', 'ACC_Y', 'ACC_Z']
for col in signal_cols:
    if col in df_full.columns:
        print(f"  {col:8s}: min={df_full[col].min():.3f}  "
              f"max={df_full[col].max():.3f}  "
              f"mean={df_full[col].mean():.3f}")