import mne
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Load the PSG file (raw signals) ──────────────────────────
psg_file = "sleep_data/SC4001E0-PSG.edf"
raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)

# See what channels (signal types) are available
print("=== Available Channels ===")
print(raw.ch_names)
# You'll see: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental, etc.

print(f"\nSampling frequency: {raw.info['sfreq']} Hz")
print(f"Total duration: {raw.times[-1]/3600:.2f} hours")

# ── 2. Load the Hypnogram (sleep stage labels) ──────────────────
hyp_file = "sleep_data/SC4001EC-Hypnogram.edf"
annotations = mne.read_annotations(hyp_file)

print("\n=== Sleep Stage Labels Found ===")
# Count how many of each stage exists
from collections import Counter
stage_counts = Counter(annotations.description)
for stage, count in stage_counts.items():
    print(f"  {stage}: {count} epochs")

# ── 3. Plot a sample of the raw signals ─────────────────────────
print("\nPlotting first 5 minutes of signals...")
raw.plot(duration=300, n_channels=4, title="Raw Sleep Signals (first 5 mins)")
plt.show()