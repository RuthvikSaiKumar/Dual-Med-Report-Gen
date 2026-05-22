# =============================================================
# NOTEBOOK 1: PTB-XL — Data Exploration & Preprocessing
# =============================================================
# Run this on Kaggle with the ptb-xl-dataset attached,
# or locally after downloading from Kaggle.
#
# Dataset path assumption (Kaggle):
#   /kaggle/input/ptb-xl-dataset/
# Locally: change BASE_PATH to wherever you extracted the zip.
# =============================================================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wfdb
import scipy.signal as sig
from collections import Counter

# ── 0. CONFIG ─────────────────────────────────────────────────
BASE_PATH   = "/kaggle/input/ptb-xl-dataset"   # change if local
SAMPLING_HZ = 100          # PTB-XL has 100 Hz and 500 Hz; 100 is fine for classification
LEADS       = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
SUPERCLASS_MAP = {
    'NORM': 'Normal',
    'MI':   'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD':   'Conduction Disturbance',
    'HYP':  'Hypertrophy'
}
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# SECTION 1 — LOAD METADATA
# =============================================================
print("=" * 60)
print("SECTION 1: Loading metadata")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE_PATH, "ptbxl_database.csv"), index_col="ecg_id")

# scp_codes is stored as a string like "{'NORM': 100.0}" — parse it
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

print(f"Total records  : {len(df)}")
print(f"Total patients : {df['patient_id'].nunique()}")
print(f"\nColumn overview:")
print(df[['age','sex','height','weight','nurse','site','device','strat_fold']].describe())
print(f"\nSex distribution:\n{df['sex'].value_counts()}")
print(f"\nAge range: {df['age'].min():.0f} – {df['age'].max():.0f} yrs  |  median: {df['age'].median():.0f} yrs")


# =============================================================
# SECTION 2 — LOAD LABEL MAPPINGS & MAP TO SUPERCLASSES
# =============================================================
print("\n" + "=" * 60)
print("SECTION 2: Label mapping → 5 superclasses")
print("=" * 60)

scp_df = pd.read_csv(os.path.join(BASE_PATH, "scp_statements.csv"), index_col=0)
# Keep only diagnostic statements
diag_scp = scp_df[scp_df["diagnostic"] == 1.0]
print(f"Diagnostic SCP codes available: {len(diag_scp)}")
print(diag_scp[["description","diagnostic_class"]].head(10).to_string())

def get_superclasses(scp_dict, threshold=50.0):
    """
    Returns list of superclasses for a record.
    threshold: minimum likelihood % to include a label (0 = definite, 100 = possible)
    """
    labels = set()
    for code, likelihood in scp_dict.items():
        if code in diag_scp.index and likelihood >= threshold:
            sc = diag_scp.loc[code, "diagnostic_class"]
            if pd.notna(sc):
                labels.add(sc)
    return list(labels) if labels else []

df["superclass"] = df["scp_codes"].apply(get_superclasses)

# Explode for per-class counts
label_counts = Counter(
    label for labels in df["superclass"] for label in labels
)
print(f"\nSuperclass distribution (multi-label, threshold=50%):")
for cls, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    pct = count / len(df) * 100
    name = SUPERCLASS_MAP.get(cls, cls)
    print(f"  {cls:6s} ({name:30s}): {count:5d}  ({pct:.1f}%)")

# Records with no diagnostic label
no_label = df["superclass"].apply(len) == 0
print(f"\nRecords with no superclass label: {no_label.sum()} — these will be excluded from training")

# Also add rhythm labels
RHYTHM_CODES = {
    "AFIB": "Atrial Fibrillation",
    "AFLT": "Atrial Flutter",
    "STACH": "Sinus Tachycardia",
    "SBRAD": "Sinus Bradycardia",
    "SVTAC": "Supraventricular Tachycardia",
}
def get_rhythm(scp_dict, threshold=50.0):
    rhythms = []
    for code, likelihood in scp_dict.items():
        if code in RHYTHM_CODES and likelihood >= threshold:
            rhythms.append(code)
    return rhythms

df["rhythm_labels"] = df["scp_codes"].apply(get_rhythm)
rhythm_counts = Counter(r for rs in df["rhythm_labels"] for r in rs)
print(f"\nRhythm label distribution:")
for code, cnt in sorted(rhythm_counts.items(), key=lambda x: -x[1]):
    print(f"  {code}: {cnt} records  ({RHYTHM_CODES[code]})")


# =============================================================
# SECTION 3 — VISUALISE LABEL DISTRIBUTION
# =============================================================
print("\n" + "=" * 60)
print("SECTION 3: Visualising label distribution")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart — superclass counts
classes  = list(label_counts.keys())
counts   = [label_counts[c] for c in classes]
colors   = ['#378ADD','#D85A30','#7F77DD','#1D9E75','#BA7517']
axes[0].barh([SUPERCLASS_MAP.get(c, c) for c in classes], counts, color=colors[:len(classes)])
axes[0].set_xlabel("Number of records")
axes[0].set_title("PTB-XL — superclass distribution (multi-label)")
for i, v in enumerate(counts):
    axes[0].text(v + 50, i, str(v), va='center', fontsize=10)

# Age distribution by sex
male   = df[df['sex'] == 0]['age'].dropna()
female = df[df['sex'] == 1]['age'].dropna()
axes[1].hist(male,   bins=20, alpha=0.6, label='Male',   color='#378ADD')
axes[1].hist(female, bins=20, alpha=0.6, label='Female', color='#D4537E')
axes[1].set_xlabel("Age (years)")
axes[1].set_ylabel("Count")
axes[1].set_title("PTB-XL — age & sex distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ptbxl_label_distribution.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ptbxl_label_distribution.png")


# =============================================================
# SECTION 4 — LOAD A RAW ECG WAVEFORM
# =============================================================
print("\n" + "=" * 60)
print("SECTION 4: Loading a raw ECG waveform")
print("=" * 60)

def load_ecg(row, base_path, sampling_rate=100):
    """Load ECG signal for one record using wfdb."""
    if sampling_rate == 100:
        file_path = os.path.join(base_path, row["filename_lr"])
    else:
        file_path = os.path.join(base_path, row["filename_hr"])
    record = wfdb.rdrecord(file_path)
    return record.p_signal.astype(np.float32)  # shape: (1000, 12) at 100 Hz

# Pick one NORM and one AF record to compare
norm_idx = df[df["superclass"].apply(lambda x: x == ["NORM"])].index[0]
af_idx   = df[df["rhythm_labels"].apply(lambda x: "AFIB" in x)].index[0]

ecg_norm = load_ecg(df.loc[norm_idx], BASE_PATH, SAMPLING_HZ)
ecg_af   = load_ecg(df.loc[af_idx],  BASE_PATH, SAMPLING_HZ)

print(f"ECG shape: {ecg_norm.shape}  → (timesteps={SAMPLING_HZ*10}, leads=12)")
print(f"Value range (NORM): {ecg_norm.min():.4f}  to  {ecg_norm.max():.4f} mV")
print(f"Value range (AFIB): {ecg_af.min():.4f}  to  {ecg_af.max():.4f} mV")

# Plot Lead II for both
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
t = np.arange(ecg_norm.shape[0]) / SAMPLING_HZ

axes[0].plot(t, ecg_norm[:, 1], color='#378ADD', linewidth=0.8)
axes[0].set_title(f"Normal ECG — Lead II  (record {norm_idx})")
axes[0].set_ylabel("Amplitude (mV)")
axes[0].set_xlabel("Time (s)")
axes[0].axhline(0, color='gray', linewidth=0.4, linestyle='--')

axes[1].plot(t, ecg_af[:, 1], color='#D85A30', linewidth=0.8)
axes[1].set_title(f"Atrial Fibrillation — Lead II  (record {af_idx})")
axes[1].set_ylabel("Amplitude (mV)")
axes[1].set_xlabel("Time (s)")
axes[1].axhline(0, color='gray', linewidth=0.4, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "raw_ecg_comparison.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: raw_ecg_comparison.png")


# =============================================================
# SECTION 5 — SIGNAL PREPROCESSING
# =============================================================
print("\n" + "=" * 60)
print("SECTION 5: Signal preprocessing")
print("=" * 60)

def bandpass_filter(signal_1d, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """Butterworth bandpass — removes baseline wander and high-freq noise."""
    nyq = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return sig.filtfilt(b, a, signal_1d)

def remove_baseline(signal_1d, fs=100, window_s=0.2):
    """Median filter for baseline wander removal."""
    window = int(fs * window_s)
    if window % 2 == 0:
        window += 1
    baseline = sig.medfilt(signal_1d, kernel_size=window)
    return signal_1d - baseline

def normalize_ecg(signal_2d):
    """Z-score normalize each lead independently."""
    mean = signal_2d.mean(axis=0, keepdims=True)
    std  = signal_2d.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (signal_2d - mean) / std

def preprocess_ecg(signal_2d, fs=100):
    """
    Full preprocessing pipeline for one ECG record.
    Input:  (timesteps, 12) raw mV values
    Output: (timesteps, 12) clean, normalized values
    """
    out = np.zeros_like(signal_2d)
    for lead_idx in range(signal_2d.shape[1]):
        lead = signal_2d[:, lead_idx].copy()
        lead = remove_baseline(lead, fs=fs)
        lead = bandpass_filter(lead, fs=fs)
        out[:, lead_idx] = lead
    out = normalize_ecg(out)
    return out

# Apply to our example records
ecg_norm_clean = preprocess_ecg(ecg_norm, fs=SAMPLING_HZ)
ecg_af_clean   = preprocess_ecg(ecg_af,   fs=SAMPLING_HZ)

print("Before preprocessing — Lead II stats (NORM):")
print(f"  mean={ecg_norm[:,1].mean():.4f}  std={ecg_norm[:,1].std():.4f}  min={ecg_norm[:,1].min():.4f}  max={ecg_norm[:,1].max():.4f}")
print("After preprocessing — Lead II stats (NORM):")
print(f"  mean={ecg_norm_clean[:,1].mean():.4f}  std={ecg_norm_clean[:,1].std():.4f}  min={ecg_norm_clean[:,1].min():.4f}  max={ecg_norm_clean[:,1].max():.4f}")

# Visualise before vs after for one lead
fig, axes = plt.subplots(2, 2, figsize=(16, 7))
t = np.arange(ecg_norm.shape[0]) / SAMPLING_HZ

axes[0,0].plot(t, ecg_norm[:,1],       color='gray',    linewidth=0.7)
axes[0,0].set_title("NORM — Lead II (raw)")
axes[0,0].set_ylabel("mV")

axes[0,1].plot(t, ecg_norm_clean[:,1], color='#378ADD', linewidth=0.7)
axes[0,1].set_title("NORM — Lead II (after preprocessing)")

axes[1,0].plot(t, ecg_af[:,1],         color='gray',    linewidth=0.7)
axes[1,0].set_title("AFIB — Lead II (raw)")
axes[1,0].set_ylabel("mV")

axes[1,1].plot(t, ecg_af_clean[:,1],   color='#D85A30', linewidth=0.7)
axes[1,1].set_title("AFIB — Lead II (after preprocessing)")

for ax in axes.flat:
    ax.set_xlabel("Time (s)")
    ax.axhline(0, color='lightgray', linewidth=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "preprocessing_comparison.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: preprocessing_comparison.png")


# =============================================================
# SECTION 6 — BUILD THE FULL PREPROCESSED DATASET
# =============================================================
print("\n" + "=" * 60)
print("SECTION 6: Building the full preprocessed dataset")
print("=" * 60)

# Filter: keep only records that have at least one superclass label
df_clean = df[df["superclass"].apply(len) > 0].copy()
print(f"Records after removing unlabelled: {len(df_clean)}")

# One-hot encode the 5 superclasses for multi-label classification
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
for sc in SUPERCLASSES:
    df_clean[f"label_{sc}"] = df_clean["superclass"].apply(lambda x: int(sc in x))

print("\nLabel matrix sample:")
print(df_clean[[f"label_{sc}" for sc in SUPERCLASSES]].head(10))

print("\nLabel co-occurrence (how many records have 2+ labels):")
df_clean["num_labels"] = df_clean[[f"label_{sc}" for sc in SUPERCLASSES]].sum(axis=1)
print(df_clean["num_labels"].value_counts().sort_index())

# Train / val / test split using official strat_fold column
train_df = df_clean[df_clean["strat_fold"] <= 8]
val_df   = df_clean[df_clean["strat_fold"] == 9]
test_df  = df_clean[df_clean["strat_fold"] == 10]
print(f"\nSplit sizes  →  train: {len(train_df)}  |  val: {len(val_df)}  |  test: {len(test_df)}")

# Save metadata splits
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_metadata.csv"))
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_metadata.csv"))
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_metadata.csv"))
print("Saved: train/val/test_metadata.csv")


# =============================================================
# SECTION 7 — BATCH PREPROCESS & SAVE AS NUMPY ARRAYS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 7: Batch preprocessing and saving arrays")
print("=" * 60)
# NOTE: This saves one .npy file per split.
# Each array has shape (N, 1000, 12) — N records, 1000 timesteps, 12 leads.
# This takes ~5–10 minutes on Kaggle CPU.

def build_array(meta_df, base_path, fs=100, verbose_every=500):
    X, y, ids = [], [], []
    for i, (ecg_id, row) in enumerate(meta_df.iterrows()):
        try:
            raw   = load_ecg(row, base_path, fs)
            clean = preprocess_ecg(raw, fs)
            label = [row[f"label_{sc}"] for sc in SUPERCLASSES]
            X.append(clean)
            y.append(label)
            ids.append(ecg_id)
        except Exception as e:
            print(f"  Skipped {ecg_id}: {e}")
        if (i + 1) % verbose_every == 0:
            print(f"  Processed {i+1}/{len(meta_df)} ...")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(ids)

print("Processing train set...")
X_train, y_train, ids_train = build_array(train_df, BASE_PATH, SAMPLING_HZ)
print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")

print("Processing val set...")
X_val, y_val, ids_val = build_array(val_df, BASE_PATH, SAMPLING_HZ)
print(f"  X_val:   {X_val.shape}   y_val:   {y_val.shape}")

print("Processing test set...")
X_test, y_test, ids_test = build_array(test_df, BASE_PATH, SAMPLING_HZ)
print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
print("\nAll arrays saved to ./outputs/")
print("Shape convention: (N_records, 1000_timesteps, 12_leads)")


# =============================================================
# SECTION 8 — SANITY CHECKS ON SAVED ARRAYS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 8: Sanity checks")
print("=" * 60)

print(f"NaN in X_train: {np.isnan(X_train).sum()}")
print(f"Inf in X_train: {np.isinf(X_train).sum()}")
print(f"\nX_train value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"X_train mean: {X_train.mean():.4f}  std: {X_train.std():.4f}")

print(f"\nLabel distribution in train set:")
for i, sc in enumerate(SUPERCLASSES):
    n = y_train[:, i].sum()
    print(f"  {sc}: {int(n):5d}  ({n/len(y_train)*100:.1f}%)")

# Visualise all 12 leads for one clean record
sample_ecg = X_train[0]
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(6, 2, hspace=0.6, wspace=0.3)
t   = np.arange(sample_ecg.shape[0]) / SAMPLING_HZ
colors_leads = ['#378ADD','#D85A30','#7F77DD','#1D9E75',
                '#BA7517','#D4537E','#378ADD','#D85A30',
                '#7F77DD','#1D9E75','#BA7517','#D4537E']
for i, lead_name in enumerate(LEADS):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    ax.plot(t, sample_ecg[:, i], linewidth=0.7, color=colors_leads[i])
    ax.set_title(f"Lead {lead_name}", fontsize=9)
    ax.set_yticks([])
    ax.axhline(0, color='lightgray', linewidth=0.4)
    if i >= 10:
        ax.set_xlabel("Time (s)", fontsize=8)

fig.suptitle(f"All 12 leads — preprocessed record {ids_train[0]}  |  label: {[SUPERCLASSES[j] for j in range(5) if y_train[0,j]==1]}",
             fontsize=11, y=1.01)
plt.savefig(os.path.join(OUTPUT_DIR, "all_12_leads_sample.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: all_12_leads_sample.png")

print("\n✓ PTB-XL exploration and preprocessing complete.")
print("Next step: load X_train.npy / y_train.npy into your CNN-LSTM model.")
