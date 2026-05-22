# =============================================================
# NOTEBOOK 2: BIDMC — Multi-Signal Exploration & Preprocessing
# =============================================================
# Run on Kaggle with "bidmc-ppg-dataset" attached,
# or locally after downloading.
#
# Dataset path assumption (Kaggle):
#   /kaggle/input/bidmc-ppg-dataset/
# Locally: change BASE_PATH below.
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io as sio
import wfdb

BASE_PATH  = "/kaggle/input/bidmc-ppg-dataset"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS_SIGNAL  = 125    # ECG, PPG, Resp sampled at 125 Hz
FS_NUMERIC = 1      # HR, SpO2, RR sampled at 1 Hz
N_RECORDS  = 53     # BIDMC has 53 recordings


# =============================================================
# SECTION 1 — UNDERSTAND THE BIDMC FILE STRUCTURE
# =============================================================
print("=" * 60)
print("SECTION 1: BIDMC file structure")
print("=" * 60)
# Each subject has:
#   bidmc##.hea  — waveform header
#   bidmc##.dat  — raw waveform (ECG, PPG, Impedance Resp) at 125 Hz
#   bidmc##n.hea — numerics header
#   bidmc##n.dat — numerics (HR, SpO2, RR) at 1 Hz
#   bidmc##.breath — manual breath annotations

# List files for subject 01
files = os.listdir(BASE_PATH)
subject01 = [f for f in sorted(files) if f.startswith("bidmc01")]
print(f"Files for subject 01: {subject01}")


# =============================================================
# SECTION 2 — LOAD ONE SUBJECT: WAVEFORMS + NUMERICS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 2: Loading signals for one subject")
print("=" * 60)

def load_bidmc_subject(subject_id, base_path):
    """
    Load all signals and numeric parameters for one BIDMC subject.
    Returns a dict with raw arrays and metadata.
    """
    prefix = os.path.join(base_path, f"bidmc{subject_id:02d}")

    # ── Waveforms (125 Hz) ──────────────────────────────────
    record  = wfdb.rdrecord(prefix)
    signals = record.p_signal.astype(np.float32)   # shape: (N, 3)
    sig_names = record.sig_name                    # e.g. ['II', 'PLETH', 'RESP']

    ecg_idx  = sig_names.index("II")       if "II"    in sig_names else 0
    ppg_idx  = sig_names.index("PLETH")    if "PLETH" in sig_names else 1
    resp_idx = sig_names.index("RESP")     if "RESP"  in sig_names else 2

    ecg  = signals[:, ecg_idx]
    ppg  = signals[:, ppg_idx]
    resp = signals[:, resp_idx]
    duration_s = len(ecg) / FS_SIGNAL

    # ── Numeric parameters (1 Hz) ───────────────────────────
    rec_num = wfdb.rdrecord(prefix + "n")
    nums    = rec_num.p_signal.astype(np.float32)
    num_names = rec_num.sig_name

    def get_num(name):
        if name in num_names:
            return nums[:, num_names.index(name)]
        return np.full(nums.shape[0], np.nan)

    hr   = get_num("HR")     # Heart Rate (bpm)
    spo2 = get_num("SpO2")   # Oxygen Saturation (%)
    rr   = get_num("RESP")   # Respiratory Rate (breaths/min)
    pr   = get_num("PULSE")  # Pulse Rate from PPG (bpm)

    return {
        "id":          subject_id,
        "duration_s":  duration_s,
        "ecg":         ecg,
        "ppg":         ppg,
        "resp":        resp,
        "hr":          hr,
        "spo2":        spo2,
        "rr":          rr,
        "pr":          pr,
        "sig_names":   sig_names,
    }

subj = load_bidmc_subject(1, BASE_PATH)
print(f"Subject 01 — duration: {subj['duration_s']:.1f}s  ({subj['duration_s']/60:.1f} min)")
print(f"Waveform signals : {subj['sig_names']}")
print(f"ECG  shape       : {subj['ecg'].shape}")
print(f"PPG  shape       : {subj['ppg'].shape}")
print(f"RESP shape       : {subj['resp'].shape}")
print(f"\nNumeric parameters (mean ± std):")
for name, arr in [("HR (bpm)", subj["hr"]), ("SpO2 (%)", subj["spo2"]),
                  ("RR (br/min)", subj["rr"]), ("PR (bpm)", subj["pr"])]:
    valid = arr[~np.isnan(arr)]
    print(f"  {name:15s}: {valid.mean():.1f} ± {valid.std():.1f}  "
          f"[{valid.min():.1f} – {valid.max():.1f}]")


# =============================================================
# SECTION 3 — VISUALISE RAW SIGNALS (one subject, first 30s)
# =============================================================
print("\n" + "=" * 60)
print("SECTION 3: Visualising raw signals")
print("=" * 60)

window_s = 30
n_samples = window_s * FS_SIGNAL
t_wave = np.arange(n_samples) / FS_SIGNAL
t_num  = np.arange(window_s)

fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=False)

axes[0].plot(t_wave, subj["ecg"][:n_samples],  color='#378ADD', linewidth=0.7)
axes[0].set_title("ECG — Lead II (125 Hz)")
axes[0].set_ylabel("mV")

axes[1].plot(t_wave, subj["ppg"][:n_samples],  color='#D85A30', linewidth=0.7)
axes[1].set_title("PPG — Photoplethysmogram (125 Hz)")
axes[1].set_ylabel("a.u.")

axes[2].plot(t_wave, subj["resp"][:n_samples], color='#7F77DD', linewidth=0.7)
axes[2].set_title("Impedance Respiratory Signal (125 Hz)")
axes[2].set_ylabel("a.u.")

axes[3].plot(t_num,  subj["hr"][:window_s],    color='#1D9E75', linewidth=1.2, marker='o', ms=3)
axes[3].set_title("Heart Rate (1 Hz)")
axes[3].set_ylabel("bpm")

axes[4].plot(t_num,  subj["spo2"][:window_s],  color='#BA7517', linewidth=1.2, marker='o', ms=3)
axes[4].set_title("SpO₂ (1 Hz)")
axes[4].set_ylabel("%")
axes[4].set_xlabel("Time (s)")
axes[4].set_ylim(85, 102)

plt.suptitle("BIDMC Subject 01 — all signals (first 30 s)", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bidmc_raw_signals.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bidmc_raw_signals.png")


# =============================================================
# SECTION 4 — SIGNAL PREPROCESSING (BIDMC)
# =============================================================
print("\n" + "=" * 60)
print("SECTION 4: Preprocessing BIDMC signals")
print("=" * 60)

def preprocess_ecg_bidmc(ecg, fs=125):
    """Bandpass 0.5–40 Hz + baseline removal for ECG."""
    nyq = fs / 2
    b, a = sig.butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    ecg_f = sig.filtfilt(b, a, ecg)
    baseline = sig.medfilt(ecg_f, kernel_size=int(fs * 0.2) | 1)
    return ecg_f - baseline

def preprocess_ppg(ppg, fs=125):
    """Bandpass 0.5–8 Hz for PPG — captures cardiac pulse only."""
    nyq = fs / 2
    b, a = sig.butter(4, [0.5/nyq, 8.0/nyq], btype='band')
    return sig.filtfilt(b, a, ppg)

def preprocess_resp(resp, fs=125):
    """Lowpass 1 Hz for respiratory signal — only breathing rate."""
    nyq = fs / 2
    b, a = sig.butter(4, 1.0/nyq, btype='low')
    return sig.filtfilt(b, a, resp)

def clean_numerics(arr, min_val, max_val):
    """Replace physiologically impossible values with NaN, then interpolate."""
    arr = arr.copy().astype(float)
    arr[(arr < min_val) | (arr > max_val)] = np.nan
    # Linear interpolation for short gaps
    s = pd.Series(arr)
    arr = s.interpolate(method='linear', limit=10).values
    return arr

ecg_clean  = preprocess_ecg_bidmc(subj["ecg"],  fs=FS_SIGNAL)
ppg_clean  = preprocess_ppg(subj["ppg"],         fs=FS_SIGNAL)
resp_clean = preprocess_resp(subj["resp"],        fs=FS_SIGNAL)
hr_clean   = clean_numerics(subj["hr"],   min_val=20,  max_val=250)
spo2_clean = clean_numerics(subj["spo2"], min_val=50,  max_val=100)
rr_clean   = clean_numerics(subj["rr"],   min_val=4,   max_val=60)

print("ECG  — before: std={:.4f}  after: std={:.4f}".format(subj["ecg"].std(),  ecg_clean.std()))
print("PPG  — before: std={:.4f}  after: std={:.4f}".format(subj["ppg"].std(),  ppg_clean.std()))
print("RESP — before: std={:.4f}  after: std={:.4f}".format(subj["resp"].std(), resp_clean.std()))

# Compare raw vs clean for ECG and PPG
fig, axes = plt.subplots(2, 2, figsize=(16, 7))
t = t_wave  # first 30 s

axes[0,0].plot(t, subj["ecg"][:n_samples],  color='gray',    lw=0.7);  axes[0,0].set_title("ECG raw")
axes[0,1].plot(t, ecg_clean[:n_samples],    color='#378ADD', lw=0.7);  axes[0,1].set_title("ECG clean")
axes[1,0].plot(t, subj["ppg"][:n_samples],  color='gray',    lw=0.7);  axes[1,0].set_title("PPG raw")
axes[1,1].plot(t, ppg_clean[:n_samples],    color='#D85A30', lw=0.7);  axes[1,1].set_title("PPG clean")
for ax in axes.flat: ax.set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bidmc_preprocessing.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bidmc_preprocessing.png")


# =============================================================
# SECTION 5 — EXTRACT CLINICAL PARAMETERS PER SUBJECT
# =============================================================
print("\n" + "=" * 60)
print("SECTION 5: Extracting clinical summary parameters")
print("=" * 60)

# These thresholds are standard clinical reference ranges
THRESHOLDS = {
    "hr":   {"low": 60, "high": 100, "unit": "bpm"},
    "spo2": {"low": 94, "high": 100, "unit": "%"},
    "rr":   {"low": 12, "high": 20,  "unit": "br/min"},
}

def classify_param(value, low, high):
    """Return clinical status string."""
    if np.isnan(value):      return "Unknown"
    if value < low:          return "Low"
    if value > high:         return "High"
    return "Normal"

def extract_subject_summary(subj_data):
    """
    Produces the structured clinical feature dict used later for report generation.
    This is the bridge between signal processing and NLP.
    """
    ecg_c  = preprocess_ecg_bidmc(subj_data["ecg"])
    ppg_c  = preprocess_ppg(subj_data["ppg"])
    resp_c = preprocess_resp(subj_data["resp"])
    hr_c   = clean_numerics(subj_data["hr"],   20, 250)
    spo2_c = clean_numerics(subj_data["spo2"], 50, 100)
    rr_c   = clean_numerics(subj_data["rr"],   4,  60)

    # Scalar summaries (median is more robust than mean for clinical params)
    hr_med   = float(np.nanmedian(hr_c))
    spo2_med = float(np.nanmedian(spo2_c))
    rr_med   = float(np.nanmedian(rr_c))

    # SpO2: flag hypoxia
    spo2_min = float(np.nanmin(spo2_c))
    hypoxia  = spo2_min < 90.0

    # PPG signal quality (simple SNR proxy)
    ppg_snr = float(np.std(ppg_c) / (np.std(ppg_c - sig.medfilt(ppg_c, 5)) + 1e-6))
    ppg_quality = "Good" if ppg_snr > 3 else ("Fair" if ppg_snr > 1.5 else "Poor")

    # RR variability (proxy for breathing regularity)
    rr_cv = float(np.nanstd(rr_c) / (np.nanmean(rr_c) + 1e-6))
    rr_regular = rr_cv < 0.2

    summary = {
        # ── Numeric values ──────────────────────────────
        "subject_id":     subj_data["id"],
        "duration_s":     subj_data["duration_s"],
        "hr_bpm":         round(hr_med, 1),
        "spo2_pct":       round(spo2_med, 1),
        "spo2_min_pct":   round(spo2_min, 1),
        "rr_per_min":     round(rr_med, 1),
        "ppg_quality":    ppg_quality,
        # ── Clinical status labels ───────────────────────
        "hr_status":      classify_param(hr_med,   **THRESHOLDS["hr"]),
        "spo2_status":    classify_param(spo2_med, **THRESHOLDS["spo2"]),
        "rr_status":      classify_param(rr_med,   **THRESHOLDS["rr"]),
        "hypoxia_flag":   hypoxia,
        "breathing_regular": rr_regular,
    }
    return summary

subj01_summary = extract_subject_summary(subj)
print("Structured clinical summary for Subject 01:")
for k, v in subj01_summary.items():
    print(f"  {k:25s}: {v}")


# =============================================================
# SECTION 6 — PROCESS ALL 53 SUBJECTS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 6: Processing all 53 subjects")
print("=" * 60)

all_summaries = []
failed = []

for sid in range(1, N_RECORDS + 1):
    try:
        subj_data = load_bidmc_subject(sid, BASE_PATH)
        summary   = extract_subject_summary(subj_data)
        all_summaries.append(summary)
        if sid % 10 == 0:
            print(f"  Processed {sid}/{N_RECORDS}")
    except Exception as e:
        print(f"  Subject {sid:02d} failed: {e}")
        failed.append(sid)

bidmc_df = pd.DataFrame(all_summaries)
print(f"\nSuccessfully processed: {len(bidmc_df)}/{N_RECORDS} subjects")
if failed:
    print(f"Failed: {failed}")

print("\nDataset overview:")
print(bidmc_df[["hr_bpm","spo2_pct","rr_per_min"]].describe().round(2))

print("\nClinical status distribution across all subjects:")
for param in ["hr_status", "spo2_status", "rr_status"]:
    print(f"\n  {param}:")
    print(bidmc_df[param].value_counts().to_string(header=False))

print(f"\nHypoxia events (SpO2 < 90% at any point): {bidmc_df['hypoxia_flag'].sum()} subjects")
print(f"PPG quality distribution:")
print(bidmc_df["ppg_quality"].value_counts().to_string(header=False))

bidmc_df.to_csv(os.path.join(OUTPUT_DIR, "bidmc_summaries.csv"), index=False)
print("\nSaved: bidmc_summaries.csv")


# =============================================================
# SECTION 7 — VISUALISE POPULATION-LEVEL DISTRIBUTIONS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 7: Population-level distributions")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
params = [
    ("hr_bpm",    "Heart Rate (bpm)",       '#1D9E75', (20,  250), (60, 100)),
    ("spo2_pct",  "SpO₂ (%)",               '#BA7517', (70,  101), (94, 100)),
    ("rr_per_min","Respiratory Rate (br/min)",'#7F77DD',(4,   60),  (12, 20)),
]
for ax, (col, label, color, xlim, (lo, hi)) in zip(axes, params):
    vals = bidmc_df[col].dropna()
    ax.hist(vals, bins=15, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(lo, color='red',   linestyle='--', linewidth=1.2, label=f'Low  < {lo}')
    ax.axvline(hi, color='orange', linestyle='--', linewidth=1.2, label=f'High > {hi}')
    ax.set_xlabel(label)
    ax.set_ylabel("Subjects")
    ax.set_xlim(xlim)
    ax.legend(fontsize=8)

plt.suptitle("BIDMC — population distribution of clinical parameters (n=53)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bidmc_population_distributions.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bidmc_population_distributions.png")


# =============================================================
# SECTION 8 — BUILD STRUCTURED NLP INPUT STRINGS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 8: Generating structured NLP input strings")
print("=" * 60)
# This is the bridge to your T5/BART report generation pipeline.
# Each string becomes the "source" sequence for seq2seq training.

def build_nlp_input(summary, ecg_condition="Unknown", ecg_severity="Unknown"):
    """
    Combines BIDMC-derived parameters with PTB-XL classifier output
    into one structured string for the NLP model.
    When you have a trained classifier, pass its output here.
    """
    hr_val    = summary["hr_bpm"]
    spo2_val  = summary["spo2_pct"]
    rr_val    = summary["rr_per_min"]
    hr_st     = summary["hr_status"].lower()
    spo2_st   = summary["spo2_status"].lower()
    rr_st     = summary["rr_status"].lower()
    hypoxia   = "yes" if summary["hypoxia_flag"]    else "no"
    breath_r  = "regular" if summary["breathing_regular"] else "irregular"
    ppg_q     = summary["ppg_quality"].lower()

    return (
        f"ecg_condition: {ecg_condition} | "
        f"severity: {ecg_severity} | "
        f"heart_rate: {hr_val} bpm {hr_st} | "
        f"spo2: {spo2_val}% {spo2_st} | "
        f"respiratory_rate: {rr_val} breaths_per_min {rr_st} | "
        f"hypoxia: {hypoxia} | "
        f"breathing_pattern: {breath_r} | "
        f"ppg_quality: {ppg_q}"
    )

# Example using a placeholder ECG condition (replace with classifier output later)
example_input = build_nlp_input(
    subj01_summary,
    ecg_condition="atrial_fibrillation",
    ecg_severity="moderate"
)
print("Example NLP input string:")
print(f"\n  {example_input}\n")

# Generate for all BIDMC subjects with placeholder ECG labels
bidmc_df["nlp_input"] = bidmc_df.apply(
    lambda row: build_nlp_input(row.to_dict(),
                                 ecg_condition="pending_classifier",
                                 ecg_severity="pending_classifier"),
    axis=1
)

print("Sample NLP input strings:")
for i, row in bidmc_df.head(3).iterrows():
    print(f"\n  Subject {int(row['subject_id']):02d}:")
    print(f"  {row['nlp_input']}")

bidmc_df.to_csv(os.path.join(OUTPUT_DIR, "bidmc_with_nlp_inputs.csv"), index=False)
print("\nSaved: bidmc_with_nlp_inputs.csv")
print("\n✓ BIDMC exploration and preprocessing complete.")
print("Next step: replace 'pending_classifier' with CNN-LSTM outputs from PTB-XL.")
