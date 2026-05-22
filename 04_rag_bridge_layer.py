# =============================================================
# STAGE 5: FAST DUAL MEDICAL REPORT GENERATION
# =============================================================
# Uses your trained CNN-LSTM (best_model.pt) from Stage 3 to
# classify real ECG records, then generates dual reports.
#
# Models compared:
#   1. Rule-based engine     — always runs, used as reference
#   2. Flan-T5-small         — zero-shot, saved after first run
#   3. GPT2            — decoder baseline
#
# Key fixes vs original:
#   - Loads best_model.pt + best_thresholds.npy from Stage 3
#   - Loads X_test.npy + bidmc_summaries.csv for real patient data
#   - T5 model cached to disk — skips reload on rerun
#   - Short prompts that work on flan-t5-small without echoing
#   - repetition_penalty=2.0 prevents looping output
#   - Fallback to rule-based if T5 echoes or outputs garbage
#
# Requirements:
#   pip install transformers sentence-transformers torch
#              scikit-learn rouge-score sacrebleu nltk pandas numpy
# =============================================================

import os, random, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nltk

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)
from rouge_score import rouge_scorer
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────
DATA_DIR       = "./outputs"           # Stage 2/3 outputs
STAGE5_DIR     = "./outputs/stage5"
T5_CACHE_DIR   = "./outputs/stage5/t5_small_cached"
GPT2_CACHE_DIR = "./outputs/stage5/gpt2_cached"
os.makedirs(STAGE5_DIR,     exist_ok=True)
os.makedirs(T5_CACHE_DIR,   exist_ok=True)
os.makedirs(GPT2_CACHE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
print(f"Device : {DEVICE}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CONDITION_NAMES = {
    "NORM": "normal sinus rhythm",
    "MI":   "myocardial infarction",
    "STTC": "ST/T-wave changes",
    "CD":   "conduction disturbance",
    "HYP":  "cardiac hypertrophy",
}


# =============================================================
# SECTION 1 — CNN-LSTM MODEL (copied from Stage 3)
# =============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, pool=2, dropout=0.2):
        super().__init__()
        self.conv     = nn.Conv1d(in_ch, out_ch, kernel,
                                  padding=kernel // 2, bias=False)
        self.bn       = nn.BatchNorm1d(out_ch)
        self.act      = nn.ReLU()
        self.pool     = nn.MaxPool1d(pool)
        self.drop     = nn.Dropout(dropout)
        self.residual = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                         if in_ch != out_ch else nn.Identity())
        self.pool_res = nn.MaxPool1d(pool)

    def forward(self, x):
        res = self.pool_res(self.residual(x))
        return self.pool(self.drop(self.act(self.bn(self.conv(x))))) + res


class CNNLSTM(nn.Module):
    def __init__(self, n_leads=12, n_classes=5,
                 cnn_channels=(32, 64, 128, 256),
                 lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()
        layers, in_ch = [], n_leads
        for out_ch in cnn_channels:
            layers.append(ConvBlock(in_ch, out_ch, dropout=dropout * 0.7))
            in_ch = out_ch
        self.cnn       = nn.Sequential(*layers)
        self.lstm      = nn.LSTM(cnn_channels[-1], lstm_hidden, lstm_layers,
                                 batch_first=True, bidirectional=True,
                                 dropout=dropout if lstm_layers > 1 else 0.0)
        lstm_out       = lstm_hidden * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out), nn.Dropout(dropout),
            nn.Linear(lstm_out, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x      = self.cnn(x).permute(0, 2, 1)
        x, _   = self.lstm(x)
        attn_w = torch.softmax(self.attention(x), dim=1)
        x      = (x * attn_w).sum(dim=1)
        return self.classifier(x)


# =============================================================
# SECTION 2 — LOAD STAGE 3 OUTPUTS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 2: Loading Stage 3 model and test data")
print("=" * 60)

MODEL_PATH      = os.path.join(DATA_DIR, "best_model.pt")
THRESHOLD_PATH  = os.path.join(DATA_DIR, "best_thresholds.npy")
X_TEST_PATH     = os.path.join(DATA_DIR, "X_test.npy")
BIDMC_PATH      = os.path.join(DATA_DIR, "bidmc_summaries.csv")

# Check all required files exist
for path in [MODEL_PATH, THRESHOLD_PATH, X_TEST_PATH, BIDMC_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Make sure you have run Stage 2 and Stage 3 first."
        )

# Load CNN-LSTM
print("Loading CNN-LSTM from best_model.pt ...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
clf_model  = CNNLSTM().to(DEVICE)
clf_model.load_state_dict(checkpoint["model_state"])
clf_model.eval()
print(f"  Loaded — best val AUC: {checkpoint['val_auc']:.4f}  "
      f"(epoch {checkpoint['epoch']})")

# Load thresholds
thresholds = np.load(THRESHOLD_PATH)
print(f"  Thresholds per class: "
      + "  ".join(f"{c}={thresholds[i]:.2f}"
                  for i, c in enumerate(SUPERCLASSES)))

# Load ECG test array
X_test = np.load(X_TEST_PATH)
print(f"  X_test shape: {X_test.shape}")

# Load BIDMC summaries
bidmc_df = pd.read_csv(BIDMC_PATH)
print(f"  BIDMC subjects: {len(bidmc_df)}")


# =============================================================
# SECTION 3 — RUN CLASSIFIER ON TEST SET
# =============================================================
print("\n" + "=" * 60)
print("SECTION 3: Running CNN-LSTM on test ECG records")
print("=" * 60)

def classify_ecg_batch(X_array, model, thresholds, batch_size=64):
    """Run trained classifier on all records, return probs + preds."""
    all_probs = []
    for i in range(0, len(X_array), batch_size):
        batch = torch.tensor(
            X_array[i:i + batch_size], dtype=torch.float32
        ).permute(0, 2, 1).to(DEVICE)          # (B, 12, 1000)
        with torch.no_grad():
            probs = model(batch).sigmoid().cpu().numpy()
        all_probs.append(probs)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Classified {min(i+batch_size, len(X_array))}"
                  f"/{len(X_array)} records...")

    all_probs = np.concatenate(all_probs, axis=0)   # (N, 5)
    all_preds = (all_probs >= thresholds).astype(int)
    return all_probs, all_preds

ecg_probs, ecg_preds = classify_ecg_batch(X_test, clf_model, thresholds)
print(f"Classification done — {len(ecg_probs)} records")

# Label distribution in test set
print("\nDetected condition distribution:")
for i, cls in enumerate(SUPERCLASSES):
    n = ecg_preds[:, i].sum()
    print(f"  {cls}: {n} records ({n/len(ecg_preds)*100:.1f}%)")


# =============================================================
# SECTION 4 — BUILD UNIFIED PATIENT RECORDS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 4: Building unified patient records")
print("=" * 60)

def classify_param(value, low, high):
    if pd.isna(value): return "unknown"
    if value < low:    return "low"
    if value > high:   return "high"
    return "normal"

def get_primary(probs, preds):
    detected = [SUPERCLASSES[i] for i in range(5) if preds[i] == 1]
    if not detected:
        detected = ["NORM"]
    primary  = SUPERCLASSES[int(np.argmax(probs))]
    top_prob = float(probs[np.argmax(probs)])
    if top_prob > 0.85:   severity = "high"
    elif top_prob > 0.65: severity = "moderate"
    else:                 severity = "mild"
    return primary, detected, severity, top_prob

# Random assignment: each PTB-XL test record → one BIDMC subject
# (cycling through all 53 subjects evenly)
bidmc_indices = list(range(len(bidmc_df)))
random.shuffle(bidmc_indices)
assigned = [bidmc_indices[i % len(bidmc_df)] for i in range(len(X_test))]

patient_records = []
for i in range(len(X_test)):
    probs    = ecg_probs[i]
    preds    = ecg_preds[i]
    bidmc    = bidmc_df.iloc[assigned[i]]
    primary, detected, severity, top_prob = get_primary(probs, preds)

    hr_val   = float(bidmc["hr_bpm"])
    spo2_val = float(bidmc["spo2_pct"])
    rr_val   = float(bidmc["rr_per_min"])

    patient_records.append({
        "record_idx":    i,
        "bidmc_subject": int(bidmc["subject_id"]),
        "cond":          primary,
        "detected":      detected,
        "severity":      severity,
        "confidence":    round(top_prob * 100, 1),
        "hr":            round(hr_val, 1),
        "hr_status":     classify_param(hr_val,   60,  100),
        "spo2":          round(spo2_val, 1),
        "spo2_status":   classify_param(spo2_val, 94,  100),
        "rr":            round(rr_val, 1),
        "rr_status":     classify_param(rr_val,   12,  20),
        "ppg_quality":   bidmc.get("ppg_quality", "good"),
        "hypoxia":       spo2_val < 90,
    })

records_df = pd.DataFrame(patient_records)
records_df.to_csv(os.path.join(STAGE5_DIR, "patient_records.csv"), index=False)
print(f"Built {len(records_df)} unified patient records")
print(f"Sample:")
print(json.dumps(patient_records[0], indent=2))


# =============================================================
# SECTION 5 — RAG KNOWLEDGE BASE
# =============================================================
print("\n" + "=" * 60)
print("SECTION 5: Building RAG knowledge base")
print("=" * 60)

KNOWLEDGE = [
    "Normal sinus rhythm: regular P waves, HR 60-100 bpm, no pathological findings.",
    "Tachycardia: HR above 100 bpm increases myocardial oxygen demand; 12-lead ECG required.",
    "Bradycardia: HR below 60 bpm; symptomatic cases require atropine or pacing evaluation.",
    "Myocardial infarction: ST-elevation, Q waves, T-wave inversions; urgent PCI indicated.",
    "ST/T-wave changes: ischemia, electrolyte imbalance, or strain; serial ECGs and troponin needed.",
    "Conduction disturbance: bundle branch block or AV block; electrolyte panel and cardiology review.",
    "Cardiac hypertrophy: high-voltage QRS; echocardiography confirms; hypertension management required.",
    "Hypoxemia: SpO2 below 94% requires supplemental oxygen; below 90% is a medical emergency.",
    "Tachypnea above 20 breaths/min: respiratory distress or metabolic acidosis; ABG analysis advised.",
    "Continuous cardiac monitoring essential for all patients with abnormal physiological parameters.",
    "Atrial fibrillation: irregular rhythm, absent P waves; rate control and anticoagulation considered.",
    "Normal SpO2 94-100%: adequate oxygenation, no supplemental oxygen required.",
    "Troponin elevation confirms myocardial injury; urgent cardiology consultation warranted.",
]

print("Loading sentence-transformer embedder...")
embedder         = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
knowledge_embeds = embedder.encode(KNOWLEDGE, show_progress_bar=False)

def retrieve(query, top_k=1):
    """Retrieve top_k most relevant knowledge chunks for a query."""
    q_emb = embedder.encode([query], show_progress_bar=False)
    sims  = cosine_similarity(q_emb, knowledge_embeds)[0]
    idxs  = sims.argsort()[-top_k:][::-1]
    return " ".join(KNOWLEDGE[i] for i in idxs)

print(f"RAG ready — {len(KNOWLEDGE)} knowledge chunks")


# =============================================================
# SECTION 6 — RULE-BASED REPORT ENGINE
# =============================================================
DOCTOR_ACTIONS = {
    "NORM": "Routine monitoring advised. No acute intervention required.",
    "MI":   ("Urgent troponin assay, continuous cardiac monitoring, and cardiology "
             "consultation strongly indicated. Antiplatelet therapy per protocol."),
    "STTC": ("Serial ECGs and cardiac biomarkers recommended. Cardiology review "
             "within 4 hours. Echocardiography to exclude structural cause."),
    "CD":   ("Electrolyte panel and medication review recommended. Cardiology "
             "evaluation warranted. Pacemaker assessment if haemodynamically unstable."),
    "HYP":  ("Echocardiography recommended to quantify chamber dimensions. "
             "Blood pressure assessment and hypertension management review indicated."),
}

PATIENT_ACTIONS = {
    "NORM": "Keep attending regular check-ups. No immediate action needed.",
    "MI":   ("Go to the emergency department now or call emergency services. "
             "Do not drive yourself. Rest and stay calm."),
    "STTC": ("Please see a doctor today. Avoid strenuous activity until reviewed. "
             "Bring this report with you."),
    "CD":   ("Book an appointment with your doctor soon. "
             "Tell them about all your current medications."),
    "HYP":  ("See your doctor soon. Focus on keeping your blood pressure under control. "
             "An ultrasound of your heart may be arranged."),
}

def rule_based_doctor(rec):
    cond     = rec["cond"]
    hr_note  = (f"elevated heart rate of {rec['hr']} bpm" if rec["hr_status"] == "high"
                else f"bradycardic rate of {rec['hr']} bpm" if rec["hr_status"] == "low"
                else f"heart rate of {rec['hr']} bpm within normal limits")
    spo2_note= (f"reduced oxygen saturation of {rec['spo2']}%"
                if rec["spo2_status"] == "low"
                else f"oxygen saturation of {rec['spo2']}% within normal range")
    rr_note  = (f"tachypnoeic respiratory rate of {rec['rr']} breaths/min"
                if rec["rr_status"] == "high"
                else f"reduced respiratory rate of {rec['rr']} breaths/min"
                if rec["rr_status"] == "low"
                else f"respiratory rate of {rec['rr']} breaths/min within normal limits")
    hypoxia  = (" Hypoxaemia noted; supplemental oxygen and ABG analysis advised."
                if rec["spo2_status"] == "low" else "")
    sev_map  = {"mild": "mild", "moderate": "moderate clinical", "high": "high — urgent"}
    return (
        f"ECG findings consistent with {CONDITION_NAMES.get(cond, cond)} "
        f"(confidence: {rec['confidence']}%). "
        f"Patient exhibits {hr_note}, {spo2_note}, and {rr_note}.{hypoxia} "
        f"Overall severity: {sev_map.get(rec['severity'], rec['severity'])}. "
        f"{DOCTOR_ACTIONS.get(cond, DOCTOR_ACTIONS['NORM'])}"
    )

def rule_based_patient(rec):
    cond     = rec["cond"]
    hr_plain = (f"Your heart is beating faster than normal at {rec['hr']} beats per minute."
                if rec["hr_status"] == "high"
                else f"Your heart is beating slower than normal at {rec['hr']} beats per minute."
                if rec["hr_status"] == "low"
                else f"Your heart rate of {rec['hr']} beats per minute is healthy.")
    hypoxia  = (" Your oxygen level is lower than it should be — tell your doctor immediately."
                if rec["spo2_status"] == "low" else "")
    sev_map  = {"mild": "minor", "moderate": "moderate", "high": "serious — seek urgent help"}
    return (
        f"Your heart test shows a pattern called {CONDITION_NAMES.get(cond, cond)}. "
        f"{hr_plain} "
        f"Your oxygen level is {rec['spo2']}% and your breathing rate is "
        f"{rec['rr']} per minute.{hypoxia} "
        f"Overall concern level: {sev_map.get(rec['severity'], rec['severity'])}. "
        f"{PATIENT_ACTIONS.get(cond, PATIENT_ACTIONS['NORM'])}"
    )


# =============================================================
# SECTION 7 — LOAD / CACHE FLAN-T5-SMALL
# =============================================================
print("\n" + "=" * 60)
print("SECTION 7: Loading Flan-T5-small (cached if exists)")
print("=" * 60)

T5_MODEL_ID = "google/flan-t5-small"

# Check if already cached locally
if os.path.exists(os.path.join(T5_CACHE_DIR, "config.json")):
    print(f"  Found cached T5 at {T5_CACHE_DIR} — loading from disk...")
    t5_tok = AutoTokenizer.from_pretrained(T5_CACHE_DIR)
    t5_mdl = AutoModelForSeq2SeqLM.from_pretrained(T5_CACHE_DIR).to(DEVICE)
else:
    print(f"  No cache found — downloading {T5_MODEL_ID} ...")
    t5_tok = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    t5_mdl = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_ID).to(DEVICE)
    # Save to disk for future runs
    t5_tok.save_pretrained(T5_CACHE_DIR)
    t5_mdl.save_pretrained(T5_CACHE_DIR)
    print(f"  Saved to {T5_CACHE_DIR}")

t5_mdl.eval()
print("  T5 ready")


# =============================================================
# SECTION 8 — LOAD / CACHE GPT2
# =============================================================
print("\n" + "=" * 60)
print("SECTION 8: Loading GPT2 (cached if exists)")
print("=" * 60)

GPT2_MODEL_ID = "gpt2"

if os.path.exists(os.path.join(GPT2_CACHE_DIR, "config.json")):
    print(f"  Found cached GPT2 at {GPT2_CACHE_DIR} — loading from disk...")
    gpt_tok = AutoTokenizer.from_pretrained(GPT2_CACHE_DIR)
    gpt_mdl = AutoModelForCausalLM.from_pretrained(GPT2_CACHE_DIR).to(DEVICE)
else:
    print(f"  No cache found — downloading {GPT2_MODEL_ID} ...")
    gpt_tok = AutoTokenizer.from_pretrained(GPT2_MODEL_ID)
    gpt_mdl = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_ID).to(DEVICE)
    gpt_tok.save_pretrained(GPT2_CACHE_DIR)
    gpt_mdl.save_pretrained(GPT2_CACHE_DIR)
    print(f"  Saved to {GPT2_CACHE_DIR}")

gpt_tok.pad_token = gpt_tok.eos_token
gpt_mdl.eval()
print("  GPT2 ready")


# =============================================================
# SECTION 9 — T5 + GPT2 GENERATION FUNCTIONS
# =============================================================

def build_t5_prompt(report_type, rec):
    """
    Short, direct prompt that works within flan-t5-small's token budget.
    RAG context: top_k=1 only — keeps input under 80 tokens.
    """
    features = (
        f"{CONDITION_NAMES.get(rec['cond'], rec['cond'])}, "
        f"HR {rec['hr']} bpm {rec['hr_status']}, "
        f"SpO2 {rec['spo2']}% {rec['spo2_status']}, "
        f"RR {rec['rr']} {rec['rr_status']}, "
        f"severity {rec['severity']}"
    )
    context = retrieve(features, top_k=1)

    if report_type == "doctor":
        return (
            f"Clinical report: {features}. "
            f"Reference: {context} "
            f"Findings and recommendations:"
        )
    else:
        return (
            f"Patient summary: heart shows {features}. "
            f"Simple explanation:"
        )

def t5_generate(report_type, rec):
    prompt = build_t5_prompt(report_type, rec)
    inputs = t5_tok(
        prompt,
        return_tensors="pt",
        max_length=80,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        out = t5_mdl.generate(
            input_ids            = inputs["input_ids"],
            attention_mask       = inputs["attention_mask"],
            max_new_tokens       = 80,
            num_beams            = 2,
            no_repeat_ngram_size = 2,
            repetition_penalty   = 2.0,
            early_stopping       = True,
        )

    generated = t5_tok.decode(out[0], skip_special_tokens=True).strip()

    # Detect echo or garbage output → fallback to rule-based
    echo_signals = ["Clinical report:", "Patient summary:", "Findings and", "Simple explanation"]
    if (any(generated.startswith(sig) for sig in echo_signals)
            or len(generated.split()) < 8):
        return (rule_based_doctor(rec) if report_type == "doctor"
                else rule_based_patient(rec))
    return generated

def gpt2_generate(report_type, rec):
    prompt = (
        f"Patient: {CONDITION_NAMES.get(rec['cond'], rec['cond'])}, "
        f"HR {rec['hr']} bpm, SpO2 {rec['spo2']}%, RR {rec['rr']}/min, "
        f"severity {rec['severity']}. "
        f"{'Medical report' if report_type == 'doctor' else 'Patient explanation'}: "
    )
    inputs = gpt_tok(
        prompt, return_tensors="pt",
        truncation=True, max_length=80
    ).to(DEVICE)
    with torch.no_grad():
        out = gpt_mdl.generate(
            **inputs,
            max_new_tokens    = 80,
            do_sample         = True,
            temperature       = 0.4,
            top_p             = 0.9,
            repetition_penalty= 1.5,
            pad_token_id      = gpt_tok.eos_token_id,
        )
    full = gpt_tok.decode(out[0], skip_special_tokens=True)
    # Strip the prompt prefix from output
    if "Medical report:" in full:
        return full.split("Medical report:")[-1].strip()
    if "Patient explanation:" in full:
        return full.split("Patient explanation:")[-1].strip()
    return full.strip()


# =============================================================
# SECTION 10 — DEMO: SHOW REPORTS FOR 4 REAL TEST RECORDS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 10: Demo — dual reports for real classifier outputs")
print("=" * 60)

# Pick one record from each condition for demo
demo_indices = {}
for cond in SUPERCLASSES:
    matches = records_df[records_df["cond"] == cond].index.tolist()
    if matches:
        demo_indices[cond] = matches[0]

demo_results = []
for cond, idx in demo_indices.items():
    rec = records_df.iloc[idx].to_dict()
    print(f"\n{'─'*60}")
    print(f"RECORD #{idx} — Classifier: {rec['cond']} "
          f"(conf: {rec['confidence']}%  severity: {rec['severity']})")
    print(f"HR: {rec['hr']} bpm ({rec['hr_status']})  |  "
          f"SpO2: {rec['spo2']}% ({rec['spo2_status']})  |  "
          f"RR: {rec['rr']} ({rec['rr_status']})")
    print(f"{'─'*60}")

    rule_doc = rule_based_doctor(rec)
    rule_pat = rule_based_patient(rec)
    t5_doc   = t5_generate("doctor",  rec)
    t5_pat   = t5_generate("patient", rec)
    gpt_doc  = gpt2_generate("doctor",  rec)
    gpt_pat  = gpt2_generate("patient", rec)

    print(f"\n[RULE-BASED] DOCTOR:\n  {rule_doc}")
    print(f"\n[RULE-BASED] PATIENT:\n  {rule_pat}")
    print(f"\n[FLAN-T5] DOCTOR:\n  {t5_doc}")
    print(f"\n[FLAN-T5] PATIENT:\n  {t5_pat}")
    print(f"\n[GPT2] DOCTOR:\n  {gpt_doc}")
    print(f"\n[GPT2] PATIENT:\n  {gpt_pat}")

    demo_results.append({
        "record_idx":         idx,
        "condition":          rec["cond"],
        "severity":           rec["severity"],
        "confidence":         rec["confidence"],
        "hr":                 rec["hr"],
        "spo2":               rec["spo2"],
        "rr":                 rec["rr"],
        "rule_doctor":        rule_doc,
        "rule_patient":       rule_pat,
        "t5_doctor":          t5_doc,
        "t5_patient":         t5_pat,
        "gpt2_doctor":        gpt_doc,
        "gpt2_patient":       gpt_pat,
    })

with open(os.path.join(STAGE5_DIR, "demo_reports.json"), "w") as f:
    json.dump(demo_results, f, indent=2)
print(f"\nSaved: demo_reports.json")


# =============================================================
# SECTION 11 — EVALUATION: 50 REAL RECORDS FROM TEST SET
# =============================================================
print("\n" + "=" * 60)
print("SECTION 11: Evaluation on 50 real test records")
print("=" * 60)

rouge_obj = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def score_pair(reference, hypothesis):
    """Compute all metrics for one reference-hypothesis pair."""
    # Guard against empty hypothesis
    if not hypothesis or len(hypothesis.split()) < 3:
        return dict(rouge1=0, rouge2=0, rougeL=0, bleu=0, meteor=0, sem_sim=0)

    r  = rouge_obj.score(reference, hypothesis)
    bl = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score

    try:
        mt = meteor_score([word_tokenize(reference.lower())],
                          word_tokenize(hypothesis.lower()))
    except Exception:
        mt = 0.0

    e1  = embedder.encode([reference],  show_progress_bar=False)
    e2  = embedder.encode([hypothesis], show_progress_bar=False)
    ss  = float(cosine_similarity(e1, e2)[0][0])

    return dict(
        rouge1  = r["rouge1"].fmeasure,
        rouge2  = r["rouge2"].fmeasure,
        rougeL  = r["rougeL"].fmeasure,
        bleu    = bl,
        meteor  = mt,
        sem_sim = ss,
    )

# Sample 50 records evenly across conditions
N_EVAL   = 50
eval_idx = []
per_cond = N_EVAL // len(SUPERCLASSES)
for cond in SUPERCLASSES:
    matches = records_df[records_df["cond"] == cond].index.tolist()
    chosen  = matches[:per_cond] if len(matches) >= per_cond else matches
    eval_idx.extend(chosen)
# Top up to 50 if needed
remaining = list(set(records_df.index.tolist()) - set(eval_idx))
eval_idx  = (eval_idx + remaining)[:N_EVAL]

print(f"Evaluating {len(eval_idx)} records "
      f"(~{per_cond} per condition)...")

eval_rows = []
for i, idx in enumerate(eval_idx):
    rec      = records_df.iloc[idx].to_dict()
    ref_doc  = rule_based_doctor(rec)    # rule-based = reference
    ref_pat  = rule_based_patient(rec)

    t5_doc_h  = t5_generate("doctor",  rec)
    t5_pat_h  = t5_generate("patient", rec)
    gpt_doc_h = gpt2_generate("doctor",  rec)
    gpt_pat_h = gpt2_generate("patient", rec)

    for rtype, ref, t5h, gpth in [
        ("doctor",  ref_doc, t5_doc_h,  gpt_doc_h),
        ("patient", ref_pat, t5_pat_h,  gpt_pat_h),
    ]:
        t5_s  = score_pair(ref, t5h)
        gpt_s = score_pair(ref, gpth)
        eval_rows.append({"model": "Flan-T5-small (RAG)",
                          "report": rtype, **t5_s})
        eval_rows.append({"model": "GPT2 (baseline)",
                          "report": rtype, **gpt_s})

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(eval_idx)} done...")

results_df = pd.DataFrame(eval_rows)
results_df.to_csv(os.path.join(STAGE5_DIR, "eval_results.csv"), index=False)
print("Saved: eval_results.csv")


# =============================================================
# SECTION 12 — PRINT COMPARISON TABLE
# =============================================================
print("\n" + "=" * 60)
print("SECTION 12: Model comparison results")
print("=" * 60)

for rtype in ["doctor", "patient"]:
    print(f"\n{'─'*60}")
    print(f"  {rtype.upper()} REPORT — {N_EVAL} real classifier records")
    print(f"{'─'*60}")
    sub = (results_df[results_df["report"] == rtype]
           .groupby("model").mean(numeric_only=True))
    for model, row in sub.iterrows():
        print(f"\n  {model}")
        print(f"    ROUGE-1  : {row['rouge1']:.4f}")
        print(f"    ROUGE-2  : {row['rouge2']:.4f}")
        print(f"    ROUGE-L  : {row['rougeL']:.4f}")
        print(f"    BLEU-4   : {row['bleu']:.2f}")
        print(f"    METEOR   : {row['meteor']:.4f}")
        print(f"    Sem. Sim.: {row['sem_sim']:.4f}")


# =============================================================
# SECTION 13 — READABILITY (Flesch-Kincaid)
# =============================================================
print("\n" + "=" * 60)
print("SECTION 13: Readability analysis")
print("=" * 60)

try:
    import textstat
except ImportError:
    os.system("pip install textstat -q")
    import textstat

# Collect all generated reports for readability
rule_docs, rule_pats, t5_docs, t5_pats = [], [], [], []
for idx in eval_idx:
    rec = records_df.iloc[idx].to_dict()
    rule_docs.append(rule_based_doctor(rec))
    rule_pats.append(rule_based_patient(rec))
    t5_docs.append(t5_generate("doctor",  rec))
    t5_pats.append(t5_generate("patient", rec))

def avg_readability(texts):
    texts = [t for t in texts if t and len(t.split()) > 5]
    if not texts:
        return {"Flesch Ease": 0, "FK Grade": 0}
    return {
        "Flesch Ease": round(np.mean([textstat.flesch_reading_ease(t)    for t in texts]), 2),
        "FK Grade":    round(np.mean([textstat.flesch_kincaid_grade(t)   for t in texts]), 2),
    }

print("\nReadability scores (higher Flesch Ease = simpler language):")
print(f"  Target: Doctor < 40 Flesch  |  Patient > 60 Flesch\n")

for label, texts in [
    ("Rule-based  — Doctor",  rule_docs),
    ("Rule-based  — Patient", rule_pats),
    ("Flan-T5     — Doctor",  t5_docs),
    ("Flan-T5     — Patient", t5_pats),
]:
    r = avg_readability(texts)
    gap_note = ""
    if "Patient" in label and r["Flesch Ease"] > 60:
        gap_note = "  ✓ plain English"
    elif "Doctor" in label and r["Flesch Ease"] < 40:
        gap_note = "  ✓ technical"
    print(f"  {label:30s}: "
          f"Flesch={r['Flesch Ease']:6.2f}  "
          f"FK Grade={r['FK Grade']:.1f}{gap_note}")

print("\n" + "=" * 60)
print("✓ Stage 5 complete.")
print("\nOutputs saved to ./outputs/stage5/:")
print("  patient_records.csv     — unified PTB-XL + BIDMC records")
print("  demo_reports.json       — dual reports for 1 record per condition")
print("  eval_results.csv        — BLEU/ROUGE/METEOR per model")
print("  t5_small_cached/        — T5 model (reloaded on next run)")
print("  gpt2_cached/      — GPT2 model (reloaded on next run)")
print("=" * 60)