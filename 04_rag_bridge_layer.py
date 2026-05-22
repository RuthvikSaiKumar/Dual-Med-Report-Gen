# =============================================================
# FAST DUAL MEDICAL REPORT GENERATION — No fine-tuning needed
# Runs in ~5-10 minutes on any GPU/CPU
#
# Strategy:
#   - Use T5-small with a STRONG structured prompt (no training)
#   - RAG context injected at inference time
#   - Rule-based fallback guarantees clean output every time
#   - Baseline: distilgpt2
#   - Full BLEU / ROUGE / METEOR / Semantic Similarity evaluation
# =============================================================

import os, random, warnings
import numpy as np
import pandas as pd
import torch
import nltk

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from rouge_score import rouge_scorer
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
print(f"Device: {DEVICE}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =============================================================
# RAG KNOWLEDGE BASE
# =============================================================

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

print("Loading RAG embedding model...")
embedder          = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
knowledge_embeds  = embedder.encode(KNOWLEDGE, show_progress_bar=False)

def retrieve(query, top_k=3):
    q_emb = embedder.encode([query], show_progress_bar=False)
    sims  = cosine_similarity(q_emb, knowledge_embeds)[0]
    idxs  = sims.argsort()[-top_k:][::-1]
    return " ".join(KNOWLEDGE[i] for i in idxs)

# =============================================================
# CONDITION METADATA
# =============================================================

CONDITION_NAMES = {
    "NORM": "normal sinus rhythm",
    "MI":   "myocardial infarction",
    "STTC": "ST/T-wave changes",
    "CD":   "conduction disturbance",
    "HYP":  "cardiac hypertrophy",
}

# =============================================================
# RULE-BASED REPORT ENGINE  (primary output — fast + reliable)
# =============================================================

def _hr_note(hr, hr_status):
    if hr_status == "high":   return f"elevated heart rate of {hr} bpm"
    if hr_status == "low":    return f"bradycardic rate of {hr} bpm"
    return f"heart rate of {hr} bpm within normal limits"

def _spo2_note(spo2, spo2_status):
    if spo2_status == "low":  return f"reduced oxygen saturation of {spo2}%"
    return f"oxygen saturation of {spo2}% within normal range"

def _rr_note(rr, rr_status):
    if rr_status == "high":   return f"tachypnoeic respiratory rate of {rr} breaths/min"
    if rr_status == "low":    return f"reduced respiratory rate of {rr} breaths/min"
    return f"respiratory rate of {rr} breaths/min within normal limits"

DOCTOR_ACTIONS = {
    "NORM": "Routine monitoring advised. No acute intervention required.",
    "MI":   ("Urgent troponin assay, continuous cardiac monitoring, and cardiology consultation "
             "are strongly indicated. Antiplatelet therapy and anticoagulation per protocol."),
    "STTC": ("Serial ECGs and cardiac biomarkers recommended. Cardiology review within 4 hours. "
             "Echocardiography to exclude structural cause."),
    "CD":   ("Electrolyte panel and medication review recommended. Cardiology evaluation warranted. "
             "Pacemaker assessment if haemodynamically unstable."),
    "HYP":  ("Echocardiography recommended to quantify chamber dimensions. "
             "Blood pressure assessment and hypertension management review indicated."),
}

PATIENT_ACTIONS = {
    "NORM": "Keep attending regular check-ups. No immediate action needed.",
    "MI":   ("Go to the emergency department now or call emergency services immediately. "
             "Do not drive yourself. Rest and stay calm."),
    "STTC": ("Please see a doctor today. Avoid strenuous activity until you are reviewed. "
             "Bring this report with you."),
    "CD":   ("Book an appointment with your doctor soon and bring this report. "
             "Tell them about all your current medications."),
    "HYP":  ("See your doctor soon. Focus on keeping your blood pressure under control. "
             "An ultrasound of your heart may be arranged."),
}

SEVERITY_PHRASES = {
    "doctor": {"mild": "mild", "moderate": "moderate clinical", "high": "high — urgent"},
    "patient": {"mild": "minor", "moderate": "moderate", "high": "serious — seek urgent help"},
}

def rule_based_doctor_report(cond, severity, hr, hr_status, spo2, spo2_status, rr, rr_status):
    cond_name  = CONDITION_NAMES.get(cond, cond)
    sev_phrase = SEVERITY_PHRASES["doctor"].get(severity, severity)
    hypoxia    = (" Hypoxaemia noted; supplemental oxygen and arterial blood gas analysis advised."
                  if spo2_status == "low" else "")
    return (
        f"ECG findings are consistent with {cond_name}. "
        f"The patient exhibits {_hr_note(hr, hr_status)}, "
        f"{_spo2_note(spo2, spo2_status)}, and {_rr_note(rr, rr_status)}.{hypoxia} "
        f"Overall severity is assessed as {sev_phrase}. "
        f"{DOCTOR_ACTIONS[cond]}"
    )

def rule_based_patient_report(cond, severity, hr, hr_status, spo2, spo2_status, rr, rr_status):
    cond_name  = CONDITION_NAMES.get(cond, cond)
    sev_phrase = SEVERITY_PHRASES["patient"].get(severity, severity)
    hypoxia    = (" Your oxygen level is lower than it should be — please tell the doctor immediately."
                  if spo2_status == "low" else "")
    hr_plain   = (f"Your heart is beating {'faster' if hr_status=='high' else 'slower'} than normal "
                  f"at {hr} beats per minute." if hr_status != "normal"
                  else f"Your heart rate of {hr} beats per minute is healthy.")
    return (
        f"Your heart test shows a pattern called {cond_name}. "
        f"{hr_plain} "
        f"Your oxygen level is {spo2}% and your breathing rate is {rr} per minute.{hypoxia} "
        f"The overall concern level is {sev_phrase}. "
        f"{PATIENT_ACTIONS[cond]}"
    )

# =============================================================
# LOAD T5-SMALL FOR NLP COMPARISON  (zero-shot, no training)
# =============================================================

T5_MODEL = "google/flan-t5-small"
print(f"\nLoading {T5_MODEL} (zero-shot)...")
t5_tok = AutoTokenizer.from_pretrained(T5_MODEL)
t5_mdl = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(DEVICE)
t5_mdl.eval()

def build_t5_prompt(report_type, cond, severity, hr, hr_status,
                    spo2, spo2_status, rr, rr_status):
    features = (
        f"ecg: {CONDITION_NAMES.get(cond, cond)}, "
        f"heart rate: {hr} bpm {hr_status}, "
        f"SpO2: {spo2}% {spo2_status}, "
        f"respiratory rate: {rr}/min {rr_status}, "
        f"severity: {severity}"
    )
    context = retrieve(features)
    if report_type == "doctor":
        instruction = (
            "Write a concise clinical medical report for a doctor. "
            "Include ECG findings, vital sign interpretation, severity, and recommended actions. "
            "Use medical terminology."
        )
    else:
        instruction = (
            "Write a simple health summary for a patient with no medical knowledge. "
            "Explain what the heart test shows, what the numbers mean, and what to do next. "
            "Use plain everyday language."
        )
    return (
        f"Medical context: {context}\n\n"
        f"Patient data: {features}\n\n"
        f"Task: {instruction}"
    )

def t5_generate(report_type, cond, severity, hr, hr_status,
                spo2, spo2_status, rr, rr_status):
    prompt = build_t5_prompt(
        report_type, cond, severity, hr, hr_status,
        spo2, spo2_status, rr, rr_status
    )
    inputs = t5_tok(
        prompt, return_tensors="pt",
        max_length=192, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        out = t5_mdl.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            early_stopping=True,
        )
    return t5_tok.decode(out[0], skip_special_tokens=True)

# =============================================================
# LOAD DISTILGPT2 BASELINE
# =============================================================

print("Loading distilgpt2 baseline...")
gpt_tok = AutoTokenizer.from_pretrained("distilgpt2")
gpt_mdl = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
gpt_tok.pad_token = gpt_tok.eos_token
gpt_mdl.eval()

def gpt2_generate(report_type, cond, severity, hr, hr_status,
                  spo2, spo2_status, rr, rr_status):
    prompt = (
        f"Patient has {CONDITION_NAMES.get(cond, cond)}, "
        f"HR {hr} bpm, SpO2 {spo2}%, RR {rr}/min, severity {severity}. "
        f"Medical report: "
    )
    inputs = gpt_tok(prompt, return_tensors="pt", truncation=True, max_length=100).to(DEVICE)
    with torch.no_grad():
        out = gpt_mdl.generate(
            **inputs, max_new_tokens=80,
            do_sample=True, temperature=0.4, top_p=0.9,
            pad_token_id=gpt_tok.eos_token_id,
        )
    return gpt_tok.decode(out[0], skip_special_tokens=True)

# =============================================================
# TEST CASES — DEMO OUTPUT
# =============================================================

test_cases = [
    dict(label="High-Risk MI + Hypoxia",
         cond="MI",   severity="high",
         hr=118, hr_status="high", spo2=88, spo2_status="low",
         rr=26,  rr_status="high"),

    dict(label="Normal ECG, All Vitals Normal",
         cond="NORM", severity="mild",
         hr=72,  hr_status="normal", spo2=98, spo2_status="normal",
         rr=15,  rr_status="normal"),

    dict(label="Conduction Disturbance + Bradycardia",
         cond="CD",   severity="moderate",
         hr=48,  hr_status="low", spo2=96, spo2_status="normal",
         rr=14,  rr_status="normal"),

    dict(label="Hypertrophy, Moderate",
         cond="HYP",  severity="moderate",
         hr=88,  hr_status="normal", spo2=95, spo2_status="normal",
         rr=18,  rr_status="normal"),
]

print("\n" + "=" * 65)
print("DEMO: DUAL REPORT GENERATION")
print("=" * 65)

for tc in test_cases:
    kw = {k: v for k, v in tc.items() if k != "label"}
    print(f"\n{'─'*65}")
    print(f"CASE: {tc['label']}")
    print(f"{'─'*65}")

    rule_doc = rule_based_doctor_report(**kw)
    rule_pat = rule_based_patient_report(**kw)
    t5_doc   = t5_generate("doctor",  **kw)
    t5_pat   = t5_generate("patient", **kw)

    print(f"\n[RULE-BASED] DOCTOR REPORT:\n  {rule_doc}")
    print(f"\n[RULE-BASED] PATIENT REPORT:\n  {rule_pat}")
    print(f"\n[FLAN-T5 ZERO-SHOT] DOCTOR:\n  {t5_doc}")
    print(f"\n[FLAN-T5 ZERO-SHOT] PATIENT:\n  {t5_pat}")

# =============================================================
# EVALUATION — Rule-based vs T5 vs GPT2  (50 random cases)
# =============================================================

print("\n" + "=" * 65)
print("EVALUATION  (n=50 random unseen cases)")
print("=" * 65)

CONDITIONS  = ["NORM", "MI", "STTC", "CD", "HYP"]
SEVERITIES  = ["mild", "moderate", "high"]
HR_POOL     = [(45,"low"),(55,"low"),(72,"normal"),(85,"normal"),(102,"high"),(118,"high")]
SPO2_POOL   = [(88,"low"),(92,"low"),(95,"normal"),(98,"normal"),(99,"normal")]
RR_POOL     = [(9,"low"),(14,"normal"),(17,"normal"),(22,"high"),(28,"high")]

def random_case():
    hr,   hr_s   = random.choice(HR_POOL)
    spo2, spo2_s = random.choice(SPO2_POOL)
    rr,   rr_s   = random.choice(RR_POOL)
    return dict(
        cond      = random.choice(CONDITIONS),
        severity  = random.choice(SEVERITIES),
        hr=hr, hr_status=hr_s,
        spo2=spo2, spo2_status=spo2_s,
        rr=rr,  rr_status=rr_s,
    )

rouge_obj = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
sem_model = embedder   # reuse — same model

def score_pair(reference, hypothesis):
    r  = rouge_obj.score(reference, hypothesis)
    bl = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score
    try:
        mt = meteor_score([word_tokenize(reference)], word_tokenize(hypothesis))
    except Exception:
        mt = 0.0
    e1 = sem_model.encode([reference],  show_progress_bar=False)
    e2 = sem_model.encode([hypothesis], show_progress_bar=False)
    ss = float(cosine_similarity(e1, e2)[0][0])
    return dict(
        rouge1=r["rouge1"].fmeasure,
        rouge2=r["rouge2"].fmeasure,
        rougeL=r["rougeL"].fmeasure,
        bleu=bl, meteor=mt, sem_sim=ss,
    )

N_EVAL = 50
print(f"Generating {N_EVAL} cases — this takes ~2-3 min...")

rows = []
for i in range(N_EVAL):
    kw        = random_case()
    ref_doc   = rule_based_doctor_report(**kw)   # rule-based = reference
    ref_pat   = rule_based_patient_report(**kw)

    t5_doc_h  = t5_generate("doctor",  **kw)
    t5_pat_h  = t5_generate("patient", **kw)
    gpt_doc_h = gpt2_generate("doctor",  **kw)
    gpt_pat_h = gpt2_generate("patient", **kw)

    for rtype, ref, t5h, gpth in [
        ("doctor",  ref_doc, t5_doc_h,  gpt_doc_h),
        ("patient", ref_pat, t5_pat_h,  gpt_pat_h),
    ]:
        t5_s  = score_pair(ref, t5h)
        gpt_s = score_pair(ref, gpth)
        rows.append({"model":"FLAN-T5 (zero-shot)", "report": rtype, **t5_s})
        rows.append({"model":"DistilGPT2 (baseline)", "report": rtype, **gpt_s})

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_EVAL} done...")

results_df = pd.DataFrame(rows)

for rtype in ["doctor", "patient"]:
    print(f"\n{'─'*65}")
    print(f"  {rtype.upper()} REPORT — averaged over {N_EVAL} cases")
    print(f"{'─'*65}")
    sub = results_df[results_df["report"] == rtype].groupby("model").mean(numeric_only=True)
    for model, row in sub.iterrows():
        print(f"\n  {model}")
        print(f"    ROUGE-1:   {row['rouge1']:.4f}")
        print(f"    ROUGE-2:   {row['rouge2']:.4f}")
        print(f"    ROUGE-L:   {row['rougeL']:.4f}")
        print(f"    BLEU-4:    {row['bleu']:.2f}")
        print(f"    METEOR:    {row['meteor']:.4f}")
        print(f"    Sem. Sim.: {row['sem_sim']:.4f}")

results_df.to_csv("eval_results.csv", index=False)
print("\nSaved: eval_results.csv")
print("\nDone.")