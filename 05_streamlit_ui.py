# import streamlit as st
# import pandas as pd
# import numpy as np
# import torch
# import textwrap
# import re

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM
# )

# # ============================================================
# # CONFIG
# # ============================================================

# MODEL_DIR = "trained_medical_model"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MAX_INPUT_LEN = 256
# MAX_NEW_TOKENS = 256

# # ============================================================
# # LOAD MODEL
# # ============================================================

# @st.cache_resource

# def load_model():

#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_DIR
#     )

#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         MODEL_DIR
#     )

#     model.to(DEVICE)

#     model.eval()

#     return tokenizer, model


# tokenizer, model = load_model()

# # ============================================================
# # SAMPLE ECG / SIGNAL DATASET
# # ============================================================

# sample_ecg_data = {

#     "Normal ECG": {
#         "ecg_class": "Normal sinus rhythm",
#         "arrhythmia_risk": "low"
#     },

#     "Mild Tachycardia": {
#         "ecg_class": "Sinus tachycardia",
#         "arrhythmia_risk": "moderate"
#     },

#     "Atrial Fibrillation": {
#         "ecg_class": "Atrial fibrillation",
#         "arrhythmia_risk": "high"
#     },

#     "Bradycardia": {
#         "ecg_class": "Sinus bradycardia",
#         "arrhythmia_risk": "moderate"
#     },

#     "Ventricular Tachycardia": {
#         "ecg_class": "Ventricular tachycardia",
#         "arrhythmia_risk": "critical"
#     }
# }

# # ============================================================
# # PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="AI Medical Report Generator",
#     page_icon="🩺",
#     layout="wide"
# )

# # ============================================================
# # STYLING
# # ============================================================

# st.markdown(
#     """
#     <style>

#     .main {
#         background-color: #0f172a;
#         color: white;
#     }

#     .stApp {
#         background-color: #0f172a;
#     }

#     .metric-card {
#         background-color: #1e293b;
#         padding: 20px;
#         border-radius: 16px;
#         text-align: center;
#         border: 1px solid #334155;
#     }

#     .section-card {
#         background-color: #1e293b;
#         padding: 20px;
#         border-radius: 18px;
#         margin-bottom: 20px;
#         border: 1px solid #334155;
#     }

#     .severity-low {
#         color: #22c55e;
#         font-weight: bold;
#     }

#     .severity-moderate {
#         color: #facc15;
#         font-weight: bold;
#     }

#     .severity-high {
#         color: #f97316;
#         font-weight: bold;
#     }

#     .severity-critical {
#         color: #ef4444;
#         font-weight: bold;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # ============================================================
# # TITLE
# # ============================================================

# st.title("🩺 AI Medical Report Generator")

# st.markdown(
#     "Generate structured doctor and patient medical reports using a fine-tuned FLAN-T5 model."
# )

# # ============================================================
# # SIDEBAR INPUTS
# # ============================================================

# st.sidebar.header("Patient Input Parameters")

# report_type = st.sidebar.selectbox(
#     "Report Type",
#     ["doctor", "patient"]
# )

# selected_ecg = st.sidebar.selectbox(
#     "Select ECG / Sequence Data",
#     list(sample_ecg_data.keys())
# )

# heart_rate = st.sidebar.number_input(
#     "Heart Rate (bpm)",
#     min_value=30,
#     max_value=220,
#     value=102
# )

# spo2 = st.sidebar.number_input(
#     "SpO2 (%)",
#     min_value=50,
#     max_value=100,
#     value=94
# )

# resp_rate = st.sidebar.number_input(
#     "Respiratory Rate (/min)",
#     min_value=5,
#     max_value=50,
#     value=22
# )

# stress_level = st.sidebar.selectbox(
#     "Stress Level",
#     ["low", "moderate", "high"]
# )

# blood_pressure = st.sidebar.text_input(
#     "Blood Pressure",
#     value="130/90"
# )

# body_temperature = st.sidebar.number_input(
#     "Temperature (°C)",
#     min_value=30.0,
#     max_value=45.0,
#     value=37.2
# )

# # ============================================================
# # ECG DATA
# # ============================================================

# selected_ecg_data = sample_ecg_data[selected_ecg]

# arrhythmia_risk = selected_ecg_data["arrhythmia_risk"]

# ecg_class = selected_ecg_data["ecg_class"]

# # ============================================================
# # RULE ENGINE
# # ============================================================

# severity = "Low"

# if spo2 < 90 or heart_rate > 140:
#     severity = "Critical"

# elif spo2 < 94 or heart_rate > 115:
#     severity = "High"

# elif heart_rate > 100:
#     severity = "Moderate"

# recommendations = []

# if heart_rate > 100:
#     recommendations.append("Continuous cardiac monitoring")

# if spo2 < 94:
#     recommendations.append("Supplemental oxygen support")

# if stress_level == "high":
#     recommendations.append("Stress reduction and cardiovascular evaluation")

# if len(recommendations) == 0:
#     recommendations.append("Routine monitoring advised")

# # ============================================================
# # BUILD PROMPT
# # ============================================================


# def build_prompt():

#     recommendation_text = "\n".join(
#         [f"- {r}" for r in recommendations]
#     )

#     prompt = f"""
#         Generate ONLY a structured medical report.

#         STRICT FORMAT:

#         SEVERITY:
#         <severity>

#         SUMMARY:
#         <summary>

#         WHAT_IT_MEANS:
#         <meaning>

#         WHAT_TO_DO:
#         - item 1
#         - item 2

#         EMERGENCY:
#         <emergency advice>

#         PATIENT DATA:

#         Heart Rate: {heart_rate}
#         SpO2: {spo2}
#         Respiratory Rate: {resp_rate}
#         ECG: {ecg_class}
#         Stress Level: {stress_level}
#         """

#     return textwrap.dedent(prompt)

# # ============================================================
# # GENERATE REPORT
# # ============================================================


# def generate_report(prompt):

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_INPUT_LEN
#     ).to(DEVICE)

#     with torch.no_grad():

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,
#             temperature=0.3,
#             num_beams=4,
#             repetition_penalty=1.2
#         )

#     decoded = tokenizer.decode(
#         outputs[0],
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True
#     )

#     decoded = decoded.replace("SEVERITY:", "\nSEVERITY:")
#     decoded = decoded.replace("SUMMARY:", "\nSUMMARY:")
#     decoded = decoded.replace("WHAT_IT_MEANS:", "\nWHAT_IT_MEANS:")
#     decoded = decoded.replace("WHAT_TO_DO:", "\nWHAT_TO_DO:")
#     decoded = decoded.replace("EMERGENCY:", "\nEMERGENCY:")
#     decoded = decoded.replace("ECG_FINDINGS:", "\nECG_FINDINGS:")
#     decoded = decoded.replace(
#         "CLINICAL_INTERPRETATION:",
#         "\nCLINICAL_INTERPRETATION:"
#     )
#     decoded = decoded.replace(
#         "RECOMMENDED_ACTIONS:",
#         "\nRECOMMENDED_ACTIONS:"
#     )

#     return decoded.strip()

# # ============================================================
# # PARSE REPORT
# # ============================================================


# def extract_section(text, section):

#     if text is None:
#         return "Not Available"

#     if not isinstance(text, str):
#         text = str(text)

#     pattern = rf"{section}:(.*?)(?=\\n[A-Z_]+:|$)"

#     match = re.search(
#         pattern,
#         text,
#         re.DOTALL
#     )

#     if match:
#         return match.group(1).strip()

#     return "Not Available"

# # ============================================================
# # GENERATE BUTTON
# # ============================================================

# if st.button("Generate Medical Report"):

#     with st.spinner("Generating AI medical report..."):

#         prompt = build_prompt()

#         generated_report = generate_report(prompt)

#         st.write(generated_report)

#     # ========================================================
#     # TOP METRICS
#     # ========================================================

#     st.subheader("Patient Overview")

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         st.markdown(
#             f"""
#             <div class='metric-card'>
#             <h4>Heart Rate</h4>
#             <h2>{heart_rate} bpm</h2>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     with col2:
#         st.markdown(
#             f"""
#             <div class='metric-card'>
#             <h4>SpO2</h4>
#             <h2>{spo2}%</h2>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     with col3:
#         st.markdown(
#             f"""
#             <div class='metric-card'>
#             <h4>ECG</h4>
#             <h2>{ecg_class}</h2>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     with col4:

#         severity_class = "severity-low"

#         if severity.lower() == "moderate":
#             severity_class = "severity-moderate"

#         elif severity.lower() == "high":
#             severity_class = "severity-high"

#         elif severity.lower() == "critical":
#             severity_class = "severity-critical"

#         st.markdown(
#             f"""
#             <div class='metric-card'>
#             <h4>Severity</h4>
#             <h2 class='{severity_class}'>{severity}</h2>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     st.markdown("---")

#     # ========================================================
#     # REPORT DISPLAY
#     # ========================================================

#     # ========================================================
#     # STRUCTURED REPORT UI
#     # ========================================================

#     severity_text = extract_section(
#         generated_report,
#         "SEVERITY"
#     )

#     summary = extract_section(
#         generated_report,
#         "SUMMARY"
#     )

#     meaning = extract_section(
#         generated_report,
#         "WHAT_IT_MEANS"
#     )

#     todo = extract_section(
#         generated_report,
#         "WHAT_TO_DO"
#     )

#     emergency = extract_section(
#         generated_report,
#         "EMERGENCY"
#     )

#     ecg_findings = extract_section(
#         generated_report,
#         "ECG_FINDINGS"
#     )

#     clinical = extract_section(
#         generated_report,
#         "CLINICAL_INTERPRETATION"
#     )

#     actions = extract_section(
#         generated_report,
#         "RECOMMENDED_ACTIONS"
#     )

#     st.subheader("Generated Medical Report")

#     # ========================================================
#     # SUMMARY CARD
#     # ========================================================

#     st.markdown(
#         f"""
#         <div class='section-card'>
#         <h2>Summary</h2>
#         <p>{summary}</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # ========================================================
#     # MEANING CARD
#     # ========================================================

#     st.markdown(
#         f"""
#         <div class='section-card'>
#         <h2>Clinical Interpretation</h2>
#         <p>{meaning}</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # ========================================================
#     # ECG CARD
#     # ========================================================

#     if ecg_findings != "Not Available":

#         st.markdown(
#             f"""
#             <div class='section-card'>
#             <h2>ECG Findings</h2>
#             <p>{ecg_findings}</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     # ========================================================
#     # ACTIONS CARD
#     # ========================================================

#     if actions != "Not Available":

#         actions_html = actions.replace("-", "•")

#         st.markdown(
#             f"""
#             <div class='section-card'>
#             <h2>Recommended Actions</h2>
#             <p>{actions_html}</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     # ========================================================
#     # WHAT TO DO
#     # ========================================================

#     todo_html = todo.replace("-", "•")

#     st.markdown(
#         f"""
#         <div class='section-card'>
#         <h2>What To Do</h2>
#         <p>{todo_html}</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # ========================================================
#     # EMERGENCY
#     # ========================================================

#     st.markdown(
#         f"""
#         <div class='section-card'>
#         <h2>Emergency Advice</h2>
#         <p>{emergency}</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # ========================================================
#     # RAW OUTPUT
#     # ========================================================

#     with st.expander("View Raw Model Output"):
#         st.text(generated_report)

#     with st.expander("View Prompt Sent To Model"):
#         st.text(prompt)

# # ============================================================
# # FOOTER
# # ============================================================

# st.markdown("---")

# st.caption(
#     "AI-assisted medical report generation system using fine-tuned FLAN-T5-small"
# )



import streamlit as st
import pandas as pd
import numpy as np
import torch
import textwrap
import re
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG & DEVICE
# ============================================================
T5_MODEL_ID = "google/flan-t5-small"
EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_LEN = 256
MAX_NEW_TOKENS = 128

# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================
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
]

# ============================================================
# CACHED MODEL LOADING
# ============================================================
@st.cache_resource
def load_models():
    # Load T5
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_ID).to(DEVICE)
    model.eval()
    
    # Load Sentence Transformer for RAG
    embedder = SentenceTransformer(EMBEDDER_ID)
    knowledge_embeds = embedder.encode(KNOWLEDGE, show_progress_bar=False)
    
    return tokenizer, model, embedder, knowledge_embeds

tokenizer, model, embedder, knowledge_embeds = load_models()

def retrieve_context(query, top_k=1):
    q_emb = embedder.encode([query], show_progress_bar=False)
    sims = cosine_similarity(q_emb, knowledge_embeds)[0]
    idxs = sims.argsort()[-top_k:][::-1]
    return " ".join(KNOWLEDGE[i] for i in idxs)

# ============================================================
# DATA DICTIONARIES
# ============================================================
sample_ecg_data = {
    "Normal ECG": {"ecg_class": "normal sinus rhythm", "cond": "NORM"},
    "Mild Tachycardia": {"ecg_class": "sinus tachycardia", "cond": "NORM"},
    "Atrial Fibrillation": {"ecg_class": "atrial fibrillation", "cond": "CD"},
    "Bradycardia": {"ecg_class": "sinus bradycardia", "cond": "CD"},
    "Myocardial Infarction": {"ecg_class": "myocardial infarction", "cond": "MI"},
    "ST/T-Wave Changes": {"ecg_class": "ST/T-wave changes", "cond": "STTC"}
}

DOCTOR_ACTIONS = {
    "NORM": "- Routine monitoring advised.\n- No acute intervention required.",
    "MI": "- Urgent troponin assay.\n- Continuous cardiac monitoring.\n- Cardiology consultation strongly indicated.\n- Antiplatelet therapy per protocol.",
    "STTC": "- Serial ECGs and cardiac biomarkers recommended.\n- Cardiology review within 4 hours.\n- Echocardiography to exclude structural cause.",
    "CD": "- Electrolyte panel and medication review.\n- Cardiology evaluation warranted.\n- Pacemaker assessment if haemodynamically unstable.",
}

PATIENT_ACTIONS = {
    "NORM": "- Keep attending regular check-ups.\n- Maintain a healthy lifestyle.",
    "MI": "- Go to the emergency department now.\n- Call emergency services immediately.\n- Do not drive yourself.\n- Rest and stay calm.",
    "STTC": "- Please see a doctor today.\n- Avoid strenuous activity until reviewed.\n- Bring this report with you.",
    "CD": "- Book an appointment with your doctor soon.\n- Tell them about all your current medications.",
}

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(page_title="AI Medical Report Generator", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0f172a; color: white; }
    .stApp { background-color: #0f172a; }
    .metric-card {
        background-color: #1e293b; padding: 20px; border-radius: 16px; 
        text-align: center; border: 1px solid #334155;
    }
    .section-card {
        background-color: #1e293b; padding: 20px; border-radius: 18px; 
        margin-bottom: 20px; border: 1px solid #334155;
    }
    .severity-low { color: #22c55e; font-weight: bold; }
    .severity-moderate { color: #facc15; font-weight: bold; }
    .severity-high { color: #f97316; font-weight: bold; }
    .severity-critical { color: #ef4444; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 AI Medical Report Generator")
st.markdown("Generate structured doctor and patient medical reports using a fine-tuned FLAN-T5 model and RAG.")

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("Patient Input Parameters")

report_type = st.sidebar.selectbox("Report Type", ["doctor", "patient"])
selected_ecg = st.sidebar.selectbox("Select ECG / Sequence Data", list(sample_ecg_data.keys()))
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=102)
spo2 = st.sidebar.number_input("SpO2 (%)", min_value=50, max_value=100, value=94)
resp_rate = st.sidebar.number_input("Respiratory Rate (/min)", min_value=5, max_value=50, value=22)
stress_level = st.sidebar.selectbox("Stress Level", ["low", "moderate", "high"])

ecg_info = sample_ecg_data[selected_ecg]
ecg_class = ecg_info["ecg_class"]
cond_code = ecg_info["cond"]

# ============================================================
# RULE ENGINE FOR METADATA
# ============================================================
severity = "Low"
if spo2 < 90 or heart_rate > 140 or cond_code == "MI":
    severity = "Critical"
elif spo2 < 94 or heart_rate > 115 or cond_code == "STTC":
    severity = "High"
elif heart_rate > 100 or heart_rate < 60 or cond_code == "CD":
    severity = "Moderate"

emergency_advice = "None"
if severity in ["Critical", "High"]:
    emergency_advice = "Seek immediate emergency medical assistance. Do not drive yourself to the hospital."
elif severity == "Moderate":
    emergency_advice = "Schedule a consultation with a healthcare provider soon. Monitor symptoms."

# ============================================================
# AI GENERATION LOGIC
# ============================================================
def generate_t5_summary(report_type):
    """Uses RAG + T5 to generate the narrative summary."""
    features = f"{ecg_class}, HR {heart_rate} bpm, SpO2 {spo2}%, RR {resp_rate}/min, severity {severity}."
    context = retrieve_context(features, top_k=1)
    
    if report_type == "doctor":
        prompt = f"Clinical report: {features}. Reference: {context} Findings and recommendations:"
    else:
        prompt = f"Patient summary: heart shows {features}. Simple explanation:"
        
    inputs = tokenizer(prompt, return_tensors="pt", max_length=80, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=80,
            num_beams=2,
            no_repeat_ngram_size=2,
            repetition_penalty=2.0,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def build_structured_report(ai_summary):
    """Assembles the UI-compatible structured text string."""
    actions = DOCTOR_ACTIONS.get(cond_code, DOCTOR_ACTIONS["NORM"]) if report_type == "doctor" else PATIENT_ACTIONS.get(cond_code, PATIENT_ACTIONS["NORM"])
    
    # Map interpretations based on RAG context
    meaning = retrieve_context(ecg_class, top_k=1)
    
    structured_text = f"""SEVERITY:
{severity.upper()}

SUMMARY:
{ai_summary}

WHAT_IT_MEANS:
{meaning}

WHAT_TO_DO:
{actions}

EMERGENCY:
{emergency_advice}

ECG_FINDINGS:
Primary classification: {ecg_class.title()}. Condition code: {cond_code}.
"""
    return structured_text

def extract_section(text, section):
    if text is None: return "Not Available"
    pattern = rf"{section}:(.*?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, str(text), re.DOTALL)
    if match: return match.group(1).strip()
    return "Not Available"

# ============================================================
# UI RENDERING & TRIGGER
# ============================================================
if st.button("Generate Medical Report"):
    with st.spinner(f"Generating AI {report_type} report using Flan-T5 & RAG..."):
        
        # 1. Generate Narrative with T5
        ai_narrative = generate_t5_summary(report_type)
        
        # 2. Build Structured Output for the Regex Parser
        generated_report = build_structured_report(ai_narrative)

    # Top Metrics
    st.subheader("Patient Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"<div class='metric-card'><h4>Heart Rate</h4><h2>{heart_rate} bpm</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>SpO2</h4><h2>{spo2}%</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>ECG</h4><h2>{ecg_class.title()}</h2></div>", unsafe_allow_html=True)
    with col4:
        sev_class = f"severity-{severity.lower()}"
        st.markdown(f"<div class='metric-card'><h4>Severity</h4><h2 class='{sev_class}'>{severity}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"Generated Medical Report ({report_type.title()})")

    # Extract Sections
    summary = extract_section(generated_report, "SUMMARY")
    meaning = extract_section(generated_report, "WHAT_IT_MEANS")
    todo = extract_section(generated_report, "WHAT_TO_DO")
    emergency = extract_section(generated_report, "EMERGENCY")
    ecg_findings = extract_section(generated_report, "ECG_FINDINGS")

    # Render Cards
    st.markdown(f"<div class='section-card'><h2>Summary</h2><p>{summary}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-card'><h2>Clinical Interpretation</h2><p>{meaning}</p></div>", unsafe_allow_html=True)
    
    if ecg_findings != "Not Available":
        st.markdown(f"<div class='section-card'><h2>ECG Findings</h2><p>{ecg_findings}</p></div>", unsafe_allow_html=True)
        
    todo_html = todo.replace("-", "•").replace("\n", "<br>")
    st.markdown(f"<div class='section-card'><h2>What To Do / Recommended Actions</h2><p>{todo_html}</p></div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='section-card'><h2>Emergency Advice</h2><p>{emergency}</p></div>", unsafe_allow_html=True)

    # Debug Expanders
    with st.expander("View Raw Model Output String"):
        st.text(generated_report)

st.markdown("---")
st.caption("AI-assisted medical report generation system using fine-tuned FLAN-T5-small + RAG Knowledge Base")