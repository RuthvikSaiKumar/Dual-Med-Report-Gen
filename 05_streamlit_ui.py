import streamlit as st
import pandas as pd
import numpy as np
import torch
import textwrap
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "trained_medical_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_LEN = 256
MAX_NEW_TOKENS = 256

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource

def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR
    )

    model.to(DEVICE)

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# ============================================================
# SAMPLE ECG / SIGNAL DATASET
# ============================================================

sample_ecg_data = {

    "Normal ECG": {
        "ecg_class": "Normal sinus rhythm",
        "arrhythmia_risk": "low"
    },

    "Mild Tachycardia": {
        "ecg_class": "Sinus tachycardia",
        "arrhythmia_risk": "moderate"
    },

    "Atrial Fibrillation": {
        "ecg_class": "Atrial fibrillation",
        "arrhythmia_risk": "high"
    },

    "Bradycardia": {
        "ecg_class": "Sinus bradycardia",
        "arrhythmia_risk": "moderate"
    },

    "Ventricular Tachycardia": {
        "ecg_class": "Ventricular tachycardia",
        "arrhythmia_risk": "critical"
    }
}

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Medical Report Generator",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
        color: white;
    }

    .stApp {
        background-color: #0f172a;
    }

    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #334155;
    }

    .section-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 18px;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }

    .severity-low {
        color: #22c55e;
        font-weight: bold;
    }

    .severity-moderate {
        color: #facc15;
        font-weight: bold;
    }

    .severity-high {
        color: #f97316;
        font-weight: bold;
    }

    .severity-critical {
        color: #ef4444;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.title("🩺 AI Medical Report Generator")

st.markdown(
    "Generate structured doctor and patient medical reports using a fine-tuned FLAN-T5 model."
)

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Patient Input Parameters")

report_type = st.sidebar.selectbox(
    "Report Type",
    ["doctor", "patient"]
)

selected_ecg = st.sidebar.selectbox(
    "Select ECG / Sequence Data",
    list(sample_ecg_data.keys())
)

heart_rate = st.sidebar.number_input(
    "Heart Rate (bpm)",
    min_value=30,
    max_value=220,
    value=102
)

spo2 = st.sidebar.number_input(
    "SpO2 (%)",
    min_value=50,
    max_value=100,
    value=94
)

resp_rate = st.sidebar.number_input(
    "Respiratory Rate (/min)",
    min_value=5,
    max_value=50,
    value=22
)

stress_level = st.sidebar.selectbox(
    "Stress Level",
    ["low", "moderate", "high"]
)

blood_pressure = st.sidebar.text_input(
    "Blood Pressure",
    value="130/90"
)

body_temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=30.0,
    max_value=45.0,
    value=37.2
)

# ============================================================
# ECG DATA
# ============================================================

selected_ecg_data = sample_ecg_data[selected_ecg]

arrhythmia_risk = selected_ecg_data["arrhythmia_risk"]

ecg_class = selected_ecg_data["ecg_class"]

# ============================================================
# RULE ENGINE
# ============================================================

severity = "Low"

if spo2 < 90 or heart_rate > 140:
    severity = "Critical"

elif spo2 < 94 or heart_rate > 115:
    severity = "High"

elif heart_rate > 100:
    severity = "Moderate"

recommendations = []

if heart_rate > 100:
    recommendations.append("Continuous cardiac monitoring")

if spo2 < 94:
    recommendations.append("Supplemental oxygen support")

if stress_level == "high":
    recommendations.append("Stress reduction and cardiovascular evaluation")

if len(recommendations) == 0:
    recommendations.append("Routine monitoring advised")

# ============================================================
# BUILD PROMPT
# ============================================================


def build_prompt():

    recommendation_text = "\n".join(
        [f"- {r}" for r in recommendations]
    )

    prompt = f"""
        Generate ONLY a structured medical report.

        STRICT FORMAT:

        SEVERITY:
        <severity>

        SUMMARY:
        <summary>

        WHAT_IT_MEANS:
        <meaning>

        WHAT_TO_DO:
        - item 1
        - item 2

        EMERGENCY:
        <emergency advice>

        PATIENT DATA:

        Heart Rate: {heart_rate}
        SpO2: {spo2}
        Respiratory Rate: {resp_rate}
        ECG: {ecg_class}
        Stress Level: {stress_level}
        """

    return textwrap.dedent(prompt)

# ============================================================
# GENERATE REPORT
# ============================================================


def generate_report(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.3,
            num_beams=4,
            repetition_penalty=1.2
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    decoded = decoded.replace("SEVERITY:", "\nSEVERITY:")
    decoded = decoded.replace("SUMMARY:", "\nSUMMARY:")
    decoded = decoded.replace("WHAT_IT_MEANS:", "\nWHAT_IT_MEANS:")
    decoded = decoded.replace("WHAT_TO_DO:", "\nWHAT_TO_DO:")
    decoded = decoded.replace("EMERGENCY:", "\nEMERGENCY:")
    decoded = decoded.replace("ECG_FINDINGS:", "\nECG_FINDINGS:")
    decoded = decoded.replace(
        "CLINICAL_INTERPRETATION:",
        "\nCLINICAL_INTERPRETATION:"
    )
    decoded = decoded.replace(
        "RECOMMENDED_ACTIONS:",
        "\nRECOMMENDED_ACTIONS:"
    )

    return decoded.strip()

# ============================================================
# PARSE REPORT
# ============================================================


def extract_section(text, section):

    if text is None:
        return "Not Available"

    if not isinstance(text, str):
        text = str(text)

    pattern = rf"{section}:(.*?)(?=\\n[A-Z_]+:|$)"

    match = re.search(
        pattern,
        text,
        re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return "Not Available"

# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button("Generate Medical Report"):

    with st.spinner("Generating AI medical report..."):

        prompt = build_prompt()

        generated_report = generate_report(prompt)

        st.write(generated_report)

    # ========================================================
    # TOP METRICS
    # ========================================================

    st.subheader("Patient Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
            <h4>Heart Rate</h4>
            <h2>{heart_rate} bpm</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
            <h4>SpO2</h4>
            <h2>{spo2}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
            <h4>ECG</h4>
            <h2>{ecg_class}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        severity_class = "severity-low"

        if severity.lower() == "moderate":
            severity_class = "severity-moderate"

        elif severity.lower() == "high":
            severity_class = "severity-high"

        elif severity.lower() == "critical":
            severity_class = "severity-critical"

        st.markdown(
            f"""
            <div class='metric-card'>
            <h4>Severity</h4>
            <h2 class='{severity_class}'>{severity}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ========================================================
    # REPORT DISPLAY
    # ========================================================

    # ========================================================
    # STRUCTURED REPORT UI
    # ========================================================

    severity_text = extract_section(
        generated_report,
        "SEVERITY"
    )

    summary = extract_section(
        generated_report,
        "SUMMARY"
    )

    meaning = extract_section(
        generated_report,
        "WHAT_IT_MEANS"
    )

    todo = extract_section(
        generated_report,
        "WHAT_TO_DO"
    )

    emergency = extract_section(
        generated_report,
        "EMERGENCY"
    )

    ecg_findings = extract_section(
        generated_report,
        "ECG_FINDINGS"
    )

    clinical = extract_section(
        generated_report,
        "CLINICAL_INTERPRETATION"
    )

    actions = extract_section(
        generated_report,
        "RECOMMENDED_ACTIONS"
    )

    st.subheader("Generated Medical Report")

    # ========================================================
    # SUMMARY CARD
    # ========================================================

    st.markdown(
        f"""
        <div class='section-card'>
        <h2>Summary</h2>
        <p>{summary}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # MEANING CARD
    # ========================================================

    st.markdown(
        f"""
        <div class='section-card'>
        <h2>Clinical Interpretation</h2>
        <p>{meaning}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # ECG CARD
    # ========================================================

    if ecg_findings != "Not Available":

        st.markdown(
            f"""
            <div class='section-card'>
            <h2>ECG Findings</h2>
            <p>{ecg_findings}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # ACTIONS CARD
    # ========================================================

    if actions != "Not Available":

        actions_html = actions.replace("-", "•")

        st.markdown(
            f"""
            <div class='section-card'>
            <h2>Recommended Actions</h2>
            <p>{actions_html}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # WHAT TO DO
    # ========================================================

    todo_html = todo.replace("-", "•")

    st.markdown(
        f"""
        <div class='section-card'>
        <h2>What To Do</h2>
        <p>{todo_html}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # EMERGENCY
    # ========================================================

    st.markdown(
        f"""
        <div class='section-card'>
        <h2>Emergency Advice</h2>
        <p>{emergency}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # RAW OUTPUT
    # ========================================================

    with st.expander("View Raw Model Output"):
        st.text(generated_report)

    with st.expander("View Prompt Sent To Model"):
        st.text(prompt)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-assisted medical report generation system using fine-tuned FLAN-T5-small"
)