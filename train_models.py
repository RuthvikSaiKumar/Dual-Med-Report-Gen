import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# =====================================================================
# 1. KNOWLEDGE BASE & DETAILED CLINICAL MAPPING
# =====================================================================

CONDITION_NAMES = {
    "NORM": "Normal sinus rhythm",
    "MI":   "Myocardial infarction",
    "STTC": "ST/T-wave changes",
    "CD":   "Conduction disturbance",
    "HYP":  "Cardiac hypertrophy",
}

# Detailed mapping to fill out the UI's exact ECG and Differential sections
ECG_DETAILS = {
    "NORM": {
        "rhythm": "Regular — sinus rhythm",
        "conduction": "Normal AV and intraventricular conduction",
        "morphology": "Normal P waves, QRS complexes, and T waves",
        "diff_dx": ["Normal physiological state", "Athlete's heart (if resting bradycardia)"]
    },
    "MI": {
        "rhythm": "Regular or irregular depending on ectopic activity",
        "conduction": "Possible block secondary to ischemia",
        "morphology": "Pathological Q waves, ST-segment elevation, T-wave inversion",
        "diff_dx": ["Acute coronary syndrome (primary)", "Acute pericarditis", "Myocarditis", "Pulmonary embolism"]
    },
    "STTC": {
        "rhythm": "Often regular, but variable",
        "conduction": "Usually intact, though repolarization is altered",
        "morphology": "ST-segment depression, flat or inverted T waves",
        "diff_dx": ["Myocardial ischemia (primary)", "Electrolyte imbalance (e.g., hypokalemia)", "Ventricular strain", "Drug effect"]
    },
    "CD": {
        "rhythm": "Underlying sinus rhythm, possible pauses",
        "conduction": "Bundle branch block or Atrioventricular (AV) block detected",
        "morphology": "Prolonged PR interval or widened QRS complex (>120ms)",
        "diff_dx": ["Degenerative conduction system disease (primary)", "Ischemic heart disease", "Electrolyte disturbance"]
    },
    "HYP": {
        "rhythm": "Regular — sinus rhythm",
        "conduction": "Intact, minor delays possible due to muscle mass",
        "morphology": "High-voltage QRS complexes, secondary ST-T changes",
        "diff_dx": ["Left ventricular hypertrophy (primary)", "Right ventricular hypertrophy", "Hypertrophic cardiomyopathy"]
    }
}

DOCTOR_ACTIONS = {
    "NORM": "* Routine monitoring advised\n* No acute intervention required",
    "MI":   "* Urgent troponin, BNP, and D-dimer\n* Continuous cardiac monitoring\n* Urgent cardiology consultation\n* Antiplatelet therapy per protocol",
    "STTC": "* Serial ECGs and cardiac biomarkers\n* Cardiology review within 4 hours\n* Echocardiography to exclude structural cause",
    "CD":   "* Electrolyte panel and medication review\n* Cardiology evaluation warranted\n* Pacemaker assessment if haemodynamically unstable",
    "HYP":  "* Echocardiography to quantify chamber dimensions\n* Blood pressure assessment\n* Hypertension management review",
}

PATIENT_ACTIONS = {
    "NORM": "* Keep attending regular check-ups.\n* No immediate action needed.",
    "MI":   "* We are transferring you for urgent cardiac care.\n* Please remain resting in bed.\n* We are administering medications to help your heart.",
    "STTC": "* We need to keep you on a heart monitor.\n* We will run some standard blood tests to check your heart health.\n* Please rest and let the care team monitor you.",
    "CD":   "* We will monitor your heart rhythm closely.\n* We need to review all medications you take at home.\n* A heart specialist may come to see you.",
    "HYP":  "* We recommend scheduling an ultrasound of your heart.\n* Focus on keeping your blood pressure under control.\n* Follow up with your primary doctor soon.",
}

# =====================================================================
# 2. SYNTHETIC DATASET GENERATION (RESTORED EXACT FORMAT)
# =====================================================================
def generate_training_data(num_samples=1000):
    data = {"prompt": [], "completion": []}
    
    for i in range(num_samples):
        audience = "Doctor" if i % 2 == 0 else "Patient"
        
        # Randomize patient physiology
        cond = random.choice(list(CONDITION_NAMES.keys()))
        hr = random.randint(45, 130)
        spo2 = random.randint(85, 100)
        rr = random.randint(10, 30)
        
        # Determine statuses based on rules
        hr_status = "Elevated" if hr > 100 else "Bradycardic" if hr < 60 else "Normal"
        spo2_status = "Below normal" if spo2 < 94 else "Normal"
        rr_status = "Tachypnea" if rr > 20 else "Bradypnea" if rr < 12 else "Normal"
        
        # Format rate classification for ECG Findings
        rate_class = "Tachycardic" if hr > 100 else "Bradycardic" if hr < 60 else "Normocardic"
        
        prompt = f"Write a {audience} report. ECG: {CONDITION_NAMES[cond]}, HR: {hr}, SpO2: {spo2}, RR: {rr}."
        
        if audience == "Doctor":
            diff_dx_list = "\n".join([f"* {dx}" for dx in ECG_DETAILS[cond]["diff_dx"]])
            
            completion = f"""
**1. PHYSIOLOGICAL PARAMETERS**
* **Heart rate:** {hr} bpm ({hr_status})
* **SpO2:** {spo2}% ({spo2_status})
* **Respiratory rate:** {rr} /min ({rr_status})
* **PPG quality:** Acceptable

**2. ECG FINDINGS**
* **Rhythm:** {ECG_DETAILS[cond]['rhythm']}
* **Rate classification:** {rate_class} — ventricular rate {hr} bpm
* **Diagnostic class:** {cond} — {CONDITION_NAMES[cond]}
* **Conduction:** {ECG_DETAILS[cond]['conduction']}
* **Morphology:** {ECG_DETAILS[cond]['morphology']}

**3. CLINICAL INTERPRETATION**
Findings are consistent with {CONDITION_NAMES[cond].lower()}. The patient's heart rate is {hr_status.lower()} and oxygen saturation is {spo2_status.lower()}. { 'Observation for cardiopulmonary compromise recommended.' if (hr>100 or spo2<94) else 'Physiological parameters are stable.' }

**4. DIFFERENTIAL DIAGNOSIS**
{diff_dx_list}

**5. RECOMMENDED ACTIONS**
{DOCTOR_ACTIONS[cond]}
            """.strip()
            
        else:
            patient_cond_desc = "irregular or showing signs of strain" if cond != "NORM" else "beating normally"
            hr_desc = "Running fast" if hr > 100 else "Running slow" if hr < 60 else "Healthy"
            spo2_desc = "Slightly low" if spo2 < 94 else "Healthy"
            rr_desc = "Faster than normal" if rr > 20 else "Healthy"
            
            completion = f"""
**1. YOUR VITALS**
* **Heart rate:** {hr} beats per minute ({hr_desc})
* **Oxygen level:** {spo2}% ({spo2_desc})
* **Breathing rate:** {rr} breaths per minute ({rr_desc})

**2. WHAT WE SAW ON THE ECG**
Your heart rhythm is {patient_cond_desc}, a condition related to {CONDITION_NAMES[cond].lower()}. Your heart rate is currently {hr} beats per minute.

**3. WHAT THIS MEANS**
{'Because your heart is beating unusually, it may be working harder than usual. This is likely why your oxygen or breathing might be affected.' if cond != 'NORM' else 'Your electrical heart signals look good, and your heart is functioning as expected for this test.'}

**4. NEXT STEPS & RECOMMENDATIONS**
{PATIENT_ACTIONS[cond]}
            """.strip()
            
        data["prompt"].append(prompt)
        data["completion"].append(completion)
        
    dataset = Dataset.from_dict(data)
    return dataset.train_test_split(test_size=0.1)

# Generate the dataset globally
dataset = generate_training_data(num_samples=1000)

# =====================================================================
# 3. TRAIN FLAN-T5 (Full Fine-Tuning) - UPDATED FOR TRANSFORMERS v4.46+
# =====================================================================
def train_flan_t5():
    print("\n--- Training Flan-T5-Small ---")
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def preprocess_function(examples):
        inputs = examples["prompt"]
        targets = examples["completion"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/flan-t5-medical",
        eval_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        num_train_epochs=4, 
        save_strategy="epoch",
        predict_with_generate=True,
        dataloader_num_workers=0,      # <--- ADD THIS LINE
    )

    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,   # <--- THIS IS THE FIX 
    )
    
    trainer.train()
    trainer.save_model("./models/flan-t5-medical-final")
    print("Flan-T5 saved successfully to ./models/flan-t5-medical-final")

# =====================================================================
# 4. TRAIN qwen 4 E2B (LoRA Fine-Tuning)
# =====================================================================
def train_qwen():
    print("\n--- Training qwen 4 E2B via LoRA ---")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"<start_of_turn>user\n{example['prompt'][i]}<end_of_turn>\n<start_of_turn>model\n{example['completion'][i]}<end_of_turn>"
            output_texts.append(text)
        return output_texts

    training_args = TrainingArguments(
        output_dir="./models/qwen-medical",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=10
    )

    trainer = SFTTrainer(
        model=model, train_dataset=dataset["train"], eval_dataset=dataset["test"],
        peft_config=peft_config, max_seq_length=512, tokenizer=tokenizer,
        args=training_args, formatting_func=formatting_prompts_func
    )
    trainer.train()
    trainer.model.save_pretrained("./models/qwen-medical-final")
    tokenizer.save_pretrained("./models/qwen-medical-final")
    print("qwen saved successfully to ./models/qwen-medical-final")

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    
    # Train T5 Model
    train_flan_t5()
    
    # Uncomment to also train qwen
    train_qwen()