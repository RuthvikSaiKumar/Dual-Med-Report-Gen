# =============================================================
# STAGE 3: CNN-LSTM Multi-Label ECG Classifier (PTB-XL)
# =============================================================
# Trains a CNN-LSTM model to classify 5 diagnostic superclasses:
#   NORM, MI, STTC, CD, HYP
# from 12-lead ECG signals preprocessed in Stage 2.
#
# Input:  X_train.npy  shape (N, 1000, 12)
#         y_train.npy  shape (N, 5)
# Output: best_model.pt         — best checkpoint by val AUC
#         training_curves.png   — loss & AUC over epochs
#         evaluation_report.txt — full metrics on test set
#         confusion_matrices.png
#
# Requirements:
#   pip install torch torchvision scikit-learn matplotlib numpy
# =============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── CONFIG ────────────────────────────────────────────────────
DATA_DIR    = "./outputs"          # where your .npy files are
OUTPUT_DIR  = "./outputs"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
LR_PATIENCE = 5       # reduce LR if val AUC doesn't improve for N epochs
ES_PATIENCE = 10      # early stop if val AUC doesn't improve for N epochs
WEIGHT_DECAY= 1e-4

# Model hyperparameters
N_LEADS     = 12
N_TIMESTEPS = 1000
N_CLASSES   = 5

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")


# =============================================================
# SECTION 1 — DATASET CLASS
# =============================================================
class ECGDataset(Dataset):
    """
    Loads preprocessed ECG arrays saved by Stage 2.
    X shape: (N, 1000, 12)  — timesteps × leads
    y shape: (N, 5)         — multi-label binary targets
    PyTorch Conv1d expects (batch, channels, length)
    so we transpose X to (N, 12, 1000) inside __getitem__.
    """
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]          # (1000, 12)
        y = self.y[idx]          # (5,)

        if self.augment:
            x = self._augment(x)

        x = x.permute(1, 0)      # → (12, 1000) for Conv1d
        return x, y

    def _augment(self, x):
        """Light augmentation — keeps clinical morphology intact."""
        # Gaussian noise
        if torch.rand(1) < 0.5:
            x = x + torch.randn_like(x) * 0.01

        # Random amplitude scaling (±10%)
        if torch.rand(1) < 0.5:
            scale = 0.9 + torch.rand(1) * 0.2
            x = x * scale

        # Random time shift (up to 50 samples = 0.5s at 100 Hz)
        if torch.rand(1) < 0.3:
            shift = int(torch.randint(-50, 50, (1,)).item())
            x = torch.roll(x, shift, dims=0)

        return x


# =============================================================
# SECTION 2 — CNN-LSTM ARCHITECTURE
# =============================================================
class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool with residual option."""
    def __init__(self, in_ch, out_ch, kernel=7, pool=2, dropout=0.2):
        super().__init__()
        self.conv   = nn.Conv1d(in_ch, out_ch, kernel,
                                padding=kernel // 2, bias=False)
        self.bn     = nn.BatchNorm1d(out_ch)
        self.act    = nn.ReLU()
        self.pool   = nn.MaxPool1d(pool)
        self.drop   = nn.Dropout(dropout)

        # 1×1 conv for residual if channel sizes differ
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.pool_res = nn.MaxPool1d(pool)

    def forward(self, x):
        res = self.pool_res(self.residual(x))
        x   = self.pool(self.drop(self.act(self.bn(self.conv(x)))))
        return x + res


class CNNLSTM(nn.Module):
    """
    Architecture:
      Input: (batch, 12, 1000)
      → 4 ConvBlocks: progressively larger channels, shrinking time dim
      → BiLSTM: captures long-range temporal patterns
      → Attention pooling: focus on clinically relevant timesteps
      → FC head: sigmoid outputs for multi-label classification

    Time dimension after ConvBlocks:
      1000 → 500 → 250 → 125 → 62  (4× MaxPool2)
    """
    def __init__(self, n_leads=12, n_classes=5,
                 cnn_channels=(32, 64, 128, 256),
                 lstm_hidden=256, lstm_layers=2,
                 dropout=0.3):
        super().__init__()

        # ── CNN feature extractor ────────────────────────────
        cnn_layers = []
        in_ch = n_leads
        for out_ch in cnn_channels:
            cnn_layers.append(ConvBlock(in_ch, out_ch, dropout=dropout * 0.7))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # ── Bidirectional LSTM ───────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = cnn_channels[-1],
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if lstm_layers > 1 else 0.0
        )
        lstm_out_size = lstm_hidden * 2   # bidirectional

        # ── Attention pooling ────────────────────────────────
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # ── Classification head ──────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes)
            # No sigmoid here — BCEWithLogitsLoss is numerically more stable
        )

    def forward(self, x):
        # x: (batch, 12, 1000)
        x = self.cnn(x)                   # (batch, 256, 62)
        x = x.permute(0, 2, 1)           # (batch, 62, 256) for LSTM

        x, _ = self.lstm(x)               # (batch, 62, 512)

        # Attention: weight each timestep
        attn_w = torch.softmax(self.attention(x), dim=1)   # (batch, 62, 1)
        x = (x * attn_w).sum(dim=1)       # (batch, 512)

        return self.classifier(x)         # (batch, 5) — raw logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================
# SECTION 3 — CLASS WEIGHTS (handle class imbalance)
# =============================================================
def compute_pos_weights(y_train):
    """
    For BCEWithLogitsLoss pos_weight:
    pos_weight[i] = (N - n_pos[i]) / n_pos[i]
    Upweights rare classes (MI, CD, HYP) vs common NORM.
    """
    n_pos = y_train.sum(axis=0)
    n_neg = len(y_train) - n_pos
    weights = n_neg / (n_pos + 1e-6)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# =============================================================
# SECTION 4 — TRAINING & VALIDATION LOOPS
# =============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        # Mixed precision (faster on GPU, harmless on CPU)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

        scaler.scale(loss).backward()
        # Gradient clipping — stabilises LSTM training
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(X_batch)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y_batch.detach().cpu())

    all_logits = torch.cat(all_logits).sigmoid().numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss   = total_loss / len(loader.dataset)

    # Macro AUC over classes that have both positive and negative samples
    try:
        auc = roc_auc_score(all_labels, all_logits, average="macro")
    except ValueError:
        auc = 0.0

    return avg_loss, auc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)

        total_loss += loss.item() * len(X_batch)
        all_logits.append(logits.cpu())
        all_labels.append(y_batch.cpu())

    all_logits = torch.cat(all_logits).sigmoid().numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss   = total_loss / len(loader.dataset)

    try:
        auc = roc_auc_score(all_labels, all_logits, average="macro")
    except ValueError:
        auc = 0.0

    return avg_loss, auc, all_logits, all_labels


# =============================================================
# SECTION 5 — FULL TRAINING PIPELINE
# =============================================================
def train(model, train_loader, val_loader, pos_weights):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_val_auc  = 0.0
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [],
                     "train_auc":  [], "val_auc":  []}

    print(f"\n{'Epoch':>6} {'Tr Loss':>9} {'Tr AUC':>8} "
          f"{'Vl Loss':>9} {'Vl AUC':>8} {'LR':>10} {'Time':>7}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_auc, _, _ = evaluate(
            model, val_loader, criterion)

        scheduler.step(vl_auc)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        print(f"{epoch:>6} {tr_loss:>9.4f} {tr_auc:>8.4f} "
              f"{vl_loss:>9.4f} {vl_auc:>8.4f} {cur_lr:>10.2e} "
              f"{elapsed:>6.1f}s")

        # Save best checkpoint
        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_auc":    best_val_auc,
                "config": {
                    "n_leads": N_LEADS, "n_classes": N_CLASSES,
                    "superclasses": SUPERCLASSES
                }
            }, os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"         ✓ Saved best model  (val AUC: {best_val_auc:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= ES_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {ES_PATIENCE} epochs)")
            break

    return history


# =============================================================
# SECTION 6 — PLOT TRAINING CURVES
# =============================================================
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color='#378ADD')
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color='#D85A30')
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Loss over epochs");  axes[0].legend()

    axes[1].plot(epochs, history["train_auc"], label="Train", color='#378ADD')
    axes[1].plot(epochs, history["val_auc"],   label="Val",   color='#D85A30')
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("Macro AUC-ROC")
    axes[1].set_title("AUC-ROC over epochs");  axes[1].legend()
    axes[1].set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: training_curves.png")


# =============================================================
# SECTION 7 — FULL EVALUATION ON TEST SET
# =============================================================
def find_best_thresholds(logits, labels):
    """
    Per-class threshold tuning on val set.
    Default 0.5 isn't always optimal for imbalanced multi-label data.
    """
    thresholds = []
    for i in range(labels.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.05):
            preds = (logits[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return np.array(thresholds)


def full_evaluation(model, test_loader, val_loader, criterion):
    print("\n" + "=" * 60)
    print("SECTION 7: Full evaluation on test set")
    print("=" * 60)

    # Tune thresholds on val set
    _, _, val_logits, val_labels = evaluate(model, val_loader, criterion)
    thresholds = find_best_thresholds(val_logits, val_labels)
    print(f"Per-class thresholds (tuned on val):")
    for cls, t in zip(SUPERCLASSES, thresholds):
        print(f"  {cls}: {t:.2f}")

    # Evaluate on test set
    _, _, test_logits, test_labels = evaluate(model, test_loader, criterion)
    test_preds = (test_logits >= thresholds).astype(int)

    # ── Per-class AUC ────────────────────────────────────────
    print("\nPer-class AUC-ROC:")
    aucs = []
    for i, cls in enumerate(SUPERCLASSES):
        try:
            auc = roc_auc_score(test_labels[:, i], test_logits[:, i])
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        print(f"  {cls}: {auc:.4f}")
    print(f"  Macro avg: {np.nanmean(aucs):.4f}")

    # ── Classification report ────────────────────────────────
    print("\nClassification report (threshold-tuned predictions):")
    report = classification_report(
        test_labels, test_preds,
        target_names=SUPERCLASSES,
        zero_division=0
    )
    print(report)

    # ── Macro & micro F1 ────────────────────────────────────
    f1_macro = f1_score(test_labels, test_preds,
                        average="macro", zero_division=0)
    f1_micro = f1_score(test_labels, test_preds,
                        average="micro", zero_division=0)
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Micro F1: {f1_micro:.4f}")

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("PTB-XL CNN-LSTM Classifier — Test Set Evaluation\n")
        f.write("=" * 55 + "\n\n")
        f.write("Per-class AUC-ROC:\n")
        for cls, auc in zip(SUPERCLASSES, aucs):
            f.write(f"  {cls}: {auc:.4f}\n")
        f.write(f"  Macro avg: {np.nanmean(aucs):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nMacro F1: {f1_macro:.4f}\n")
        f.write(f"Micro F1: {f1_micro:.4f}\n")
    print(f"Saved: evaluation_report.txt")

    # ── Confusion matrices ───────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, (cls, ax) in enumerate(zip(SUPERCLASSES, axes)):
        cm = confusion_matrix(test_labels[:, i], test_preds[:, i])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{cls}")
    plt.suptitle("Confusion matrices — test set (one per class)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: confusion_matrices.png")

    return test_logits, test_preds, thresholds


# =============================================================
# SECTION 8 — INFERENCE FUNCTION
# =============================================================
def predict_single(ecg_array, model_path, threshold=None):
    """
    Run inference on one preprocessed ECG record.

    Args:
        ecg_array  : np.ndarray shape (1000, 12) — already preprocessed
        model_path : path to best_model.pt
        threshold  : np.ndarray shape (5,) — per-class thresholds
                     if None, uses 0.5 for all classes

    Returns:
        dict with probabilities and predicted labels
    """
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x = torch.tensor(ecg_array, dtype=torch.float32)
    x = x.permute(1, 0).unsqueeze(0).to(DEVICE)  # (1, 12, 1000)

    with torch.no_grad():
        logits = model(x)
        probs  = logits.sigmoid().cpu().numpy()[0]  # (5,)

    if threshold is None:
        threshold = np.full(N_CLASSES, 0.5)

    predictions = {}
    for cls, prob, t in zip(SUPERCLASSES, probs, threshold):
        predictions[cls] = {
            "probability": round(float(prob), 4),
            "predicted":   bool(prob >= t),
        }

    # Determine primary condition for report generation
    detected = [cls for cls, v in predictions.items() if v["predicted"]]
    top_cls  = SUPERCLASSES[int(np.argmax(probs))]

    # Severity from top class probability
    top_prob = float(probs[np.argmax(probs)])
    if top_prob > 0.85:   severity = "high"
    elif top_prob > 0.65: severity = "moderate"
    else:                 severity = "mild"

    return {
        "per_class":         predictions,
        "detected_conditions": detected,
        "primary_condition": top_cls if detected else "NORM",
        "top_probability":   round(top_prob, 4),
        "severity":          severity,
    }


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":

    # ── Load preprocessed arrays ─────────────────────────────
    print("=" * 60)
    print("Loading preprocessed data from Stage 2...")
    print("=" * 60)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}     y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}    y_test : {y_test.shape}")

    # ── Datasets & loaders ───────────────────────────────────
    train_ds = ECGDataset(X_train, y_train, augment=True)
    val_ds   = ECGDataset(X_val,   y_val,   augment=False)
    test_ds  = ECGDataset(X_test,  y_test,  augment=False)

    # num_workers=0 on Windows to avoid multiprocessing issues
    # increase to 4 on Linux/Mac for faster loading
    n_workers = 0 if os.name == "nt" else 4

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=n_workers,
                              pin_memory=DEVICE.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=n_workers,
                              pin_memory=DEVICE.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=n_workers,
                              pin_memory=DEVICE.type == "cuda")

    print(f"\nBatches — train: {len(train_loader)}  "
          f"val: {len(val_loader)}  test: {len(test_loader)}")

    # ── Build model ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building CNN-LSTM model...")
    print("=" * 60)
    model = CNNLSTM(
        n_leads     = N_LEADS,
        n_classes   = N_CLASSES,
        cnn_channels= (32, 64, 128, 256),
        lstm_hidden = 256,
        lstm_layers = 2,
        dropout     = 0.3
    ).to(DEVICE)

    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"\nModel summary:")
    print(f"  CNN   : 4 × ConvBlock (32→64→128→256 channels)")
    print(f"  Output: time 1000 → 500 → 250 → 125 → 62")
    print(f"  LSTM  : BiLSTM (hidden=256, layers=2) → 512 features")
    print(f"  Attn  : soft attention pooling over 62 timesteps")
    print(f"  Head  : FC(512→128→5) + sigmoid")

    # ── Class weights ────────────────────────────────────────
    pos_weights = compute_pos_weights(y_train)
    print(f"\nPositive weights (for class imbalance):")
    for cls, w in zip(SUPERCLASSES, pos_weights.cpu().numpy()):
        print(f"  {cls}: {w:.2f}×")

    # ── Train ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    history = train(model, train_loader, val_loader, pos_weights)
    plot_training_curves(history)

    # ── Load best model for evaluation ───────────────────────
    print("\nLoading best checkpoint for evaluation...")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"),
                            map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Best model was from epoch {checkpoint['epoch']} "
          f"(val AUC: {checkpoint['val_auc']:.4f})")

    # ── Evaluate ─────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    test_logits, test_preds, thresholds = full_evaluation(
        model, test_loader, val_loader, criterion
    )

    # Save thresholds for inference
    np.save(os.path.join(OUTPUT_DIR, "best_thresholds.npy"), thresholds)
    print("Saved: best_thresholds.npy")

    # ── Quick inference example ──────────────────────────────
    print("\n" + "=" * 60)
    print("Example: running inference on one test record")
    print("=" * 60)
    sample_ecg = X_test[0]   # (1000, 12)
    result = predict_single(
        sample_ecg,
        model_path = os.path.join(OUTPUT_DIR, "best_model.pt"),
        threshold  = thresholds
    )
    print("Inference result:")
    print(f"  Detected conditions : {result['detected_conditions']}")
    print(f"  Primary condition   : {result['primary_condition']}")
    print(f"  Severity            : {result['severity']}")
    print(f"  Top probability     : {result['top_probability']}")
    print(f"\n  Per-class breakdown:")
    for cls, v in result["per_class"].items():
        flag = "✓" if v["predicted"] else " "
        print(f"  {flag} {cls}: prob={v['probability']:.4f}")

    print("\n" + "=" * 60)
    print("✓ Stage 3 complete.")
    print("Outputs saved to ./outputs/:")
    print("  best_model.pt         — model checkpoint")
    print("  best_thresholds.npy   — per-class decision thresholds")
    print("  training_curves.png   — loss & AUC curves")
    print("  evaluation_report.txt — full test set metrics")
    print("  confusion_matrices.png")
    print("\nNext: Stage 4 — structured feature vector + RAG + NLP report generation")
    print("=" * 60)
