"""
HAM10000 DINOv2-LoRA — Test-Set Evaluation
Run:  python evaluate.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import login, snapshot_download
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from transformers import Dinov2Model
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from config import HUB_REPO, HAM_LABELS, NUM_CLASSES
from dataset import load_splits, HAMDataset, eval_collate_fn
from torch.utils.data import DataLoader

# ── Auth ───────────────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    login(hf_token)
except Exception:
    pass   # token already cached / not on Kaggle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load Model ─────────────────────────────────────────────────────────────────
print(f"[INFO] Downloading model from {HUB_REPO}...")
local = snapshot_download(HUB_REPO)

base      = Dinov2Model.from_pretrained("facebook/dinov2-base")
lora_dir  = os.path.join(local, "dinov2_lora")
base      = PeftModel.from_pretrained(base, lora_dir)

head = nn.Sequential(
    nn.LayerNorm(768 * 2),
    nn.Dropout(0.1),
    nn.Linear(768 * 2, NUM_CLASSES),
)
head.load_state_dict(torch.load(os.path.join(local, "head.pt"), map_location="cpu"))

class DINOv2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov2 = base
        self.head   = head
    def forward(self, pixel_values):
        out  = self.dinov2(pixel_values=pixel_values)
        cls  = out.last_hidden_state[:, 0]
        mean = out.last_hidden_state[:, 1:].mean(dim=1)
        return self.head(torch.cat([cls, mean], dim=1))

model = DINOv2Classifier().to(DEVICE).eval()
print("[INFO] Model ready.")

# ── Data ───────────────────────────────────────────────────────────────────────
_, _, test_raw = load_splits()
test_ds  = HAMDataset(test_raw, augment=False)
test_dl  = DataLoader(test_ds, batch_size=64, shuffle=False,
                      collate_fn=eval_collate_fn, num_workers=4)

# ── Inference ──────────────────────────────────────────────────────────────────
all_logits, all_labels = [], []

with torch.no_grad():
    for batch in test_dl:
        pixel_values = batch["pixel_values"].to(DEVICE)
        logits       = model(pixel_values).cpu()
        all_logits.append(logits)
        all_labels.extend(batch["labels"].tolist())

logits = torch.cat(all_logits).numpy()
labels = np.array(all_labels)
preds  = np.argmax(logits, axis=-1)
probs  = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

# ── Metrics ────────────────────────────────────────────────────────────────────
print("\n=== Classification Report ===")
print(classification_report(labels, preds, target_names=HAM_LABELS, zero_division=0))

acc  = float(accuracy_score(labels, preds))
f1_m = float(f1_score(labels, preds, average="macro",    zero_division=0))
f1_w = float(f1_score(labels, preds, average="weighted", zero_division=0))
try:
    auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
except Exception:
    auc = 0.0

print(f"Accuracy        : {acc:.4f}")
print(f"F1 (macro)      : {f1_m:.4f}")
print(f"F1 (weighted)   : {f1_w:.4f}")
print(f"ROC-AUC (macro) : {auc:.4f}")

results = {
    "accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w, "roc_auc_macro": auc,
    "per_class": classification_report(
        labels, preds, target_names=HAM_LABELS, output_dict=True, zero_division=0
    ),
}
with open("ham10000_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[INFO] Results saved to ham10000_results.json")

# ── Plots ──────────────────────────────────────────────────────────────────────
cm      = confusion_matrix(labels, preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
per_f1  = f1_score(labels, preds, average=None, zero_division=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

im = axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks(range(NUM_CLASSES))
axes[0].set_xticklabels(HAM_LABELS, rotation=45, ha="right")
axes[0].set_yticks(range(NUM_CLASSES))
axes[0].set_yticklabels(HAM_LABELS)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].set_title("Normalised Confusion Matrix")
for r in range(NUM_CLASSES):
    for c in range(NUM_CLASSES):
        axes[0].text(c, r, f"{cm_norm[r,c]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if cm_norm[r, c] > 0.5 else "black")
plt.colorbar(im, ax=axes[0])

axes[1].bar(HAM_LABELS, per_f1, color="steelblue")
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("F1 Score")
axes[1].set_title("Per-Class F1 on Test Set")
plt.xticks(rotation=30, ha="right")
for i, v in enumerate(per_f1):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("ham10000_eval.png", dpi=150)
plt.show()
print("[INFO] Plot saved to ham10000_eval.png")
