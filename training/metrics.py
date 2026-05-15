"""
HAM10000 DINOv2-LoRA — Evaluation Metrics
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=-1)
    probs  = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    ).numpy()

    metrics = {
        "accuracy":    float(accuracy_score(labels, preds)),
        "f1_macro":    float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }
    try:
        metrics["roc_auc_macro"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["roc_auc_macro"] = 0.0

    return metrics
