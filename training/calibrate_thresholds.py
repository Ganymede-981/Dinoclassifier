"""
calibrate_thresholds.py
-----------------------
Derive per-class decision thresholds from the *validation* split by
maximising the per-class F1 score over a fine threshold grid.

The test split is intentionally kept untouched so it remains a clean,
unbiased held-out benchmark.

Output
------
class_thresholds.json   — drop-in replacement for the hand-tuned file;
                          compatible with the backend's load_model() format.
calibration_curve.png   — optional diagnostic plot (threshold vs F1 per class).

Usage
-----
    python calibrate_thresholds.py [--output PATH] [--steps N] [--metric {f1,sensitivity,youden}]

Arguments
---------
--output   Where to write the JSON file.  Default: class_thresholds.json
--steps    Number of threshold candidates in [0.01, 0.99].  Default: 99
--metric   Objective to maximise per class:
             f1          — maximises F1  (default, good general-purpose choice)
             sensitivity — maximises recall (prioritises missing no positives)
             youden      — maximises Youden's J = sensitivity + specificity - 1
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from peft import PeftModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import Dinov2Model

# ── Make sure sibling modules are importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from config import HUB_REPO, HAM_LABELS, NUM_CLASSES
from dataset import load_splits, HAMDataset, eval_collate_fn


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Calibrate per-class decision thresholds.")
    p.add_argument(
        "--output", default="class_thresholds.json",
        help="Output JSON path (default: class_thresholds.json)",
    )
    p.add_argument(
        "--steps", type=int, default=99,
        help="Number of threshold grid points in [0.01, 0.99] (default: 99)",
    )
    p.add_argument(
        "--metric", choices=["f1", "sensitivity", "youden"], default="f1",
        help="Per-class objective to maximise (default: f1)",
    )
    return p.parse_args()


# ── Auth & model download ─────────────────────────────────────────────────────
def setup_model():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set.  Add it to the .env file at the project root."
        )
    login(hf_token)

    print(f"[INFO] Downloading model from {HUB_REPO} ...")
    local = snapshot_download(HUB_REPO)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base     = Dinov2Model.from_pretrained("facebook/dinov2-base")
    lora_dir = os.path.join(local, "dinov2_lora")
    base     = PeftModel.from_pretrained(base, lora_dir).to(device)

    head = nn.Sequential(
        nn.LayerNorm(768 * 2),
        nn.Dropout(0.1),
        nn.Linear(768 * 2, NUM_CLASSES),
    )
    head.load_state_dict(
        torch.load(os.path.join(local, "head.pt"), map_location="cpu")
    )
    head = head.to(device)

    class DINOv2Classifier(nn.Module):
        def forward(self, pixel_values):
            out  = base(pixel_values=pixel_values)
            cls  = out.last_hidden_state[:, 0]
            mean = out.last_hidden_state[:, 1:].mean(dim=1)
            return head(torch.cat([cls, mean], dim=1))

    model  = DINOv2Classifier().to(device).eval()
    print(f"[INFO] Model ready on {device}.")
    return model, device


# ── Inference on the validation set ──────────────────────────────────────────
def collect_val_probs(model, device):
    _, val_raw, _ = load_splits()
    val_ds = HAMDataset(val_raw, augment=False)
    val_dl = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        collate_fn=eval_collate_fn, num_workers=4,
    )

    all_logits, all_labels = [], []
    print("[INFO] Running inference on validation set ...")
    with torch.no_grad():
        for batch in val_dl:
            logits = model(batch["pixel_values"].to(device)).cpu()
            all_logits.append(logits)
            all_labels.extend(batch["labels"].tolist())

    logits = torch.cat(all_logits)
    probs  = torch.softmax(logits.float(), dim=-1).numpy()   # (N, C)
    labels = np.array(all_labels)                             # (N,)
    print(f"[INFO] Collected {len(labels):,} validation samples.")
    return probs, labels


# ── Threshold search ──────────────────────────────────────────────────────────
def best_threshold_per_class(probs, labels, steps, metric):
    """
    For every class c, treat it as a binary problem:
        positive  -> true label == c
        score     -> probs[:, c]

    Then sweep candidate thresholds and pick the one that maximises
    the chosen metric.

    Returns
    -------
    thresholds : dict  {label_name: float}
    val_scores : dict  {label_name: float}   (score at the chosen threshold)
    curve_data : list of (label_name, grid, scores_along_grid)
    """
    grid       = np.linspace(0.01, 0.99, steps)
    thresholds = {}
    val_scores = {}
    curve_data = []

    for c, label in enumerate(HAM_LABELS):
        y_true = (labels == c).astype(int)      # binary ground truth
        scores = probs[:, c]                     # model confidence for class c

        best_thr   = 0.5
        best_score = -1.0
        score_grid = np.zeros(len(grid))

        for i, thr in enumerate(grid):
            y_pred = (scores >= thr).astype(int)

            if metric == "f1":
                s = f1_score(y_true, y_pred, zero_division=0)

            elif metric == "sensitivity":
                # recall of the positive class
                tp = int(((y_pred == 1) & (y_true == 1)).sum())
                fn = int(((y_pred == 0) & (y_true == 1)).sum())
                s  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            elif metric == "youden":
                tp = int(((y_pred == 1) & (y_true == 1)).sum())
                fn = int(((y_pred == 0) & (y_true == 1)).sum())
                tn = int(((y_pred == 0) & (y_true == 0)).sum())
                fp = int(((y_pred == 1) & (y_true == 0)).sum())
                sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                s = sensitivity + specificity - 1.0

            score_grid[i] = s
            if s > best_score:
                best_score = s
                best_thr   = thr

        # Round to 4 decimal places — still precise, but readable
        thresholds[label] = round(float(best_thr), 4)
        val_scores[label] = round(float(best_score), 4)
        curve_data.append((label, grid, score_grid))

        print(
            f"  {label:<40s}  thr={best_thr:.4f}  {metric}={best_score:.4f}"
        )

    return thresholds, val_scores, curve_data


# ── Diagnostic plot ───────────────────────────────────────────────────────────
def save_calibration_plot(curve_data, thresholds, metric, out_path):
    n   = len(curve_data)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, (label, grid, scores) in enumerate(curve_data):
        ax = axes[i]
        ax.plot(grid, scores, linewidth=1.5, color="steelblue")
        best_thr = thresholds[label]
        best_score = scores[np.argmin(np.abs(grid - best_thr))]
        ax.axvline(best_thr, color="tomato", linestyle="--", linewidth=1.2,
                   label=f"thr={best_thr:.4f}")
        ax.scatter([best_thr], [best_score], color="tomato", zorder=5)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Threshold", fontsize=7)
        ax.set_ylabel(metric, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide the spare subplot (7 classes -> 8th cell unused)
    if n < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle(f"Threshold calibration -- maximising per-class {metric}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Calibration plot saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    model, device        = setup_model()
    probs, labels        = collect_val_probs(model, device)

    print(f"\n[INFO] Optimising thresholds (metric={args.metric}, steps={args.steps}) ...\n")
    thresholds, val_scores, curve_data = best_threshold_per_class(
        probs, labels, steps=args.steps, metric=args.metric
    )

    # ── Build output compatible with backend's class_thresholds.json format ──
    output = {
        "thresholds":         thresholds,
        "val_f1":             val_scores,     # scores at the chosen thresholds
        "calibration_metric": args.metric,
        "calibration_steps":  args.steps,
        "labels":             HAM_LABELS,
        "model_repo":         HUB_REPO,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Thresholds saved to {args.output}")

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    plot_path = os.path.splitext(args.output)[0] + "_calibration_curve.png"
    save_calibration_plot(curve_data, thresholds, args.metric, plot_path)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n-- Calibrated thresholds --------------------------------------------------")
    print(f"{'Class':<40s}  {'Threshold':>9}  {args.metric:>12}")
    print("-" * 66)
    for label in HAM_LABELS:
        print(
            f"  {label:<38s}  {thresholds[label]:>9.4f}  {val_scores[label]:>12.4f}"
        )
    print("-" * 66)
    print(
        "\nDrop the generated class_thresholds.json into your model repo "
        "or backend/.hub_cache/ to activate the calibrated thresholds.\n"
    )


if __name__ == "__main__":
    main()
