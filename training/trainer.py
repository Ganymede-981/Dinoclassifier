"""
HAM10000 DINOv2-LoRA — Custom Trainer & Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer

from config import LABEL_SMOOTHING
from dataset import train_collate_fn, eval_collate_fn


# ── Loss ───────────────────────────────────────────────────────────────────────
class SoftTargetCrossEntropy(nn.Module):
    """
    Handles both:
      • hard labels  (LongTensor)  → standard weighted CE
      • soft labels  (FloatTensor) → manual CE (from CutMix / MixUp)
    """

    def __init__(self, weight=None, label_smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.weight = weight
        self.ls     = label_smoothing

    def forward(self, logits, labels):
        if labels.ndim == 1:
            return F.cross_entropy(
                logits, labels,
                weight=self.weight,
                label_smoothing=self.ls,
            )
        # Soft labels
        n      = logits.size(1)
        labels = labels * (1 - self.ls) + self.ls / n
        log_p  = F.log_softmax(logits, dim=-1)
        if self.weight is not None:
            w    = (labels * self.weight).sum(dim=-1)
            loss = -(labels * log_p).sum(dim=-1) * w
        else:
            loss = -(labels * log_p).sum(dim=-1)
        return loss.mean()


# ── Trainer ────────────────────────────────────────────────────────────────────
class HAMTrainer(Trainer):
    """
    Extends HF Trainer with:
      • WeightedRandomSampler + CutMix/MixUp for training
      • Plain eval collator for validation / test
      • SoftTargetCrossEntropy loss
    """

    def __init__(self, *args, sampler, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler  = sampler
        self._loss_fn  = loss_fn

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._sampler,
            collate_fn=train_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            ds,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=eval_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = self._loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
