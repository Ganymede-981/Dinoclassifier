"""
HAM10000 DINOv2-LoRA — Model Architecture
Defines the DINOv2Classifier used for both training and inference.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model
from transformers import Dinov2Model
from transformers.utils import ModelOutput as HFModelOutput


@dataclass
class ModelOutput(HFModelOutput):
    logits: torch.Tensor = None


class DINOv2Classifier(nn.Module):
    """
    DINOv2-base backbone with a lightweight two-layer head.

    Head input  = [CLS token | mean of patch tokens]  →  768 * 2 = 1536 dims
    Head output = num_classes logits
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.1):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        hidden = self.dinov2.config.hidden_size  # 768 for -base
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, pixel_values, labels=None):
        out  = self.dinov2(pixel_values=pixel_values)
        cls  = out.last_hidden_state[:, 0]
        mean = out.last_hidden_state[:, 1:].mean(dim=1)
        feats  = torch.cat([cls, mean], dim=1)
        logits = self.head(feats)
        return ModelOutput(logits=logits)


def build_model(
    num_classes: int  = 7,
    lora_r: int       = 32,
    lora_alpha: int   = 64,
    lora_dropout: float = 0.05,
    head_dropout: float = 0.1,
) -> DINOv2Classifier:
    """Build a LoRA-wrapped DINOv2Classifier with gradient checkpointing."""
    model = DINOv2Classifier(num_classes=num_classes, dropout=head_dropout)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value", "out_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.dinov2 = get_peft_model(model.dinov2, lora_cfg)
    model.dinov2.base_model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.dinov2.print_trainable_parameters()
    return model
