import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2 as T2
from collections import Counter
from datasets import load_dataset, ClassLabel

from config import (
    DATASET_ID, LABEL_FIELD,
    HAM_LABELS, NUM_CLASSES, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD,
    CUTMIX_ALPHA, MIXUP_ALPHA,
)


LABEL2IDX = {l: i for i, l in enumerate(HAM_LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(HAM_LABELS)}


def load_splits():
    print(f"Loading {DATASET_ID}...")
    raw = load_dataset(DATASET_ID)
    print(raw)

    new_features = raw["train"].features.copy()
    new_features[LABEL_FIELD] = ClassLabel(names=HAM_LABELS)

    # Cast all three official splits to ClassLabel
    raw["train"]      = raw["train"].cast(new_features)
    raw["validation"] = raw["validation"].cast(new_features)
    raw["test"]       = raw["test"].cast(new_features)

    train_raw = raw["train"]       # 9,577 samples
    val_raw   = raw["validation"]  # 2,492 samples (official split)
    test_raw  = raw["test"]        # 1,285 samples (held-out)

    print(f"[INFO] Split sizes -> Train={len(train_raw):,}  "
          f"Val={len(val_raw):,}  Test={len(test_raw):,}")

    return train_raw, val_raw, test_raw


class HAMDataset(Dataset):
    def __init__(self, hf_split, augment: bool = False):
        self.data = hf_split
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ])

    def __len__(self):
        return len(self.data)

    def _get_label(self, sample) -> int:
        raw = sample[LABEL_FIELD]
        if isinstance(raw, int):   return raw
        if isinstance(raw, str):   return LABEL2IDX[raw.strip().lower()]
        return int(raw)

    def __getitem__(self, idx):
        s   = self.data[idx]
        img = s.get("image", s.get("img"))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        return {"pixel_values": self.transform(img), "labels": self._get_label(s)}


_cutmix = T2.CutMix(num_classes=NUM_CLASSES, alpha=CUTMIX_ALPHA)
_mixup  = T2.MixUp(num_classes=NUM_CLASSES,  alpha=MIXUP_ALPHA)


def train_collate_fn(batch):
    """Stochastic CutMix / MixUp for the training dataloader."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    if torch.rand(1).item() < 0.5:
        pixel_values, labels = _cutmix(pixel_values, labels)
    else:
        pixel_values, labels = _mixup(pixel_values, labels)
    return {"pixel_values": pixel_values, "labels": labels}


def eval_collate_fn(batch):
    """Plain collator — hard integer labels, no augmentation."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def build_sampler_and_weights(train_raw):
    train_labels = list(train_raw[LABEL_FIELD])
    if isinstance(train_labels[0], str):
        train_labels = [LABEL2IDX[l] for l in train_labels]

    counts    = Counter(train_labels)
    counts_t  = torch.tensor([counts[i] for i in range(NUM_CLASSES)], dtype=torch.float32)
    total     = counts_t.sum()

    class_weights  = (total / (NUM_CLASSES * counts_t)).clamp(0.5, 20.0)
    sample_weights = torch.tensor(
        [1.0 / counts[l] for l in train_labels], dtype=torch.float64
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_weights
