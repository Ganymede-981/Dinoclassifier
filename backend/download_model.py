"""
Pre-download script run during Docker IMAGE BUILD (not at runtime).
This bakes all model weights into the image layer so cold starts are instant.

Downloads:
  1. Ganymede981/ham10000-vit  (LoRA adapter + head.pt + class_thresholds.json)
  2. facebook/dinov2-base       (backbone weights)
"""

import os
import json

# Silence symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download
from transformers import Dinov2Model

CACHE_DIR = "/app/.hub_cache"
HAM_REPO  = "Ganymede981/ham10000-vit"

print(f"[BUILD] Downloading {HAM_REPO} → {CACHE_DIR}")
snapshot_download(
    HAM_REPO,
    local_dir=CACHE_DIR,
    local_dir_use_symlinks=False,
)
print(f"[BUILD] {HAM_REPO} download complete.")

print("[BUILD] Downloading facebook/dinov2-base backbone...")
# Download to HF_HOME cache so transformers finds it at runtime
Dinov2Model.from_pretrained("facebook/dinov2-base")
print("[BUILD] facebook/dinov2-base download complete.")

print("[BUILD] All model files cached. Docker layer ready.")
