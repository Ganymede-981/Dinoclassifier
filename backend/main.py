import io
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from transformers import Dinov2Model

HUB_REPO   = "Ganymede981/ham10000-vit"
# Absolute path written by download_model.py at Docker BUILD time.
# At runtime, the weights are already on disk — no download needed.
_HUB_CACHE = "/app/.hub_cache"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

HAM_LABELS = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis-like_lesions",
    "dermatofibroma",
    "melanocytic_Nevi",
    "melanoma",
    "vascular_lesions",
]

LABEL_DESCRIPTIONS = {
    "actinic_keratoses":             "Actinic Keratoses (pre-cancerous)",
    "basal_cell_carcinoma":          "Basal Cell Carcinoma",
    "benign_keratosis-like_lesions": "Benign Keratosis-like Lesions",
    "dermatofibroma":                "Dermatofibroma",
    "melanocytic_Nevi":              "Melanocytic Nevi (mole)",
    "melanoma":                      "Melanoma (skin cancer)",
    "vascular_lesions":              "Vascular Lesions",
}


model      = None
transform  = None
THRESHOLDS = {} 

_DEFAULT_THRESHOLDS = {
    "actinic_keratoses":              0.5,
    "basal_cell_carcinoma":           0.5,
    "benign_keratosis-like_lesions":  0.5,
    "dermatofibroma":                 0.5,
    "melanocytic_Nevi":               0.5,
    "melanoma":                       0.5,
    "vascular_lesions":               0.5,
}

class DINOv2Classifier(nn.Module):
    def __init__(self, dinov2_backbone, head):
        super().__init__()
        self.dinov2 = dinov2_backbone
        self.head   = head

    def forward(self, pixel_values):
        out  = self.dinov2(pixel_values=pixel_values)
        cls  = out.last_hidden_state[:, 0]
        mean = out.last_hidden_state[:, 1:].mean(dim=1)
        return self.head(torch.cat([cls, mean], dim=1))


def predict_calibrated(
    probs: np.ndarray,
    threshold_map: dict,
) -> dict:
    thresholds = np.array(
        [threshold_map.get(label, 0.5) for label in HAM_LABELS],
        dtype=np.float32,
    )
    margins     = probs - thresholds
    pred_idx    = int(np.argmax(margins))
    pred_label  = HAM_LABELS[pred_idx]

    return {
        "prediction":       pred_label,
        "description":      LABEL_DESCRIPTIONS[pred_label],
        "confidence":       round(float(probs[pred_idx]), 4),
        "threshold":        round(float(thresholds[pred_idx]), 4),
        "threshold_cleared": bool(margins[pred_idx] >= 0),
        "probabilities":    {
            label: round(float(p), 4)
            for label, p in zip(HAM_LABELS, probs)
        },
        "thresholds":       {
            label: round(float(t), 4)
            for label, t in zip(HAM_LABELS, thresholds)
        },
        "descriptions":     LABEL_DESCRIPTIONS,
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform, THRESHOLDS
    print(f"[INFO] Loading model from pre-baked cache: {_HUB_CACHE}")
    print(f"[INFO] Device: {DEVICE}")

    # ── Resolve local weights directory ──────────────────────────────────────
    # In the Docker image, weights were downloaded by download_model.py at
    # BUILD time. snapshot_download is idempotent: if all files are already
    # present it returns immediately without any network traffic.
    local = snapshot_download(
        HUB_REPO,
        local_dir=_HUB_CACHE,
        local_dir_use_symlinks=False,
    )

    # ── Load thresholds ───────────────────────────────────────────────────────
    thresholds_path = os.path.join(local, "class_thresholds.json")
    if os.path.isfile(thresholds_path):
        with open(thresholds_path) as f:
            raw = json.load(f)
        THRESHOLDS = raw.get("thresholds", raw)
        print(f"[INFO] Thresholds loaded: {THRESHOLDS}")
    else:
        THRESHOLDS = _DEFAULT_THRESHOLDS
        print("[WARN] class_thresholds.json not found — using default 0.5 thresholds.")

    # ── Build model from cached weights ───────────────────────────────────────
    # transformers reads facebook/dinov2-base from HF_HOME (/app/.cache),
    # which was populated by download_model.py — no internet call needed.
    base     = Dinov2Model.from_pretrained("facebook/dinov2-base")
    lora_dir = os.path.join(local, "dinov2_lora")
    base     = PeftModel.from_pretrained(base, lora_dir)
    head = nn.Sequential(
        nn.LayerNorm(768 * 2),
        nn.Dropout(0.1),
        nn.Linear(768 * 2, len(HAM_LABELS)),
    )
    head_path = os.path.join(local, "head.pt")
    head.load_state_dict(torch.load(head_path, map_location="cpu"))

    model = DINOv2Classifier(base, head).to(DEVICE).eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.7630, 0.5456, 0.5700],
            std =[0.1409, 0.1521, 0.1697],
        ),
    ])

    print("[INFO] Model loaded and ready.")
    yield

app = FastAPI(
    title="HAM10000 Skin Lesion Classifier",
    description="DINOv2-LoRA model fine-tuned on HAM10000 for 7-class skin lesion classification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "HAM10000 Skin Lesion Classifier API",
        "labels": HAM_LABELS,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "device": DEVICE, "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    return predict_calibrated(probs, threshold_map=THRESHOLDS)


@app.get("/thresholds")
def get_thresholds():
    """Return the currently active per-class thresholds."""
    return {"thresholds": THRESHOLDS}
