"""
HAM10000 DINOv2-LoRA Skin Lesion Classifier
FastAPI Backend - Loads model from HuggingFace Hub at startup
"""

import os
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import Dinov2Model
from PIL import Image
import torchvision.transforms as T
from contextlib import asynccontextmanager

# ── Constants ──────────────────────────────────────────────────────────────────
HUB_REPO = "Ganymede981/ham10000-vit"
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

# Global model reference
model     = None
transform = None


# ── Model Definition ───────────────────────────────────────────────────────────
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


# ── Startup / Shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform
    print(f"[INFO] Loading model from Hub: {HUB_REPO}")
    print(f"[INFO] Device: {DEVICE}")

    local = snapshot_download(HUB_REPO)

    # Rebuild backbone + LoRA adapter
    base      = Dinov2Model.from_pretrained("facebook/dinov2-base")
    lora_dir  = os.path.join(local, "dinov2_lora")
    base      = PeftModel.from_pretrained(base, lora_dir)

    # Rebuild classification head (must match training architecture)
    head = nn.Sequential(
        nn.LayerNorm(768 * 2),
        nn.Dropout(0.1),
        nn.Linear(768 * 2, len(HAM_LABELS)),
    )
    head_path = os.path.join(local, "head.pt")
    head.load_state_dict(torch.load(head_path, map_location="cpu"))

    model = DINOv2Classifier(base, head).to(DEVICE).eval()

    # HAM10000 channel statistics
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
    # cleanup (nothing to do)


# ── App ────────────────────────────────────────────────────────────────────────
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


# ── Routes ─────────────────────────────────────────────────────────────────────
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
        probs  = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    pred_idx   = int(torch.argmax(torch.tensor(probs)))
    pred_label = HAM_LABELS[pred_idx]

    return {
        "prediction":        pred_label,
        "description":       LABEL_DESCRIPTIONS[pred_label],
        "confidence":        round(probs[pred_idx], 4),
        "probabilities":     {
            label: round(p, 4)
            for label, p in zip(HAM_LABELS, probs)
        },
        "descriptions":      LABEL_DESCRIPTIONS,
    }
