---
title: HAM10000 Skin Lesion Classifier API
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# HAM10000 Skin Lesion Classifier — Backend API

FastAPI backend serving a **DINOv2-LoRA** model fine-tuned on the HAM10000 skin lesion dataset.  
Classifies dermoscopy images into 7 categories with per-class calibrated confidence thresholds.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Status + label list |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Upload an image → returns prediction + probabilities |
| `GET` | `/thresholds` | Currently active per-class thresholds |

## Classes
- Actinic Keratoses
- Basal Cell Carcinoma
- Benign Keratosis-like Lesions
- Dermatofibroma
- Melanocytic Nevi
- Melanoma
- Vascular Lesions

## Model
Weights loaded from [`Ganymede981/ham10000-vit`](https://huggingface.co/Ganymede981/ham10000-vit) at startup.
