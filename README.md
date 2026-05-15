# HAM10000 Skin Lesion Classifier 🩺

A **DINOv2-LoRA** model fine-tuned on the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dermatoscopy dataset for **7-class skin lesion classification**, paired with a FastAPI inference server and a polished drag-and-drop web interface.

---

## 🏆 Model Performance (Test Set)

| Metric | Score |
|---|---|
| Accuracy | **91.67 %** |
| F1 Macro | **93.18 %** |
| ROC-AUC Macro | **99.49 %** |

Model weights are hosted on 🤗 Hugging Face: [`Ganymede981/ham10000-vit`](https://huggingface.co/Ganymede981/ham10000-vit)

---

## 📂 Repository Layout

```
Dinoclassifier/
├── backend/
│   ├── main.py              ← FastAPI app (loads model from Hub at startup)
│   └── requirements.txt
├── frontend/
│   └── index.html           ← Single-file drag-and-drop UI (no build step)
├── model/
│   └── architecture.py      ← DINOv2Classifier + build_model() helper
├── training/
│   ├── config.py            ← All hyper-parameters in one place
│   ├── dataset.py           ← HAMDataset, collators, WeightedRandomSampler
│   ├── trainer.py           ← HAMTrainer (custom HF Trainer) + SoftCE loss
│   ├── metrics.py           ← compute_metrics for HF Trainer callback
│   ├── train.py             ← Kaggle training entry-point
│   └── evaluate.py          ← Standalone test-set evaluation + plots
└── notebook/
    └── training_notebook.ipynb   ← Original Kaggle notebook (reference)
```

---

## 🚀 Running the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The first startup downloads model weights from the Hub (~350 MB), then serves:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info + label list |
| `/health` | GET | Liveness check |
| `/predict` | POST | Upload an image → get prediction |

### Quick test with `curl`

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@your_skin_image.jpg"
```

Response:

```json
{
  "prediction": "melanocytic_Nevi",
  "description": "Melanocytic Nevi (mole)",
  "confidence": 0.9723,
  "probabilities": { ... }
}
```

---

## 🖼 Frontend

Open `frontend/index.html` directly in any browser, or serve it with:

```bash
python -m http.server 3000 --directory frontend
```

> **Note:** change `API_URL` at the top of `index.html` if your backend runs on a different host/port.

---

## 🏋️ Re-training (Kaggle)

Upload the `training/` folder to a Kaggle notebook, add your `HF_TOKEN` secret, and run `train.py`. The script:

- Downloads `marmal88/skin_cancer` from 🤗 Datasets
- Applies stratified 85/15 train-val split
- Trains DINOv2-base with LoRA (r=32) + CutMix/MixUp + WeightedRandomSampler
- Auto-resumes from the latest checkpoint pushed to the Hub
- Pushes the final model back to the Hub

---

## 🔬 Classes

| Short code | Full name |
|---|---|
| `akiec` | Actinic Keratoses |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions |
| `df` | Dermatofibroma |
| `nv` | Melanocytic Nevi |
| `mel` | Melanoma |
| `vasc` | Vascular Lesions |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist.
