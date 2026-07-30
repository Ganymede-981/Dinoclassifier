FROM python:3.11-slim

EXPOSE 7860

WORKDIR /app

# ── System deps for Pillow / PyTorch ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch first (much smaller image on free tier) ─────────
RUN pip install --no-cache-dir \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ──────────────────────────────────────────────────────────────
COPY main.py .
COPY llm_report.py .
COPY download_model.py .

# ── Paths & env ──────────────────────────────────────────────────────────────
#   HF_HOME  → where transformers caches facebook/dinov2-base
#   HAM cache→ /app/.hub_cache  (hardcoded in download_model.py & main.py)
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/app/.cache

# ── PRE-DOWNLOAD MODELS AT BUILD TIME ────────────────────────────────────────
#   The HF_TOKEN build-arg is passed during `docker build` on HF Spaces
#   (set it as a Space SECRET named HF_TOKEN).
#   This bakes all weights into the image layer → zero download at cold start.
ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

RUN python download_model.py

# ── Runtime ───────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
