FROM python:3.11-slim

EXPOSE 7860

WORKDIR /app

# System deps for Pillow / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller image on Spaces free tier)
RUN pip install --no-cache-dir \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY main.py .

# HF_TOKEN is injected as a Space secret — snapshot_download uses it automatically
# via the HUGGING_FACE_HUB_TOKEN env var (set in Space settings).
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/app/.cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
