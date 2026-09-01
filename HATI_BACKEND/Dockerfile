FROM python:3.10-slim

# ffmpeg: transformers' ASR pipeline shells out to it to decode the saved
# audio file path passed into stt_model(). libsndfile1: required by the
# soundfile package (used for the audio emotion classifier).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces runs Docker containers as a non-root user (UID 1000)
# and only guarantees the container's own working dir + $HOME as writable —
# without this, transformers' default model cache dir (~/.cache/huggingface)
# fails to write and every model load crashes.
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
ENV HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
RUN mkdir -p $HF_HOME && chown -R appuser:appuser /home/appuser
USER appuser

EXPOSE 7860

# Single worker, multiple threads: PyTorch releases the GIL during tensor
# ops, so threads give real concurrency here without loading a second copy
# of the 3 ML models (each worker process would load its own). Long timeout
# because CPU-only STT/emotion inference is slow compared to GPU.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
