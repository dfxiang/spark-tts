FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SPARK_TTS_MODEL_DIR=/models/Spark-TTS-0.5B \
    SPARK_TTS_VOICE_DIR=/data/voices \
    SPARK_TTS_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch wheels explicitly so the image does not pull CUDA
# runtime packages. The remaining requirements are installed separately.
COPY requirements.txt /tmp/spark-tts-requirements.txt
RUN grep -Ev '^(torch|torchaudio)==' /tmp/spark-tts-requirements.txt \
        > /tmp/spark-tts-runtime-requirements.txt \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
        torchaudio==2.5.1 \
    && python -m pip install \
        -r /tmp/spark-tts-runtime-requirements.txt \
    && rm /tmp/spark-tts-requirements.txt \
        /tmp/spark-tts-runtime-requirements.txt

COPY api ./api
COPY cli ./cli
COPY sparktts ./sparktts

RUN mkdir -p /data/voices

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
