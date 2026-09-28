# ZeroMem "lite" for Render's free plan (512 MB RAM, 0.1 CPU).
# ZeroMem is the only model; weights are downloaded from Hugging Face at build time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
WORKDIR /app

COPY deploy/render/requirements.txt /tmp/requirements.txt
# CPU-only torch (~200 MB) instead of the default CUDA build (~2 GB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt

COPY zeromem ./zeromem
COPY deploy/render/fetch_weights.py ./fetch_weights.py

# Render passes the service's environment variables to the build as build args
ARG WEIGHTS_REPO=balaji9043/zeromem-pointer
ARG HF_TOKEN=
RUN WEIGHTS_REPO="$WEIGHTS_REPO" HF_TOKEN="$HF_TOKEN" python fetch_weights.py

# Render sets PORT; chat_app reads it
CMD ["python", "-m", "zeromem.chat_app", "--lite"]
