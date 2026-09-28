# ZeroMem on Hugging Face Spaces (Docker SDK, CPU basic: 2 vCPU / 16 GB RAM, free).
# Built by deploy/hf_space/deploy.py, which copies the code, tokenizer and pointer weights in.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf-cache
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
# CPU-only torch (~200 MB) instead of the default CUDA build (~2 GB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt
# bake the reranker into the image so the first visitor doesn't wait for a download
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && chown -R user /app

COPY --chown=user . .
USER user

ENV MIN_KNOW=__MIN_KNOW__ REPO_URL=__REPO_URL__
EXPOSE 7860
CMD ["python", "-m", "zeromem.chat_app", "--public", "--port", "7860"]
