"""A small web UI to test the ZeroMem pipeline interactively.

The Pipeline is built once at startup (its reader/reranker/cache load lazily on first use,
same as the CLI). Every /ask call reuses it, so only the first question pays model-loading cost.

Run:
    cd /Users/balajikd/projects/zeromem-ai && source .venv/bin/activate
    uvicorn zeromem.api.server:app --reload --port 8000
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from zeromem.pipeline import Pipeline

app = FastAPI(title="ZeroMem")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# verbose=False: the UI shows structured JSON, not the CLI's printed step-by-step log.
pipeline = Pipeline(verbose=False)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    ans = pipeline.ask(req.question)
    return {
        "question": ans.question,
        "answered": ans.answered,
        "text": ans.text,
        "source_url": ans.source_url,
        "route": ans.route,
        "confidence": ans.confidence,
        "picked_by": ans.picked_by,
        "relevance": ans.relevance,
        "smalltalk": ans.smalltalk,
        "fell_back": ans.fell_back,
        "resolved": ans.resolved,
        "continued": ans.continued,
        "seconds": round(ans.seconds, 1),
        "considered": [
            {"url": u, "rerank_score": round(s, 2), "verdict": v}
            for u, s, v in ans.considered
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True, "cache_size": pipeline.cache.size() if pipeline._cache else 0}
