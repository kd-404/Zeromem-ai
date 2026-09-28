"""The semantic cache: ZeroMem's growing local knowledge base.

This is the "save it and reuse it" layer from the design conversation:
  - Every fetched-and-chunked page gets embedded and stored here PERMANENTLY.
  - Every new query is checked against this store FIRST, by embedding
    similarity, before hitting the internet again.
  - A cache hit skips the web entirely: faster, works offline for repeat
    topics, and doesn't burn Brave Search API quota.
  - A cache miss falls through to zeromem/scraper/fetch.py, and whatever
    comes back gets stored here too, so the store only ever grows.

Hard rule (see BLUEPRINT.md): this store holds facts as retrievable text.
It is never used to construct training labels for weight updates in an
automated way — stage-2 fine-tuning data gets pulled from here manually,
with a human review/filter step in between.
"""

from __future__ import annotations

import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "zeromem_chunks"
DEFAULT_DB_PATH = "chroma_db"

# Below this cosine similarity, treat it as a cache miss and go fetch fresh
# content instead of reusing a loosely-related cached chunk.
DEFAULT_SIMILARITY_THRESHOLD = 0.75


class SemanticCache:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, embedding_model: str = EMBEDDING_MODEL_NAME):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(embedding_model)

    @staticmethod
    def _chunk_id(source_url: str, chunk_index: int, text: str) -> str:
        # Stable id so re-fetching the same page's same chunk overwrites
        # rather than duplicates, but a changed page produces a new id.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{source_url}::{chunk_index}::{digest}"

    def store_chunks(self, chunks: list) -> int:
        """chunks: list of zeromem.scraper.chunker.Chunk. Returns count stored."""
        if not chunks:
            return 0

        ids = [self._chunk_id(c.source_url, c.chunk_index, c.text) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{"source_url": c.source_url, "chunk_index": c.chunk_index} for c in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()

        self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Raw similarity search — returns whatever's in the store, doesn't
        apply the cache-hit threshold itself. Use `lookup()` for the
        cache-or-miss decision."""
        embedding = self.embedder.encode([query_text], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=top_k)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            similarity = 1 - dist  # chromadb cosine space returns distance = 1 - similarity
            hits.append({"text": doc, "source_url": meta["source_url"], "similarity": similarity,
                         "chunk_index": meta.get("chunk_index", -1)})
        return hits

    def lookup(
        self, query_text: str, top_k: int = 5, threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[dict] | None:
        """The actual cache decision: return cached chunks if we have
        anything similar enough, else None (signals a cache miss -> go
        fetch fresh content via zeromem/scraper/fetch.py)."""
        if self.collection.count() == 0:
            return None

        hits = self.query(query_text, top_k=top_k)
        good_hits = [h for h in hits if h["similarity"] >= threshold]
        return good_hits or None

    def page_chunks(self, source_url: str) -> list[tuple[int, str]]:
        """Every stored chunk of one page as (chunk_index, text), in page order. Used to show a
        longer passage (recipes, steps) or continue reading after an answer."""
        got = self.collection.get(where={"source_url": source_url}, include=["documents", "metadatas"])
        rows = {(m.get("chunk_index", -1), d) for d, m in zip(got["documents"], got["metadatas"])}
        return sorted(rows)

    def size(self) -> int:
        return self.collection.count()
