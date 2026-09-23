"""Web search + fetch + clean, feeding the semantic cache.

Flow: Brave Search API (finds candidate URLs) -> trafilatura (extracts
clean article text, stripping nav/ads/boilerplate) -> chunker (splits into
retrieval-sized pieces). The caller is responsible for embedding and
storing the resulting chunks (see zeromem/cache/vector_store.py) — this
module only ever produces chunks, it never touches the model or its
weights (see BLUEPRINT.md's hard rule: facts enter via context, not
training).

Requires a Brave Search API key (free tier): https://brave.com/search/api/
Set it as the BRAVE_API_KEY environment variable.
"""

from __future__ import annotations

import os

import requests
import trafilatura

from zeromem.scraper.chunker import Chunk, chunk_text

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def brave_search(query: str, api_key: str | None = None, count: int = 5) -> list[str]:
    """Return a list of candidate result URLs for a query."""
    api_key = api_key or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Brave Search API key found. Set BRAVE_API_KEY in your environment "
            "(get a free-tier key at https://brave.com/search/api/)."
        )

    resp = requests.get(
        BRAVE_SEARCH_ENDPOINT,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query, "count": count},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("web", {}).get("results", [])
    return [r["url"] for r in results if "url" in r]


def fetch_clean_text(url: str) -> str | None:
    """Download a page and strip it to clean article text. Returns None on
    any failure (dead link, paywall, non-HTML content, etc.) — the caller
    should skip and move to the next URL rather than crash the pipeline."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        return trafilatura.extract(downloaded)
    except Exception:
        return None


def search_and_chunk(
    query: str,
    api_key: str | None = None,
    result_count: int = 5,
    chunk_size_words: int = 200,
    overlap_words: int = 40,
) -> list[Chunk]:
    """End-to-end: search -> fetch -> clean -> chunk. Does NOT embed or
    cache — that's the vector_store layer's job, kept separate so this
    module stays testable without a running ChromaDB instance."""
    urls = brave_search(query, api_key=api_key, count=result_count)

    all_chunks: list[Chunk] = []
    for url in urls:
        text = fetch_clean_text(url)
        if not text:
            continue
        all_chunks.extend(chunk_text(text, source_url=url,
                                      chunk_size_words=chunk_size_words,
                                      overlap_words=overlap_words))
    return all_chunks
