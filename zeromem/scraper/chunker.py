"""Split fetched page text into retrieval-sized chunks.

Kept deliberately simple (word-count windows with overlap) for v1 — a
sentence-boundary-aware splitter is a easy upgrade later, but a naive
splitter is enough to validate the rest of the pipeline first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source_url: str
    chunk_index: int  # position of this chunk within its source document


def chunk_text(
    text: str,
    source_url: str,
    chunk_size_words: int = 200,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Sliding-window chunking by word count.

    Overlap keeps a sentence that straddles a chunk boundary from having
    its meaning split across two chunks with no shared context.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(chunk_size_words - overlap_words, 1)

    while start < len(words):
        window = words[start:start + chunk_size_words]
        chunk_str = " ".join(window)
        chunks.append(Chunk(text=chunk_str, source_url=source_url, chunk_index=index))
        index += 1
        start += step

    return chunks
