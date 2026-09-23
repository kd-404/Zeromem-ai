"""Turns raw text + a trained tokenizer into fixed-length training windows.

Kept intentionally simple: concatenate all documents (separated by <EOS>),
tokenize once, then slice into non-overlapping blocks of `seq_len` tokens.
This is the standard nanoGPT-style approach — no padding needed since
every block is full, which keeps stage-1 pretraining simple. Padding
becomes necessary in stage 2, once examples are structured (question +
chunks + answer) rather than raw concatenated text.
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from zeromem.tokenizer.special_tokens import EOS


def _iter_documents(text_path: str, read_chunk_bytes: int = 8 * 1024 * 1024) -> Iterator[str]:
    """Yields documents (blank-line-separated blocks) without ever holding
    the full file content in memory at once — reads in fixed-size chunks
    and only keeps the trailing partial document across reads."""
    buf = ""
    with open(text_path, "r", encoding="utf-8") as f:
        while True:
            piece = f.read(read_chunk_bytes)
            if not piece:
                break
            buf += piece
            parts = buf.split("\n\n")
            buf = parts.pop()  # last piece may be incomplete, carry to next read
            for p in parts:
                p = p.strip()
                if p:
                    yield p
    tail = buf.strip()
    if tail:
        yield tail


def tokenize_corpus_to_ids(text_path: str, tokenizer_path: str, batch_docs: int = 20_000) -> np.ndarray:
    """Reads a raw text file (one document per blank-line-separated block,
    as written by download_tinystories.py) and returns a single flat
    array of token ids with <EOS> inserted between documents.

    Streams the whole way through: documents are read incrementally
    (never the full file in memory), tokenized in small batches, and
    written straight to a temp binary file on disk rather than
    accumulated in a Python list or a growing list of numpy arrays.

    This matters concretely on an 8GB machine: an earlier version that
    built one Python list of ~466M ints got silently OOM-killed (no
    traceback possible — the OS just kills the process), and a version
    that batched into numpy arrays but still kept every batch in memory
    projected to ~8.5GB peak on the full corpus — measured by running it
    on a 500MB slice (2.6GB peak) and scaling. This version's peak memory
    is bounded by one batch of documents (tens of MB) regardless of
    total corpus size.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    file_size = os.path.getsize(text_path)
    tmp_path = text_path + ".tokenizing.tmp.bin"

    eos_bytes = np.array([EOS], dtype=np.uint16).tobytes()
    total_tokens = 0
    docs_done = 0

    with open(tmp_path, "wb") as out_f:
        batch: list[str] = []

        def flush_batch() -> None:
            nonlocal total_tokens, docs_done
            if not batch:
                return
            encodings = tokenizer.encode_batch(batch)
            for enc in encodings:
                ids_arr = np.array(enc.ids, dtype=np.uint16)
                out_f.write(ids_arr.tobytes())
                out_f.write(eos_bytes)
                total_tokens += len(ids_arr) + 1
            docs_done += len(batch)
            batch.clear()

        for doc in _iter_documents(text_path):
            batch.append(doc)
            if len(batch) >= batch_docs:
                flush_batch()
                pct = 100.0 * out_f.tell() / max(file_size, 1)  # approximate: bytes written vs input size
                print(f"  tokenized {docs_done:,} documents, {total_tokens:,} tokens so far "
                      f"(~{pct:.0f}% through)", flush=True)
        flush_batch()

    print(f"Tokenizing done: {docs_done:,} documents, {total_tokens:,} tokens total.", flush=True)

    token_ids = np.fromfile(tmp_path, dtype=np.uint16)
    os.remove(tmp_path)
    return token_ids


class BlockDataset(Dataset):
    """Slices a flat token-id array into fixed-length (input, target) pairs
    for next-token-prediction pretraining."""

    def __init__(self, token_ids: np.ndarray, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len
        # Each usable sample needs seq_len+1 tokens (inputs + shifted targets).
        self.n_samples = (len(token_ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.token_ids[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
