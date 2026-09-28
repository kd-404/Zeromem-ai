"""The "arrange" task format: what the scraper hands ZeroMem, and what it must output.

Input (built by code from scraped, already-chunked text):
    Question: <q>\n[1] <chunk 1 text>\n<CHUNK_SEP>[2] <chunk 2 text>\n<CHUNK_SEP> ... Answer:

Output (what ZeroMem is trained to produce):
    <KNOW>[n] <one sentence copied verbatim from chunk n><DONE>     answer is in the text
    <REFUSE><DONE>                                                  no chunk answers it

Why the chunk number comes FIRST: it splits the job into two small steps for a
tiny model, (1) point at a chunk, (2) copy a sentence out of it, instead of
copying first and having to remember where the words came from.

Citation rule when the same sentence appears in several chunks (overlapping
chunks make this common): cite the LOWEST-numbered chunk containing it. One
fixed rule keeps the labels consistent.

The one hard property: the quoted sentence is COPIED, never generated, so a
plain string search can verify it really appears in chunk [n].

Per-token chunk ids drive the chunk-isolation attention mask: tokens of
chunk i carry id i (>0) and can only attend within that chunk; the question,
"Answer:" and the output carry id 0 and see everything.

Scraped text is SANITIZED before tokenizing: any literal control-token text
("<REFUSE>", "<KNOW>", ...) found on a web page is stripped, otherwise a page
could inject real control tokens into the model's input.
"""

from __future__ import annotations

import re

from tokenizers import Tokenizer

from zeromem.tokenizer.special_tokens import CHUNK_SEP, PAD, SPECIAL_TOKENS

# The training data and the live scraper must chunk identically, or the model
# is trained on a different distribution than it sees at inference.
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30

REFUSE_TARGET = "<REFUSE><DONE>"

_CONTROL_RE = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))
_KNOW_RE = re.compile(r"^<KNOW>\[(\d+)\] (.*)<DONE>$", re.S)


def sanitize(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def format_know(chunk_number: int, sentence: str) -> str:
    return f"<KNOW>[{chunk_number}] {sentence}<DONE>"


def parse_target(text: str) -> dict | None:
    """Parse a model output. Returns {"verdict": "KNOW", "chunk": n, "quote": s},
    {"verdict": "REFUSE"}, or None if it doesn't match the format."""
    text = text.strip()
    if text == REFUSE_TARGET:
        return {"verdict": "REFUSE"}
    m = _KNOW_RE.match(text)
    if m:
        return {"verdict": "KNOW", "chunk": int(m.group(1)), "quote": m.group(2)}
    return None


def encode_prompt(tokenizer: Tokenizer, chunks: list[str], question: str) -> tuple[list[int], list[int]]:
    """Returns (ids, chunk_ids) for the prompt part only (chunks + question + 'Answer:')."""
    # Question FIRST (global, id 0): every chunk can then attend to it, so each chunk's
    # representation is built knowing what is being asked (Fusion-in-Decoder style),
    # while the mask still stops chunks seeing each other. With the question last
    # (v1) chunk tokens could not see it, and the model never learned to use the evidence.
    q = tokenizer.encode(f"Question: {sanitize(question).strip()}\n").ids
    ids: list[int] = list(q)
    chunk_ids: list[int] = [0] * len(q)
    for i, chunk in enumerate(chunks, start=1):
        seg = tokenizer.encode(f"[{i}] {sanitize(chunk).strip()}\n").ids + [CHUNK_SEP]
        ids += seg
        chunk_ids += [i] * len(seg)  # the separator belongs to the chunk it closes
    tail = tokenizer.encode("Answer:").ids
    ids += tail
    chunk_ids += [0] * len(tail)
    return ids, chunk_ids


def encode_example(tokenizer: Tokenizer, chunks: list[str], question: str, target: str):
    """Returns (ids, labels, chunk_ids) for training. Loss only on the target."""
    p_ids, p_chunks = encode_prompt(tokenizer, chunks, question)
    t_ids = tokenizer.encode(target.strip()).ids  # no leading space: it would become a stray token next to <KNOW>/<REFUSE>
    ids = p_ids + t_ids
    labels = [PAD] * len(p_ids) + t_ids
    chunk_ids = p_chunks + [0] * len(t_ids)
    return ids, labels, chunk_ids
