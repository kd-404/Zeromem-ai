"""The POINTER task format: ZeroMem points at a sentence instead of retyping it.

Why: in the copy format (arrange_format.py) ZeroMem had to regenerate the answer token by
token. It usually found the right sentence but corrupted it while copying ("cooler" ->
"chickens", byte-split words -> "Bir�", losing its place). Here the model only makes
decisions; code does the copying, so every shown answer is real by construction.

Input: the chunk is split into sentences, each tagged with a letter marker:
    Question: <q>\n[1] [A] <sentence> [B] <sentence> ... [Z] <sentence>\n<CHUNK_SEP>Answer:

Output (5 tokens at most):
    <KNOW>[C]<DONE>        sentence C answers the question
    <REFUSE><DONE>         nothing in the chunk answers it

Letters, not numbers: in this tokenizer 12..30 are two tokens ("1","2"), but A..Z are each
one token, so pointing is ONE prediction. Up to 26 sentences per chunk (a 120-word chunk
rarely has more than 12). Reserved for later: ranges like <KNOW>[C-F]<DONE> for steps.

The same split_sentences() runs in training and at inference, so the letters always match.
"""

from __future__ import annotations

import re

MARKERS = [chr(ord("A") + i) for i in range(26)]
MAX_SENTENCES = len(MARKERS)
REFUSE_TARGET = "<REFUSE><DONE>"

SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")  # same rule as build_arrange_dataset
_MARKER_LIKE = re.compile(r"\[[A-Z](?:-[A-Z])?\]")  # page text that looks like our markers
_POINTER_RE = re.compile(r"^\s*<KNOW>\s*\[([A-Z])(?:-([A-Z]))?\]\s*(?:<DONE>)?\s*$")


def norm(s: str) -> str:
    return " ".join(s.split())


def split_sentences(chunk: str) -> list[str]:
    """Chunk -> sentences, in order. Lines are split too (scraped pages put menus, headings
    and list items on their own lines). Tiny pieces (<= 3 chars) are dropped."""
    out: list[str] = []
    for line in chunk.split("\n"):
        starts = [0] + [m.end() for m in SENT_SPLIT.finditer(line)]
        ends = starts[1:] + [len(line)]
        for a, b in zip(starts, ends):
            s = norm(line[a:b])
            if len(s) > 3:
                out.append(s)
    return out[:MAX_SENTENCES]


def render_chunk(sentences: list[str]) -> str:
    """Sentences with their letter markers, as the model reads them. Marker-looking text
    from the page is removed so a page can't fake a marker."""
    return " ".join(f"[{MARKERS[i]}] {_MARKER_LIKE.sub('', s).strip()}" for i, s in enumerate(sentences))


def pointer_target(index: int | None) -> str:
    return REFUSE_TARGET if index is None else f"<KNOW>[{MARKERS[index]}]<DONE>"


def parse_pointer(text: str) -> dict | None:
    """'<KNOW>[C]<DONE>' -> {'verdict': 'KNOW', 'start': 2, 'end': 2}; '<REFUSE><DONE>' -> REFUSE."""
    t = text.strip()
    if t.startswith("<REFUSE>"):
        return {"verdict": "REFUSE"}
    m = _POINTER_RE.match(t)
    if not m:
        return None
    a = ord(m.group(1)) - ord("A")
    b = ord(m.group(2)) - ord("A") if m.group(2) else a
    return {"verdict": "KNOW", "start": min(a, b), "end": max(a, b)}


def find_gold(sentences: list[str], gold: str) -> int | None:
    """Index of the sentence that IS the gold answer. Exact match first; else the sentence
    that contains it / is contained in it with the biggest word overlap (the old dataset
    split some lines differently)."""
    g = norm(gold)
    for i, s in enumerate(sentences):
        if s == g:
            return i
    best, best_score = None, 0.0
    gw = set(g.lower().split())
    for i, s in enumerate(sentences):
        if g in s or s in g:
            sw = set(s.lower().split())
            score = len(gw & sw) / max(len(gw | sw), 1)
            if score > best_score:
                best, best_score = i, score
    return best if best_score >= 0.6 else None
