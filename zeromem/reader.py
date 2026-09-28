"""ZeroMem as a reader: give it one chunk and a question, get back a VERIFIED result.

ZeroMem outputs either
    <KNOW>[1] <sentence><DONE>     "this chunk answers it, here is the sentence"
    <REFUSE><DONE>                 "this chunk does not answer it"

The reader never trusts a quote on its face. The quoted sentence must appear verbatim in
the chunk it was read from (whitespace-normalised). A quote that isn't there is what a
hallucination looks like, so it is REJECTED and reported as such; it never reaches the user.
This is what keeps citations checkable even while the model is still weak.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from zeromem.data.arrange_format import encode_prompt
from zeromem.tokenizer.special_tokens import DONE
from zeromem.train.generate import load_model

DEFAULT_CKPT = "checkpoints/arrange_1chunk_rp/best.pt"  # after reading practice
DEFAULT_TOKENIZER = "zeromem/tokenizer/trained/tokenizer.json"

_KNOW_RE = re.compile(r"\s*<KNOW>\s*\[(\d+)\]\s*(.*?)\s*(?:<DONE>)?\s*$", re.S)


def norm(s: str) -> str:
    return " ".join(s.split())


@dataclass
class ReadResult:
    verdict: str            # "KNOW" | "REFUSE" | "MALFORMED"
    quote: str | None       # the sentence the model quoted (KNOW only)
    verified: bool          # True only if that quote is verbatim inside the chunk
    raw: str                # the model's raw output, for debugging
    seconds: float
    pick: str | None = None     # pointer reader: the sentence letter it chose
    prob: float | None = None   # pointer reader: P(answer) = softmax over <KNOW>/<REFUSE>


def pick_device(name: str | None = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ZeroMemReader:
    def __init__(self, ckpt: str = DEFAULT_CKPT, tokenizer: str = DEFAULT_TOKENIZER, device: str | None = None):
        self.device = pick_device(device)
        self.tok = Tokenizer.from_file(tokenizer)
        self.model = load_model(ckpt, self.device)

    @torch.no_grad()
    def read(self, question: str, chunk: str, max_new: int = 90) -> ReadResult:
        t0 = time.time()
        ids, cids = encode_prompt(self.tok, [chunk], question)
        n_prompt = len(ids)
        for _ in range(max_new):
            x = torch.tensor([ids], device=self.device)
            c = torch.tensor([cids], device=self.device)
            logits, _ = self.model(x, chunk_ids=c)
            nxt = int(logits[0, -1].argmax())  # greedy: same input always gives the same output
            ids.append(nxt)
            cids.append(0)
            if nxt == DONE:
                break
        finished = ids[-1] == DONE
        raw = self.tok.decode(ids[n_prompt:], skip_special_tokens=False)
        secs = time.time() - t0

        if raw.strip().startswith("<REFUSE>"):
            return ReadResult("REFUSE", None, False, raw, secs)
        m = _KNOW_RE.match(raw)
        if m and m.group(2).strip():
            quote = norm(m.group(2))
            if not finished:
                # Generation hit the length limit mid-sentence: keep only complete sentences.
                ends = list(re.finditer(r"[.!?][\"')\]]*(?:\[\d+\])*", quote))
                if ends:
                    quote = quote[: ends[-1].end()]
            verified = len(quote.split()) >= 3 and quote in norm(chunk)
            return ReadResult("KNOW", quote, verified, raw, secs)
        return ReadResult("MALFORMED", None, False, raw, secs)


# ---------------------------------------------------------------------------------------------
# Pointer reader: ZeroMem POINTS at a sentence ([A], [B], ...) instead of retyping it.
# Format and training: zeromem/data/pointer_format.py, zeromem/train/finetune_pointer.py.
# ---------------------------------------------------------------------------------------------
POINTER_CKPT = "checkpoints/pointer/best.pt"


class PointerReader:
    """Reads (question, chunk) and returns KNOW + the chosen sentence, or REFUSE.

    One forward pass per batch of chunks: the prompt is followed by '<KNOW> [' so the logits
    at 'Answer:' give the verdict and the logits after '[' give the letter, both at once.
    Letters beyond the chunk's last sentence are masked out, so the pick is always a real
    sentence of that chunk: verified = True by construction."""

    def __init__(self, ckpt: str = POINTER_CKPT, tokenizer: str = DEFAULT_TOKENIZER, device: str | None = None,
                 min_know: float = 0.5):
        from zeromem.data.pointer_format import MARKERS
        from zeromem.tokenizer.special_tokens import KNOW, PAD, REFUSE
        self.device = pick_device(device)
        self.tok = Tokenizer.from_file(tokenizer)
        self.model = load_model(ckpt, self.device)
        self.KNOW, self.REFUSE, self.PAD = KNOW, REFUSE, PAD
        self.LB = self.tok.token_to_id("[")
        self.mids = [self.tok.token_to_id(m) for m in MARKERS]
        self.min_know = min_know  # answer when P(KNOW) >= this (0.5 = plain argmax)

    def read(self, question: str, chunk: str, max_new: int = 0) -> ReadResult:
        return self.read_many(question, [chunk])[0]

    def read_many(self, question: str, chunks: list[str]) -> list[ReadResult]:
        from zeromem.data.pointer_format import split_sentences
        return self.read_sentences(question, [split_sentences(c) for c in chunks])

    @torch.no_grad()
    def read_sentences(self, question: str, sentence_lists: list[list[str]]) -> list[ReadResult]:
        """Core: one batched pass over chunks that are already split into sentences."""
        from zeromem.data.pointer_format import MARKERS, MAX_SENTENCES, render_chunk
        t0 = time.time()
        rows = []
        for sents in sentence_lists:
            sents = sents[:MAX_SENTENCES]
            ids, cids = encode_prompt(self.tok, [render_chunk(sents)], question) if sents else ([], [])
            rows.append((sents, ids, cids))
        todo = [i for i, r in enumerate(rows) if r[0]]
        results: list[ReadResult | None] = [None] * len(rows)
        if todo:
            L = max(len(rows[i][1]) for i in todo) + 2
            x = torch.full((len(todo), L), self.PAD, dtype=torch.long)
            c = torch.zeros((len(todo), L), dtype=torch.long)
            for b, i in enumerate(todo):
                _, ids, cids = rows[i]
                seq = ids + [self.KNOW, self.LB]  # right-padded; causal attention ignores what comes after
                x[b, :len(seq)] = torch.tensor(seq)
                c[b, :len(cids)] = torch.tensor(cids)
            logits, _ = self.model(x.to(self.device), chunk_ids=c.to(self.device))
            logits = logits.float().cpu()
            secs = (time.time() - t0) / len(todo)
            for b, i in enumerate(todo):
                sents, ids, _ = rows[i]
                n = len(ids)
                v = torch.softmax(logits[b, n - 1, [self.KNOW, self.REFUSE]], dim=-1)
                p_know = float(v[0])
                letters = torch.softmax(logits[b, n + 1, self.mids[:len(sents)]], dim=-1)
                k = int(torch.argmax(letters))
                if p_know < self.min_know:
                    results[i] = ReadResult("REFUSE", None, False, "<REFUSE><DONE>", secs, None, p_know)
                else:
                    results[i] = ReadResult("KNOW", sents[k], True, f"<KNOW>[{MARKERS[k]}]<DONE>", secs,
                                            MARKERS[k], p_know)
        for i, r in enumerate(results):
            if r is None:  # nothing readable in the chunk
                results[i] = ReadResult("REFUSE", None, False, "<REFUSE><DONE> (no sentences)", 0.0, None, 0.0)
        return results  # type: ignore[return-value]


def make_reader(kind: str = "auto", ckpt: str | None = None, device: str | None = None):
    """copy: the old retyping reader. pointer: the pointer reader.
    auto: pointer if --ckpt is a pointer checkpoint (path contains 'pointer'), or, with no
    --ckpt, if checkpoints/pointer/best.pt exists; otherwise the copy reader."""
    import os
    if kind == "copy":
        return ZeroMemReader(ckpt or DEFAULT_CKPT, device=device)
    if kind == "pointer":
        return PointerReader(ckpt or POINTER_CKPT, device=device)
    if ckpt:
        return PointerReader(ckpt, device=device) if "pointer" in ckpt else ZeroMemReader(ckpt, device=device)
    if os.path.exists(POINTER_CKPT):
        return PointerReader(POINTER_CKPT, device=device)
    return ZeroMemReader(DEFAULT_CKPT, device=device)
