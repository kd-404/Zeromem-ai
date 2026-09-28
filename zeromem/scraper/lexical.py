"""BM25 keyword ranker: a no-model stand-in for the cross-encoder, for small servers.

Same call shape as sentence_transformers.CrossEncoder.predict(pairs) so the pipeline can use
either. It only ORDERS chunks for ZeroMem to read; ZeroMem still makes every decision. Scores
are BM25 (not comparable to cross-encoder scores), so the relevance thresholds of the full
system don't apply: use it with --zeromem-only / --lite.

No torch, no downloads, ~0 MB: this is what lets ZeroMem run in 512 MB (Render free tier).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the is are was were be been being of to in on at by for with from as and or but not no
it its this that these those what which who whom whose when where why how do does did can could should would
will shall may might must i you he she they we me him her them us my your his their our about into than then
there here also just only very so such any some more most other""".split())


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


class LexicalRanker:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def predict(self, pairs, show_progress_bar: bool = False, **_) -> list[float]:
        """pairs: [(question, text), ...]. IDF is computed over the texts in this call (the
        chunks retrieved for this question), which is what matters for ordering them."""
        if not pairs:
            return []
        docs = [Counter(_terms(t)) for _, t in pairs]
        lens = [sum(d.values()) for d in docs]
        avg = sum(lens) / len(lens) or 1.0
        n = len(docs)
        df = Counter(w for d in docs for w in d)
        scores = []
        for (q, _), d, L in zip(pairs, docs, lens):
            s = 0.0
            for w in set(_terms(q)):
                f = d.get(w, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / avg))
            scores.append(s)
        return scores
