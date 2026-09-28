"""The ZeroMem pipeline: question -> web -> verified, cited answer. No LLM brain.

    question
      1. cache      look for chunks we already fetched (bge-small + ChromaDB)
      2. route      pick the source for this kind of question (scraper/router.py):
                    facts -> Wikipedia API, code -> docs/Q&A sites, news -> trusted news sites,
                    everything else -> open web. Only on a cache miss.
      3. search     get pages from that source (web search is ddgs / brave / tavily / exa),
                    fetch in parallel, clean with trafilatura, cut into 120-word chunks
      4. store      embed and save new chunks in the cache (permanent, grows over time)
      5. rerank     cross-encoder picks the top-k chunks for this question
      6. read       ZeroMem reads each top chunk in order; first VERIFIED quote wins
      7. answer     the quote + its source URL, or "no verified answer"

    If a routed source (wiki/code/news) returns nothing, or ZeroMem finds no verified answer in
    it, steps 3-6 run once more on the open web. A wrong route costs time, never the answer.

Hard rules (BLUEPRINT.md): scraped text only ever enters through the context window; it
never trains the weights. And a quote is only shown if it is verbatim inside its chunk.

Usage:
    python -m zeromem.pipeline "Who designed the Eiffel Tower?"
    python -m zeromem.pipeline --provider tavily --k 5 "your question"
    python -m zeromem.pipeline --route web "..."   # old behaviour: open web only
    python -m zeromem.pipeline --route wiki "..."  # force one source
    python -m zeromem.pipeline            # interactive
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from zeromem.scraper.chunker import Chunk
from zeromem.scraper.fetch import CODE_INDEX, fetch_and_chunk, is_code_chunk
from zeromem.scraper.router import ROUTES, Route, forced_route, on_allowed_site, route_question, site_query
from zeromem.scraper.search import PROVIDERS, search
from zeromem.scraper.wikipedia import wiki_chunks

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_GREETING = re.compile(r"^(hi+|hello+|hey+|yo|hola|namaste|good (morning|afternoon|evening)|sup|thanks?( you)?|thank you|ok(ay)?|bye|goodbye)$")
_ABOUT_ME = re.compile(r"^(who|what) (are|r) (you|u|zeromem)|^what can you do|^help$|^what is zeromem")
INTRO = ("I'm ZeroMem, a small language model built from scratch. I don't answer from memory: I search the web, "
         "read the pages, and quote a sentence from a source (with its link), or tell you I couldn't verify an answer. "
         "Ask me a factual question, like \"When was the Eiffel Tower built?\".")


def clean_question(q: str) -> str:
    """Trim stray characters people type by accident (trailing slashes, repeated punctuation)."""
    q = q.strip().strip("/\\|~`")
    return re.sub(r"[?!.]{2,}$", "?", q).strip()


def small_talk_reply(q: str) -> str | None:
    """A fixed system message for greetings and questions about ZeroMem itself, so they don't
    trigger a pointless web search. Not an answer to a factual question, so it is labelled as such."""
    t = re.sub(r"[^a-z' ]", "", q.lower()).strip()
    if not t or len(t) < 2:
        return "Please type a question."
    if _ABOUT_ME.match(t) or _GREETING.match(t):
        return INTRO
    return None
MAX_CHUNKS_PER_PAGE = 40
# Backup answers: when ZeroMem gives no verified answer, the best real sentence from the top chunks
# is still shown (labelled LOW CONFIDENCE) if the cross-encoder scores it above this. 0 is its
# relevant/irrelevant boundary, so a page that doesn't contain the answer still gives no answer.
SOFT_FLOOR = 0.0
_SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(])")


# How-to questions want steps, not one sentence: their answer is a passage from the page.
_HOWTO = re.compile(r"^how (to|do i|do you|can i|should i|would i|do we|can we|does one)\b|\b(recipe|recipes|steps|"
                    r"step by step|procedure|instructions|method to|process of|process to|guide to|guide for|how to|"
                    r"tips (to|for|on)|ways to|best way to|checklist|tutorial)\b", re.I)
# "next", "continue", "more": keep reading the previous answer's page, no new search.
_CONTINUE = re.compile(r"^(next( steps?| one| part)?|continue|go on|keep going|more|tell me more|and then|then|"
                       r"what next|what's next|whats next|after that|the rest|rest|and|ok next|okay next)$", re.I)
_PRONOUN = re.compile(r"\b(he|she|it|they|him|her|them|his|its|their|this|that|there)\b", re.I)
# "write a python code for X": the answer is a code block copied from a page, not a sentence.
_WANTS_CODE = re.compile(r"\b(write|give|show|generate|create|make|need|want)\b.{0,40}\b(code|program|script|"
                         r"function|snippet|class|example)\b|\b(code|program|script|snippet|example) (for|to|of)\b|"
                         r"\bsample code\b|\bexample code\b|\bsource code\b|\bhow to (write|code|implement|program)\b",
                         re.I)
# A cache answer is trusted without searching again only if it is verified, or the reranker
# is sure the sentence is on topic. Otherwise the cache gave related-but-wrong pages (e.g.
# 'how to make biriyani' matched stored rasam pages), so search fresh.
_ASK_VERB = re.compile(r"^(please\s+)?(write|create|build|make|generate|implement|design|develop|give|show|code)\b", re.I)
_CODE_ARTIFACT = re.compile(r"\b(query|queries|regex|command|commands|snippet|one-liner) (to|for|that)\b", re.I)


def wants_code(question: str) -> bool:
    """True when the answer should be code, not prose: 'write a python code for...', 'sql query to...',
    'create a login page html css' (a request verb on a programming question)."""
    if _WANTS_CODE.search(question) or _CODE_ARTIFACT.search(question):
        return True
    from zeromem.scraper.router import route_question
    return bool(_ASK_VERB.match(question.strip())) and route_question(question).name == "code"


CACHE_TRUST = 3.0
PASSAGE_WORDS = 110        # how much a how-to answer / a "next" shows at a time


def is_continue(q: str) -> bool:
    return bool(_CONTINUE.match(re.sub(r"[^a-z' ]", "", q.lower()).strip()))


def subject_of(question: str) -> str:
    """The thing a question is about: 'who is rajinikanth' -> 'rajinikanth',
    'how to make rasam' -> 'rasam'."""
    from zeromem.scraper.router import search_query
    s = search_query(question)
    s = re.sub(r"^(how\s+)?(to\s+)?(make|cook|prepare|do|build|create|write|get|use|install|fix)\s+", "", s, flags=re.I)
    return s.strip(" ?.!") or question


def resolve_followup(question: str, last_question: str | None) -> str | None:
    """'when was he born' after 'who is rajinikanth' -> 'when was rajinikanth born'. Only short
    questions with a pronoun count, so a new full question is never rewritten."""
    if not last_question or len(question.split()) > 8 or not _PRONOUN.search(question):
        return None
    subj = subject_of(last_question)
    if not subj or subj.lower() in question.lower():
        return None
    first = [True]

    def swap(m):  # replace the first pronoun only, keep possessives readable
        if not first[0]:
            return m.group(0)
        first[0] = False
        return subj + ("'s" if m.group(1).lower() in ("his", "its", "their") else "")
    return _PRONOUN.sub(swap, question, count=1)


def merge_page(chunks: list[tuple[int, str]]) -> str:
    """Rebuild page text from its overlapping chunks (index order). Consecutive chunks share
    a word overlap, which is removed; a gap (missing chunk) is joined with ' ... '."""
    text_words: list[str] = []
    prev_idx = None
    for idx, t in chunks:
        w = t.split()
        if prev_idx is not None and idx == prev_idx + 1:
            k = next((k for k in range(min(60, len(w), len(text_words)), 0, -1) if text_words[-k:] == w[:k]), 0)
            text_words += w[k:]
        elif prev_idx is not None and idx == prev_idx:
            continue
        else:
            text_words += (["..."] if text_words else []) + w
        prev_idx = idx
    return " ".join(text_words)


def passage(page: str, start: int, words: int = PASSAGE_WORDS) -> tuple[str, int]:
    """Verbatim text from `start` up to ~`words` words, cut at a sentence end. Returns (text, end)."""
    ends = [m.end() for m in re.finditer(r"[.!?][\"')\]]*(?=\s|$)", page[start:])]
    if not ends:
        return page[start:].strip(), len(page)
    best = ends[0]
    for e in ends:
        if len(page[start:start + e].split()) > words:
            break
        best = e
    return page[start:start + best].strip(), start + best


def sentences(text: str) -> list[str]:
    """Whole sentences only: chunk edges cut sentences in half, so fragments are dropped."""
    out = []
    for s in _SENT.split(text):
        s = s.strip()
        if len(s.split()) >= 4 and s[0].isupper() and s[-1] in ".!?\"')":
            out.append(s)
    return out


@dataclass
class Answer:
    question: str
    answered: bool
    text: str | None = None          # the verified quote
    source_url: str | None = None
    considered: list[tuple[str, float, str]] = field(default_factory=list)  # (url, rerank score, ZeroMem verdict)
    seconds: float = 0.0
    smalltalk: bool = False          # True when a fixed system message was returned instead of searching
    route: str = ""                  # source the answer came from: cache | wiki | code | news | web
    confidence: str = ""             # "verified" (passed every check) | "low" (real sentence, weaker relevance)
    picked_by: str = ""              # "zeromem" | "reranker" (backup: best-scoring sentence when ZeroMem gave none)
    relevance: float | None = None   # cross-encoder score of the shown sentence vs the question
    resolved: str | None = None      # follow-up rewritten with the previous subject ("he" -> "Rajinikanth")
    continued: bool = False          # True for "next"/"more": the next part of the previous source
    readings: list[dict] = field(default_factory=list)  # zeromem-only mode: what ZeroMem wrote for each chunk
    fell_back: bool = False          # True when the routed source failed and the open web was used


class Pipeline:
    def __init__(self, provider: str = "ddgs", ckpt: str | None = None, device: str | None = None,
                 use_cache: bool = True, k: int = 3, cache_dir: str = "chroma_db",
                 cache_threshold: float = 0.70, min_relevance: float = 2.0, verbose: bool = True,
                 route: str = "auto", zeromem_only: bool = False, reader: str = "auto",
                 min_know: float = 0.5):
        self.min_know = min_know  # pointer reader: answer only when P(KNOW) >= this
        self.reader_kind = reader  # auto | pointer | copy (see reader.make_reader)
        # zeromem_only: TEST MODE. Scraped chunks go into ZeroMem and whatever it writes is the
        # answer: no reranker backup, no relevance bar, no verbatim gate, no passage/code
        # extraction. Whether the quote is really in the page is still MEASURED and shown, but it
        # never blocks the answer. Use it to see what the model itself can do.
        self.zeromem_only = zeromem_only
        if route != "auto" and route not in ROUTES:
            raise ValueError(f"route must be 'auto' or one of {ROUTES}")
        self.provider, self.k, self.use_cache, self.route = provider, k, use_cache, route
        self.cache_threshold, self.verbose = cache_threshold, verbose
        # A quote must be real (verbatim in its chunk) AND relevant to the question. The second
        # check exists because ZeroMem can copy a real sentence that does not answer anything.
        # Calibrated on held-out data (AUC 0.95): score >= 0 keeps 84% of correct sentences and lets through
        # 8% of wrong ones; score >= 2 keeps 78% and lets through 3%. We use 2: a missed answer costs less
        # than a confident wrong one (a +0.1 sentence about 'inscriptions' was shown for a 'when' question at 0).
        self.min_relevance = min_relevance
        self._ckpt, self._device, self._cache_dir = ckpt, device, cache_dir
        self._reader = self._reranker = self._cache = None
        self.last: dict | None = None  # previous answer: question, url, page text, where it ended

    # -- lazy loading: don't pay for a model until a step needs it ----------
    @property
    def reader(self):
        if self._reader is None:
            from zeromem.reader import make_reader
            self._reader = make_reader(self.reader_kind, self._ckpt, self._device, self.min_know)
            self._say(f"      reader: {type(self._reader).__name__}")
        return self._reader

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER, device="cpu")
        return self._reranker

    @property
    def cache(self):
        if self._cache is None:
            from zeromem.cache.vector_store import SemanticCache
            self._cache = SemanticCache(self._cache_dir)
        return self._cache

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- the pipeline --------------------------------------------------------
    def _web_pages(self, query: str, sites: tuple[str, ...] = ()) -> list[Chunk]:
        """Search -> fetch -> chunk. With `sites`, only pages on those domains are kept."""
        t = time.time()
        hits = search(site_query(query, sites) if sites else query, self.provider, n=10)
        if sites:
            hits = [h for h in hits if on_allowed_site(h.url, sites)]
        self._say(f"      search ({self.provider}{', allowed sites only' if sites else ''}): "
                  f"{len(hits)} results ({time.time() - t:.1f}s)")
        if not hits:
            return []
        t = time.time()
        chunks, ok, reasons = fetch_and_chunk([h.url for h in hits])
        per_page: dict[str, int] = {}
        kept = []
        for c in chunks:  # cap per page so one huge site can't crowd out the rest
            per_page[c.source_url] = per_page.get(c.source_url, 0) + 1
            if per_page[c.source_url] <= MAX_CHUNKS_PER_PAGE:
                kept.append(c)
        why = ", ".join(f"{n}x {r}" for r, n in reasons.most_common())
        self._say(f"      scraped {ok}/{len(hits)} pages -> {len(kept)} chunks ({time.time() - t:.1f}s)"
                  + (f"  [skipped: {why}]" if why else ""))
        return kept

    def _retrieve(self, question: str, route: Route) -> list[Chunk]:
        """Get chunks from the source this route points at."""
        if route.name == "wiki":
            t = time.time()
            chunks, ok, reasons = wiki_chunks(route.query)
            why = ", ".join(f"{n}x {r}" for r, n in reasons.most_common())
            self._say(f"      wikipedia ({route.query!r}): {ok} articles -> {len(chunks)} chunks ({time.time() - t:.1f}s)"
                      + (f"  [{why}]" if why else ""))
            return chunks
        return self._web_pages(route.query, route.sites)

    def _read(self, question: str, chunks: list[Chunk], considered: list) -> dict | None:
        """Rerank, then ZeroMem reads the top-k chunks in order.

        Returns the first quote that is verbatim in its chunk AND scores >= min_relevance
        (confidence "verified"). If there is none, returns the best REAL sentence from those
        chunks that scores above SOFT_FLOOR (confidence "low"): either a quote ZeroMem made that
        missed the bar, or, if ZeroMem refused, the reranker's own top sentence. Every shown
        sentence is still copied verbatim from a source, so nothing is ever made up."""
        t = time.time()
        if wants_code(question):
            codes = [c for c in chunks if is_code_chunk(c.text)]
            if codes:
                cs = self.reranker.predict([(question, c.text) for c in codes], show_progress_bar=False)
                best = max(range(len(codes)), key=lambda j: cs[j])
                self._say(f"[4/5] code question: scored {len(codes)} code blocks, best {float(cs[best]):+.1f} "
                          f"({urlparse(codes[best].source_url).netloc}) ({time.time() - t:.1f}s)")
                if cs[best] > SOFT_FLOOR:
                    return {"text": codes[best].text, "url": codes[best].source_url, "confidence": "low",
                            "picked_by": "reranker", "relevance": float(cs[best])}
            else:
                self._say("[4/5] code question, but these pages have no code blocks")
            chunks = [c for c in chunks if not is_code_chunk(c.text)] or chunks
            if not _HOWTO.search(question):
                return None  # a prose sentence is not an answer to 'write code for X'
        chunks = [c for c in chunks if not is_code_chunk(c.text)] or chunks
        scores = self.reranker.predict([(question, c.text) for c in chunks], show_progress_bar=False)
        top = sorted(zip(scores, chunks), key=lambda x: -x[0])[: self.k]
        self._say(f"[4/5] reranked {len(chunks)} chunks, reading the top {len(top)} ({time.time() - t:.1f}s)")
        backup: list[tuple[float, str, str, str]] = []  # (relevance, sentence, url, picked_by)
        reads = self._read_all(question, [c.text for _, c in top])
        for rank, ((score, chunk), r) in enumerate(zip(top, reads), 1):
            accepted = False
            if r.verdict != "KNOW":
                note = r.verdict
            elif not r.verified:
                note = "KNOW (REJECTED: quote not in chunk)"
            else:
                rel = float(self.reranker.predict([(question, r.quote)], show_progress_bar=False)[0])
                accepted = rel >= self.min_relevance
                note = (f"KNOW (verified, relevance {rel:+.1f})" if accepted
                        else f"KNOW (relevance {rel:+.1f} is below {self.min_relevance:+.1f}, kept as backup)")
                if not accepted:
                    backup.append((rel, r.quote, chunk.source_url, "zeromem"))
            considered.append((chunk.source_url, float(score), note))
            self._say(f"   chunk {rank}: rerank {score:+.1f} | ZeroMem: {note} | {urlparse(chunk.source_url).netloc} ({r.seconds:.1f}s)")
            if accepted:
                return {"text": r.quote, "url": chunk.source_url, "confidence": "verified", "picked_by": "zeromem", "relevance": rel}

        # Backup: the reranker scores every whole sentence in the chunks ZeroMem just read.
        sents = [(snt, c.source_url) for _, c in top for snt in sentences(c.text)]
        if sents:
            t = time.time()
            ss = self.reranker.predict([(question, snt) for snt, _ in sents], show_progress_bar=False)
            i = max(range(len(sents)), key=lambda j: ss[j])
            backup.append((float(ss[i]), sents[i][0], sents[i][1], "reranker"))
            self._say(f"   backup: reranker scored {len(sents)} sentences, best {float(ss[i]):+.1f} ({time.time() - t:.1f}s)")
        if backup:
            rel, text, url, by = max(backup)
            if rel > SOFT_FLOOR:
                self._say(f"   LOW-CONFIDENCE answer picked by {by} (relevance {rel:+.1f})")
                return {"text": text, "url": url, "confidence": "low", "picked_by": by, "relevance": rel}
            self._say(f"   best sentence scored {rel:+.1f} (not above {SOFT_FLOOR:+.1f}): these pages don't answer it")
        return None

    def _page(self, url: str, pool: list[Chunk]) -> str:
        """Full text of one source page, from this question's chunks plus everything cached."""
        rows = {(c.chunk_index, c.text) for c in pool if c.source_url == url and 0 <= c.chunk_index < CODE_INDEX}
        if self.use_cache:
            try:
                rows |= set(self.cache.page_chunks(url))
            except Exception:  # noqa: BLE001 - older cache without page lookup: use what we have
                pass
        rows = {r for r in rows if 0 <= r[0] < CODE_INDEX and not is_code_chunk(r[1])}
        if not rows:
            return " ".join(c.text for c in pool if c.source_url == url)
        return merge_page(sorted(rows))

    def _continue(self, question: str, t_all: float) -> Answer:
        last = self.last
        page, start = last["page"], last["end"]
        while start < len(page) and page[start].isspace():
            start += 1
        if start >= len(page) - 3:
            msg = "That's the end of what I have from that source."
            self._say(f"[continue] {msg}")
            return Answer(question, False, text=msg, smalltalk=True, seconds=time.time() - t_all, continued=True)
        text, end = passage(page, start)
        last["end"] = end
        self._say(f"[continue] next part of {last['url']} (after your question: {last['question']!r})")
        ans = Answer(question, True, text, last["url"], [], time.time() - t_all, route="continue",
                     confidence=last["confidence"], picked_by=last["picked_by"], continued=True)
        self._say(f"\nANSWER: {text}\nSOURCE: {last['url']}   [continued] ({ans.seconds:.1f}s total)")
        return ans

    def _store(self, chunks: list[Chunk]) -> None:
        if self.use_cache and chunks:
            self.cache.store_chunks(chunks)
            self._say(f"      stored in cache (now {self.cache.size():,} chunks)")

    def _read_all(self, question: str, texts: list[str]) -> list:
        """ZeroMem reads several chunks: one batched pass for the pointer reader, else one by one."""
        if hasattr(self.reader, "read_many"):
            return self.reader.read_many(question, texts)
        return [self.reader.read(question, t) for t in texts]

    def _zeromem_read(self, question: str, chunks: list[Chunk], readings: list[dict]) -> dict | None:
        """ZeroMem reads the top-k chunks; its own output is the answer. The reranker only
        decides which chunks ZeroMem reads first (a search step, not a judge)."""
        norm = lambda x: " ".join(x.split())  # same whitespace rule as reader.norm
        t = time.time()
        chunks = [c for c in chunks if not is_code_chunk(c.text)] or chunks
        scores = self.reranker.predict([(question, c.text) for c in chunks], show_progress_bar=False)
        top = sorted(zip(scores, chunks), key=lambda x: -x[0])[: self.k]
        self._say(f"[4/5] ZeroMem reads the top {len(top)} of {len(chunks)} chunks ({time.time() - t:.1f}s)")
        reads = self._read_all(question, [c.text for _, c in top])
        for rank, ((score, chunk), r) in enumerate(zip(top, reads), 1):
            raw = re.sub(r"<(KNOW|REFUSE|UNSURE|DONE)>|\[\d+\]", " ", getattr(r, "raw", "") or "")
            raw = " ".join(raw.split())
            item = {"url": chunk.source_url, "verdict": r.verdict, "text": r.quote or raw,
                    "in_source": bool(r.quote) and norm(r.quote) in norm(chunk.text), "seconds": r.seconds}
            readings.append(item)
            mark = "in source" if item["in_source"] else "NOT in source"
            if getattr(r, "prob", None) is not None:
                item["pick"], item["prob"] = r.pick, r.prob
            self._say(f"   chunk {rank} ({urlparse(chunk.source_url).netloc}): ZeroMem {r.verdict}"
                      + (f" [{r.pick}] p={r.prob:.2f}" if getattr(r, "pick", None) else
                         (f" p={r.prob:.2f}" if getattr(r, "prob", None) is not None else ""))
                      + (f" [{mark}]: {item['text']}" if r.verdict != "REFUSE" else "") + f" ({r.seconds:.1f}s)")
        # Its best reading: a quote found in the page, else any quote it wrote, else raw output.
        for want in (lambda x: x["verdict"] == "KNOW" and x["in_source"],
                     lambda x: x["verdict"] == "KNOW" and x["text"],
                     lambda x: x["verdict"] == "MALFORMED" and x["text"]):
            pick = next((x for x in readings if want(x)), None)
            if pick:
                return {"text": pick["text"], "url": pick["url"], "picked_by": "zeromem", "relevance": None,
                        "confidence": "in-source" if pick["in_source"] else "not-in-source"}
        return None

    def _ask_zeromem(self, question: str, t_all: float) -> Answer:
        readings: list[dict] = []

        def done(hit: dict | None, route: str, fell_back: bool = False) -> Answer:
            if hit:
                ans = Answer(question, True, hit["text"], hit["url"], [], time.time() - t_all, route=route,
                             fell_back=fell_back, confidence=hit["confidence"], picked_by="zeromem", readings=readings)
                self._say(f"\nZEROMEM SAYS: {ans.text}  [{ans.confidence}]\nSOURCE: {ans.source_url}   "
                          f"[{route}] ({ans.seconds:.1f}s total)")
            else:
                ans = Answer(question, False, seconds=time.time() - t_all, route=route, fell_back=fell_back,
                             readings=readings)
                self._say(f"\nZeroMem refused every chunk it read. ({ans.seconds:.1f}s total)")
            return ans

        if self.use_cache:
            hits = self.cache.lookup(question, top_k=20, threshold=self.cache_threshold)
            if hits and len(hits) >= self.k:
                self._say(f"[1/5] cache HIT: {len(hits)} stored chunks")
                hit = self._zeromem_read(question, [Chunk(h["text"], h["source_url"], h.get("chunk_index", -1)) for h in hits], readings)
                if hit and hit["confidence"] == "in-source":
                    return done(hit, "cache")
                self._say("      no in-source answer from the cache, searching fresh")
            else:
                self._say("[1/5] cache miss")
        route = route_question(question) if self.route == "auto" else forced_route(question, self.route)
        self._say(f"[2/5] route: {route.name.upper()} ({route.why})")
        self._say(f"[3/5] retrieving from {route.name}")
        chunks = self._retrieve(question, route)
        self._store(chunks)
        hit = self._zeromem_read(question, chunks, readings) if chunks else None
        if (hit and hit["confidence"] == "in-source") or route.name == "web":
            return done(hit, route.name)
        self._say(f"      trying the open web too")
        seen = {c.text for c in chunks}
        web = [c for c in self._web_pages(question) if c.text not in seen]
        self._store(web)
        web_hit = self._zeromem_read(question, web, readings) if web else None
        best = web_hit if (web_hit and (web_hit["confidence"] == "in-source" or not hit)) else hit
        return done(best, "web" if best is web_hit and web_hit else route.name, fell_back=best is web_hit and bool(web_hit))

    def ask(self, question: str) -> Answer:
        t_all = time.time()
        question = clean_question(question)
        self._say(f"\nQ: {question}")
        msg = small_talk_reply(question)
        if msg:
            self._say(f"\n[no search] {msg}")
            return Answer(question, False, text=msg, smalltalk=True, seconds=time.time() - t_all)
        if self.zeromem_only:
            return self._ask_zeromem(question, t_all)
        # "next" / "more": keep reading the previous answer's page
        if self.last and is_continue(question):
            return self._continue(question, t_all)
        resolved = resolve_followup(question, self.last["question"] if self.last else None)
        if resolved:
            self._say(f"[follow-up] {question!r} -> {resolved!r}")
            question = resolved
        considered: list[tuple[str, float, str]] = []
        pool: list[Chunk] = []  # every chunk seen for this question, to rebuild the answer's page

        def done(hit: dict | None, route: str, fell_back: bool = False) -> Answer:
            if hit:
                page = self._page(hit["url"], pool)
                at = page.find(hit["text"])
                if at < 0:  # page couldn't be rebuilt around it: fall back to the sentence alone
                    page, at = hit["text"], 0
                if _HOWTO.search(question) and not is_code_chunk(hit["text"]):
                    hit["text"], end = passage(page, at)
                else:
                    end = at + len(hit["text"])
                self.last = {"question": question, "url": hit["url"], "page": page, "end": end,
                             "confidence": hit["confidence"], "picked_by": hit["picked_by"]}
                ans = Answer(question, True, hit["text"], hit["url"], considered, time.time() - t_all, route=route,
                             fell_back=fell_back, confidence=hit["confidence"], picked_by=hit["picked_by"],
                             relevance=hit["relevance"], resolved=resolved)
                tag = "" if ans.confidence == "verified" else f"  (LOW CONFIDENCE, picked by {ans.picked_by})"
                self._say(f"\nANSWER: {ans.text}{tag}\nSOURCE: {ans.source_url}   [{route}] ({ans.seconds:.1f}s total)")
                return ans
            ans = Answer(question, False, considered=considered, seconds=time.time() - t_all, route=route,
                         fell_back=fell_back, resolved=resolved)
            if considered:
                self._say("\nNo verified answer. ZeroMem did not produce a quote that is really in the sources.\n"
                          "Sources looked at:\n" + "\n".join(f"  - {u}" for u, _, _ in considered) + f"\n({ans.seconds:.1f}s total)")
            else:
                self._say(f"\nNo usable pages were retrieved. ({ans.seconds:.1f}s total)")
            return ans

        cache_backup: dict | None = None

        def finish(hit: dict | None, route: str, fell_back: bool = False) -> Answer:
            if hit is None and cache_backup is not None:
                return done(cache_backup, "cache")
            return done(hit, route, fell_back)

        # 1. cache: questions asked before are answered from stored chunks, no web at all
        if self.use_cache:
            t = time.time()
            hits = self.cache.lookup(question, top_k=20, threshold=self.cache_threshold)
            if hits and len(hits) >= self.k:
                chunks = [Chunk(h["text"], h["source_url"], h.get("chunk_index", -1)) for h in hits]
                pool += chunks
                self._say(f"[1/5] cache HIT: {len(chunks)} stored chunks ({time.time() - t:.1f}s)")
                hit = self._read(question, chunks, considered)
                good = hit and (hit["confidence"] == "verified" or hit["relevance"] >= CACHE_TRUST) and \
                    (not wants_code(question) or is_code_chunk(hit["text"]))
                if good:
                    return done(hit, "cache")
                cache_backup = hit
                self._say("      cache had no good answer" + (f" (best {hit['relevance']:+.1f}, kept as last resort)" if hit else "")
                          + ", searching fresh")
            else:
                self._say(f"[1/5] cache miss ({self.cache.size():,} chunks stored) ({time.time() - t:.1f}s)")

        # 2. route: decide where to look
        route = route_question(question) if self.route == "auto" else forced_route(question, self.route)
        self._say(f"[2/5] route: {route.name.upper()} ({route.why})")

        # 3. retrieve from that source
        self._say(f"[3/5] retrieving from {route.name}")
        chunks = self._retrieve(question, route)
        pool += chunks
        self._store(chunks)
        if chunks:
            hit = self._read(question, chunks, considered)
            if hit or route.name == "web":
                return finish(hit, route.name)
            self._say(f"      no verified answer from {route.name}, falling back to the open web")
        elif route.name == "web":
            return finish(None, "web")
        else:
            self._say(f"      {route.name} returned nothing, falling back to the open web")

        # fallback: open web, same steps
        seen = {c.text for c in chunks}
        web = [c for c in self._web_pages(question) if c.text not in seen]
        pool += web
        self._store(web)
        if not web:
            return finish(None, "web", fell_back=True)
        return finish(self._read(question, web, considered), "web", fell_back=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="omit for interactive mode")
    ap.add_argument("--provider", default="ddgs", choices=PROVIDERS)
    ap.add_argument("--k", type=int, default=3, help="how many top chunks ZeroMem reads")
    ap.add_argument("--ckpt", default=None, help="ZeroMem checkpoint (default: the one-chunk arrange model)")
    ap.add_argument("--device", default=None, help="auto | cpu | mps")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--min-relevance", type=float, default=2.0, help="cross-encoder score a quote must reach")
    ap.add_argument("--route", default="auto", choices=("auto",) + ROUTES,
                    help="auto = pick the source per question; web = open web only (old behaviour)")
    ap.add_argument("--zeromem-only", action="store_true",
                    help="TEST MODE: show whatever ZeroMem writes; no backup, no relevance bar, no verbatim gate")
    ap.add_argument("--reader", default="auto", choices=("auto", "pointer", "copy"),
                    help="auto = pointer model if checkpoints/pointer/best.pt exists, else the old copy model")
    ap.add_argument("--min-know", type=float, default=0.5,
                    help="pointer model: answer only when P(KNOW) >= this (higher = refuses more, fewer wrong answers)")
    args = ap.parse_args()

    p = Pipeline(args.provider, args.ckpt, args.device, use_cache=not args.no_cache, k=args.k,
                 min_relevance=args.min_relevance, route=args.route, zeromem_only=args.zeromem_only,
                 reader=args.reader, min_know=args.min_know)
    if args.question:
        p.ask(" ".join(args.question))
        return
    print("ZeroMem pipeline. Ask a question (empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        p.ask(q)


if __name__ == "__main__":
    main()
