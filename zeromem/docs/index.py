"""Your documents as ZeroMem's source: load a folder (or files), chunk them, find chunks for a question.

    idx = DocIndex(["samples/company_docs"])
    idx.summary()                 # files, pages, chunks, how each file was read (text / OCR)
    idx.search("how many days of paid leave?", n=40)   # candidate chunks for ZeroMem

Chunks are the same 120-word windows (30-word overlap) as web pages, so ZeroMem sees the input
shape it was trained on. Each chunk's source is "doc://<file>#page=<n>", which the chat page shows
as "<file>, page n" and links to the file.

Candidate search is BM25 keywords over all chunks (no model, instant for hundreds of pages); the
pipeline then orders those candidates with its ranker and ZeroMem reads the top ones.
"""

from __future__ import annotations

import re
import difflib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from zeromem.data.arrange_format import CHUNK_OVERLAP, CHUNK_WORDS
from zeromem.docs.extract import SUPPORTED, extract
from zeromem.scraper.chunker import Chunk, chunk_text
from zeromem.scraper.lexical import LexicalRanker

MIN_CHUNK_WORDS = 8  # documents are short and dense: keep small tail chunks too


def doc_url(name: str, page: str) -> str:
    return f"doc://{quote(name)}#page={quote(page)}"


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop headers/footers: lines that appear on at least half of the pages (3+ pages).
    Page numbers ("Page 3") are normalised so they count as the same line."""
    if len(pages) < 3:
        return pages
    norm = lambda l: re.sub(r"\d+", "#", l.strip().lower())
    counts = Counter(n for p in pages for n in {norm(l) for l in p.splitlines() if l.strip()})
    repeated = {n for n, c in counts.items() if c >= max(3, len(pages) // 2)}
    return ["\n".join(l for l in p.splitlines() if norm(l) not in repeated) for p in pages]


def end_headings(text: str) -> str:
    """Give heading-like lines a full stop so they don't fuse with the next sentence
    ("...factory closures The Tiruppur factory will be closed..."). A line counts as a heading
    when it has no end punctuation, the next line starts a new sentence (capital or digit), and it
    is clearly shorter than the page's longest line (lines wrapped mid-sentence run full width)."""
    lines = text.splitlines()
    widest = max((len(l.strip()) for l in lines), default=0)
    out = []
    for i, line in enumerate(lines):
        t = line.rstrip()
        nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        open_ended = bool(t.strip()) and t[-1] not in ".!?:;,)\"'"
        next_starts_sentence = nxt[:1].isupper() or nxt[:1].isdigit()
        if open_ended and next_starts_sentence and len(t.strip()) < 0.85 * widest:
            t += "."
        out.append(t)
    return "\n".join(out)


@dataclass
class DocFile:
    name: str
    path: Path
    method: str
    pages: int
    chunks: int
    notes: list[str] = field(default_factory=list)


class DocIndex:
    def __init__(self, paths: list[str | Path], verbose: bool = True):
        self.files: list[DocFile] = []
        self.chunks: list[Chunk] = []
        self.failed: list[tuple[str, str]] = []
        self.ranker = LexicalRanker()
        self._by_name: dict[str, Path] = {}
        for p in self._expand(paths):
            self._add(p, verbose)
        text = " ".join(c.text for c in self.chunks)
        self.vocab = {w for w in re.findall(r"[a-z][a-z'-]+", text.lower()) if len(w) >= 3}
        self.main_subject = self._main_subject(text)

    @staticmethod
    def _main_subject(text: str) -> str | None:
        """The name the documents are mostly about ("Kaveri Loom"): the most frequent pair of
        capitalised words. 'it' / 'they' in a question with no other topic refers to it."""
        skip = {"The", "This", "That", "These", "Every", "All", "Each", "Our", "Its", "Their", "A", "An", "In",
                "On", "For", "Of", "And", "Page", "Note", "Customers", "Employees", "Business", "Company"}
        pairs = Counter(m for m in re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", text) if m.split()[0] not in skip)
        if not pairs:
            return None
        name, count = pairs.most_common(1)[0]
        return name if count >= 3 else None

    def fix_spelling(self, question: str) -> str:
        """Replace words the documents never use with the closest word they do use
        ("manufacuture" -> "manufacturer"). Short words and known words are left alone."""
        def fix(m):
            w = m.group(0)
            lw = w.lower()
            if len(lw) < 5 or lw in self.vocab:
                return w
            close = difflib.get_close_matches(lw, self.vocab, n=1, cutoff=0.82)
            return close[0] if close else w
        return re.sub(r"[A-Za-z][A-Za-z'-]+", fix, question)

    @staticmethod
    def _expand(paths):
        for p in map(Path, paths):
            if p.is_dir():
                yield from sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED
                                  and not f.name.startswith("."))
            elif p.is_file():
                yield p

    def _add(self, path: Path, verbose: bool) -> None:
        name = path.name
        if name in self._by_name:  # same file name in two folders: keep both, make the second unique
            name = f"{path.parent.name}/{path.name}"
        try:
            ex = extract(path)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the others
            self.failed.append((name, f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"  ! {name}: {e}", flush=True)
            return
        texts = [end_headings(t) for t in strip_repeated_lines([t for _, t in ex.pages])]
        n0 = len(self.chunks)
        for (label, _), text in zip(ex.pages, texts):
            for c in chunk_text(text, doc_url(name, label), CHUNK_WORDS, CHUNK_OVERLAP):
                if len(c.text.split()) >= MIN_CHUNK_WORDS:
                    self.chunks.append(c)
        self._by_name[name] = path
        f = DocFile(name, path, ex.method, len(ex.pages), len(self.chunks) - n0, ex.notes)
        self.files.append(f)
        if verbose:
            extra = f"  [{'; '.join(ex.notes)}]" if ex.notes else ""
            print(f"  {name}: {f.pages} page(s) -> {f.chunks} chunks via {ex.method}{extra}", flush=True)

    def path_of(self, name: str) -> Path | None:
        """The file behind a doc:// link (only files that were loaded, so no path tricks)."""
        return self._by_name.get(name)

    def search(self, question: str, n: int = 60) -> list[Chunk]:
        if not self.chunks:
            return []
        scores = self.ranker.predict([(question, c.text) for c in self.chunks])
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        # Keywords only TRIM large collections; they never decide. A chunk without the question's exact
        # words (a synonym, a typo like "manufacuture") still reaches the ranker and ZeroMem.
        return [self.chunks[i] for i in order[:n]]

    def summary(self) -> str:
        pages = sum(f.pages for f in self.files)
        lines = [f"{len(self.files)} file(s), {pages} page(s), {len(self.chunks)} chunks"]
        lines += [f"  - {f.name}: {f.pages} page(s), {f.method}" for f in self.files]
        lines += [f"  ! {n}: {why}" for n, why in self.failed]
        return "\n".join(lines)
