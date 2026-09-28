"""Wikipedia as a clean source: search + plain-text article extracts through the official API.

No HTML scraping, no bot blocks, no ads or menus to strip: the API returns article text
directly. Returns the same (chunks, pages_ok, failure_reasons) shape as fetch_and_chunk, so the
pipeline treats it like any other source.

    python -m zeromem.scraper.wikipedia "Eiffel Tower designer"
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests

from zeromem.data.arrange_format import CHUNK_OVERLAP, CHUNK_WORDS
from zeromem.scraper.chunker import Chunk, chunk_text
from zeromem.scraper.fetch import MIN_CHUNK_WORDS

API = "https://{lang}.wikipedia.org/w/api.php"
# Wikimedia asks API clients to identify themselves with a descriptive User-Agent.
HEADERS = {"User-Agent": "ZeroMem-research-bot/0.1 (local research project; python-requests)"}

# Sections that are lists of links or citations, not prose: no answer sentence lives there.
_DROP_SECTIONS = {"references", "see also", "external links", "further reading", "notes",
                  "bibliography", "sources", "citations", "footnotes", "gallery"}
_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


def search_titles(query: str, n: int = 3, lang: str = "en", timeout: float = 10.0) -> list[str]:
    r = requests.get(API.format(lang=lang), headers=HEADERS, timeout=timeout, params={
        "action": "query", "list": "search", "srsearch": query, "srlimit": n,
        "srnamespace": 0, "format": "json", "formatversion": 2})
    r.raise_for_status()
    return [x["title"] for x in r.json().get("query", {}).get("search", [])]


def article_text(title: str, lang: str = "en", timeout: float = 10.0) -> str:
    """Full plain-text article (one title per request: the API only returns one full extract
    at a time). Redirects are followed, so a search title always resolves to an article."""
    r = requests.get(API.format(lang=lang), headers=HEADERS, timeout=timeout, params={
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
        "titles": title, "redirects": 1, "format": "json", "formatversion": 2})
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", [])
    return pages[0].get("extract", "") if pages else ""


def clean_article(text: str) -> str:
    """Drop headings and link/citation sections; keep the prose paragraphs."""
    out, skipping = [], False
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            level, name = len(m.group(1)), m.group(2).strip().lower()
            if level == 2:
                skipping = name in _DROP_SECTIONS
            elif name in _DROP_SECTIONS:
                skipping = True
            continue  # a heading is not a sentence: never let it reach the reader
        if not skipping and line.strip():
            out.append(line.strip())
    return "\n".join(out)


def page_url(title: str, lang: str = "en") -> str:
    return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def wiki_chunks(query: str, n_pages: int = 3, lang: str = "en", timeout: float = 10.0,
                chunk_size_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP
                ) -> tuple[list[Chunk], int, Counter]:
    reasons: Counter = Counter()
    try:
        titles = search_titles(query, n_pages, lang, timeout)
    except Exception as e:  # noqa: BLE001 - network/API errors: report, let the pipeline fall back
        reasons[f"wiki search failed: {type(e).__name__}"] += 1
        return [], 0, reasons
    if not titles:
        reasons["no wikipedia results"] += 1
        return [], 0, reasons

    def get(title: str) -> tuple[str, str | None, str]:
        try:
            return title, clean_article(article_text(title, lang, timeout)), ""
        except Exception as e:  # noqa: BLE001
            return title, None, type(e).__name__

    with ThreadPoolExecutor(max_workers=len(titles)) as pool:
        results = list(pool.map(get, titles))
    chunks: list[Chunk] = []
    ok = 0
    for title, text, err in results:
        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            reasons[err or "empty article"] += 1
            continue
        ok += 1
        chunks += [c for c in chunk_text(text, page_url(title, lang), chunk_size_words, overlap_words)
                   if len(c.text.split()) >= MIN_CHUNK_WORDS]
    return chunks, ok, reasons


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Eiffel Tower"
    cs, ok, why = wiki_chunks(q)
    print(f"{ok} pages -> {len(cs)} chunks {dict(why) if why else ''}")
    for c in cs[:3]:
        print(f"\n[{c.source_url} #{c.chunk_index}]\n{c.text[:300]}...")
