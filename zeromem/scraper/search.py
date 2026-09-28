"""Web search with swappable providers. Returns URLs (+ titles) for a question.

    ddgs    DuckDuckGo via the `ddgs` package. No account, no key. Unofficial, so it
            can be rate-limited; this is the default so the pipeline works out of the box.
    brave   Brave Search API      (env BRAVE_API_KEY)  - no free tier anymore, kept for completeness
    tavily  Tavily search API     (env TAVILY_API_KEY) - 1,000 free searches/month
    exa     Exa search API        (env EXA_API_KEY)

Only the search step differs per provider. Everything after it (fetch, chunk, cache,
rerank, read) is identical, so switching provider is one flag.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

PROVIDERS = ("ddgs", "brave", "tavily", "exa")


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str = ""


def _need_key(name: str) -> str:
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(f"{name} is not set. Put it in your environment (or use --provider ddgs, which needs no key).")
    return key


def _ddgs(query: str, n: int) -> list[SearchHit]:
    from ddgs import DDGS

    last_err: Exception | None = None
    for attempt in range(3):  # ddgs is unofficial: retry a couple of times on rate limits
        try:
            rows = DDGS().text(query, max_results=n)
            return [SearchHit(r["href"], r.get("title", ""), r.get("body", "")) for r in rows if r.get("href")]
        except Exception as e:  # noqa: BLE001 - provider errors vary; surface after retries
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ddgs search failed after 3 attempts: {last_err}")


def _brave(query: str, n: int) -> list[SearchHit]:
    r = requests.get("https://api.search.brave.com/res/v1/web/search",
                     headers={"Accept": "application/json", "X-Subscription-Token": _need_key("BRAVE_API_KEY")},
                     params={"q": query, "count": n}, timeout=15)
    r.raise_for_status()
    return [SearchHit(x["url"], x.get("title", ""), x.get("description", ""))
            for x in r.json().get("web", {}).get("results", []) if "url" in x]


def _tavily(query: str, n: int) -> list[SearchHit]:
    r = requests.post("https://api.tavily.com/search",
                      json={"api_key": _need_key("TAVILY_API_KEY"), "query": query, "max_results": n}, timeout=20)
    r.raise_for_status()
    return [SearchHit(x["url"], x.get("title", ""), x.get("content", "")) for x in r.json().get("results", []) if "url" in x]


def _exa(query: str, n: int) -> list[SearchHit]:
    r = requests.post("https://api.exa.ai/search", headers={"x-api-key": _need_key("EXA_API_KEY")},
                      json={"query": query, "numResults": n}, timeout=20)
    r.raise_for_status()
    return [SearchHit(x["url"], x.get("title", ""), "") for x in r.json().get("results", []) if "url" in x]


_IMPL = {"ddgs": _ddgs, "brave": _brave, "tavily": _tavily, "exa": _exa}


def search(query: str, provider: str = "ddgs", n: int = 6) -> list[SearchHit]:
    if provider not in _IMPL:
        raise ValueError(f"unknown provider {provider!r}; choose one of {PROVIDERS}")
    seen: set[str] = set()
    hits = []
    for h in _IMPL[provider](query, n):
        if h.url not in seen:
            seen.add(h.url)
            hits.append(h)
    return hits
