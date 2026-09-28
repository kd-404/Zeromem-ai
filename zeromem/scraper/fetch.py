"""Fetch pages, strip them to clean article text, and cut them into chunks.

Search lives in search.py (swappable providers). This module only turns URLs into
Chunk objects. It never touches a model. Facts enter through the context window,
never through weights (see BLUEPRINT.md).

Pages are fetched in parallel with a hard timeout and size cap, so one slow or huge
site can't stall the pipeline. A page that fails is skipped, never raised, but the REASON
is recorded so a low success rate is explainable (many sites answer bots with HTTP 403).
"""

from __future__ import annotations

import html as htmllib
import ipaddress
import re
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
import trafilatura

from zeromem.data.arrange_format import CHUNK_OVERLAP, CHUNK_WORDS
from zeromem.scraper.chunker import Chunk, chunk_text

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ZeroMem-research-bot/0.1)"}
MAX_BYTES = 2_000_000
MIN_CHUNK_WORDS = 30  # drop the short leftover tail of a page
CODE_INDEX = 10_000   # code-block chunks get chunk_index >= this, so they never mix into page text
MAX_CODE_BLOCKS = 8

_PRE = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.S | re.I)
_CODE = re.compile(r"<code\b[^>]*>(.*?)</code>", re.S | re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


def _strip_tags(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"</(div|p|li|tr)>", "\n", fragment, flags=re.I)
    return htmllib.unescape(re.sub(r"<[^>]+>", "", fragment))


def code_blocks(html: str) -> tuple[str, list[str]]:
    """Code examples on a page, line breaks intact: every <pre> block, plus multi-line <code>
    blocks outside <pre>. trafilatura flattens code into prose, so it is pulled out separately."""
    title = _strip_tags(m.group(1)).strip() if (m := _TITLE.search(html)) else ""
    found = [m.group(1) for m in _PRE.finditer(html)]
    rest = _PRE.sub("", html)
    found += [m.group(1) for m in _CODE.finditer(rest) if m.group(1).count("\n") + len(re.findall(r"<br", m.group(1), re.I)) >= 2]
    out, seen = [], set()
    for f in found:
        code = "\n".join(line.rstrip() for line in _strip_tags(f).strip("\n").splitlines())
        code = re.sub(r"\n{3,}", "\n\n", code).strip("\n")
        if code.count("\n") < 2 or len(code) < 40 or code in seen:
            continue  # one-liners are usually shell prompts or inline names, not examples
        seen.add(code)
        out.append("\n".join(code.splitlines()[:80]))
        if len(out) >= MAX_CODE_BLOCKS:
            break
    return title, out


def code_chunk_text(title: str, code: str) -> str:
    return f"{title}\n```\n{code}\n```" if title else f"```\n{code}\n```"


def is_code_chunk(text: str) -> bool:
    return "\n```\n" in text or text.startswith("```\n")



def check_url(url: str) -> str | None:
    """None if the URL is safe to fetch, else the reason it isn't. Only public http(s) hosts
    are fetched, so a poisoned search result can't make us hit localhost or the home network.
    DNS lookups are retried because they can time out transiently on a busy machine."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return "bad url"
    infos = None
    for attempt in range(3):
        try:
            infos = socket.getaddrinfo(p.hostname, None)
            break
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    if not infos:
        return "dns failed"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        shown = ip
        # IPv6-only networks use NAT64: an IPv4 site appears as 64:ff9b::<its IPv4 address>. That
        # prefix is "reserved" but is just a translated public site, so judge the embedded IPv4.
        if ip.version == 6 and ip in ipaddress.ip_network("64:ff9b::/96"):
            ip = ipaddress.IPv4Address(ip.packed[-4:])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return f"non-public address {shown}"
    return None


def fetch_clean_text(url: str, timeout: float = 10.0) -> tuple[str | None, str]:
    """Download one page. Returns (clean article text, "") on success, or (None, reason)."""
    text, reason, _ = fetch_page(url, timeout)
    return text, reason


def fetch_page(url: str, timeout: float = 10.0) -> tuple[str | None, str, tuple[str, list[str]]]:
    """Download one page: (clean article text or None, failure reason, (title, code blocks))."""
    bad = check_url(url)
    if bad:
        return None, bad, ("", [])
    reason = "unknown error"
    for _ in range(2):  # one retry, for transient network errors only
        try:
            with requests.get(url, headers=HEADERS, timeout=timeout, stream=True) as r:
                if r.status_code >= 400:
                    return None, f"HTTP {r.status_code}", ("", [])
                if "html" not in r.headers.get("content-type", "").lower():
                    return None, "not html", ("", [])
                html = r.raw.read(MAX_BYTES, decode_content=True)
            text = trafilatura.extract(html, include_comments=False, include_tables=False)
            try:
                codes = code_blocks(html.decode(r.encoding or "utf-8", errors="replace"))
            except Exception:  # noqa: BLE001 - code extraction is a bonus, never a failure
                codes = ("", [])
            if (not text or len(text.split()) < MIN_CHUNK_WORDS) and not codes[1]:
                return None, "no readable text", ("", [])
            return text, "", codes
        except (requests.ConnectionError, requests.Timeout) as e:
            reason = type(e).__name__
            time.sleep(1.0)
        except Exception as e:  # noqa: BLE001 - bad HTML etc: skip this page
            return None, type(e).__name__, ("", [])
    return None, reason, ("", [])


def fetch_and_chunk(urls: list[str], max_workers: int = 6, timeout: float = 10.0,
                    chunk_size_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP
                    ) -> tuple[list[Chunk], int, Counter]:
    """Fetch URLs in parallel and chunk each page. Returns (chunks, pages_that_worked, failure_reasons).
    Chunk size must match what the arrange dataset was built with (arrange_format.py)."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda u: fetch_page(u, timeout), urls))
    chunks: list[Chunk] = []
    ok = 0
    reasons: Counter = Counter()
    for url, (text, reason, (title, codes)) in zip(urls, results):
        if reason:
            reasons[reason] += 1
            continue
        ok += 1
        if text and len(text.split()) >= MIN_CHUNK_WORDS:
            chunks += [c for c in chunk_text(text, url, chunk_size_words, overlap_words)
                       if len(c.text.split()) >= MIN_CHUNK_WORDS]
        chunks += [Chunk(code_chunk_text(title, code), url, CODE_INDEX + i) for i, code in enumerate(codes)]
    return chunks, ok, reasons
