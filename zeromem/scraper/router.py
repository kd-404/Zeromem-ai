"""Question router: decide WHERE to look before searching, so ZeroMem reads clean data.

Open-web search hands the reader a mix of good pages, SEO spam and off-topic pages. A small
reader can't sort that out, so we narrow the sources first:

    wiki   general facts (who/what/when/where, history, definitions)  -> Wikipedia API
    code   programming questions                                     -> docs + Q&A sites only
    news   latest / current events                                   -> trusted news sites only
    web    everything else (prices, products, opinions, local info)   -> open web, as before

Every routed source falls back to the open web if it finds nothing usable (pipeline.py), so
a wrong route costs a little time, never the answer.

Rule-based on purpose: it is instant, needs no model, and every decision can be explained
(`Route.why`). Swap in a classifier later if the rules miss too much.

    python -m zeromem.scraper.router "Who designed the Eiffel Tower?"
    python -m zeromem.scraper.router          # runs the built-in examples
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from urllib.parse import urlparse

ROUTES = ("wiki", "code", "news", "web")

# Sites each route is allowed to read. Kept short: fewer, cleaner sources is the whole point.
CODE_SITES = ("docs.python.org", "stackoverflow.com", "realpython.com", "geeksforgeeks.org",
              "w3schools.com", "developer.mozilla.org", "pypi.org", "learn.microsoft.com",
              "docs.oracle.com", "pytorch.org", "numpy.org", "pandas.pydata.org")
NEWS_SITES = ("reuters.com", "apnews.com", "bbc.com", "thehindu.com", "indianexpress.com",
              "ndtv.com", "theguardian.com", "aljazeera.com", "npr.org")

_W = r"(?:^|\b)"  # word-start anchor that still works for things like "c++"

# Strong code words route on their own. Weak ones ("function", "class", "error") also appear in
# ordinary questions ("function of the liver"), so they need a second code word to count.
_CODE_STRONG = re.compile(_W + r"(python|javascript|typescript|java|c\+\+|c#|golang|rustlang|sql|bash|powershell|html|css|"
                          r"react|node\.?js|django|flask|fastapi|numpy|pandas|pytorch|tensorflow|regex|json|yaml|"
                          r"code|coding|script|scripts|syntax|compiler|traceback|stack trace|pip|npm|github|"
                          r"list comprehension|[a-z]*(?:error|exception)(?=:))\b|\b[A-Z][a-zA-Z]*(Error|Exception)\b", re.I)
_CODE_WEAK = re.compile(_W + r"(program|programming|function|method|class|variable|loop|array|list|dictionary|string|"
                        r"integer|debug|bug|error|compile|install|import|library|module|package|api|git|"
                        r"file|csv|database|query|server)\b", re.I)
_NEWS = re.compile(_W + r"(latest|breaking|news|today|todays|today's|yesterday|this week|this month|right now|"
                   r"currently|just announced|recent|recently|headline|headlines|election results)\b", re.I)
# Live data (prices, scores, weather, stocks) changes by the minute: no trusted static source
# has it, so it goes to the open web rather than Wikipedia or news archives.
_LIVE = re.compile(_W + r"(price|prices|cost|costs|cheapest|cheap|buy|deal|deals|flight|flights|ticket|tickets|"
                   r"fare|fares|stock|stocks|share price|weather|forecast|score|scores|near me|open now|"
                   r"review|reviews|best|live score|live scores)\b", re.I)
_FACT = re.compile(r"^(who|what|when|where|which|why|how (many|much|old|long|tall|far|big|did|does|was|were|is))\b|"
                   + _W + r"(history|invented|discovered|founded|born|died|capital|population|meaning|define|"
                   r"definition|biography|located|origin|language of|currency of|president of|author of|"
                   r"written by|designed|built|explain|tell me about)\b", re.I)

# Question words and filler stripped to turn a question into a search query.
_FILLER = re.compile(r"^(please\s+)?(can you\s+|could you\s+)?(tell me|explain|define|describe|give me|show me)?\s*"
                     r"(about|the meaning of)?\s*"
                     r"(who|what|when|where|which|why|how)?\s*"
                     r"(is|are|was|were|did|does|do|has|have|had)?\s*(the|a|an)?\s+", re.I)


@dataclass
class Route:
    name: str                  # one of ROUTES
    why: str                   # which rule fired, for the log
    query: str                 # what to send to the source
    sites: tuple[str, ...] = field(default_factory=tuple)  # allowed domains (code/news only)


def search_query(question: str) -> str:
    """Question -> keyword query: drop the question scaffolding, keep the subject."""
    q = question.strip().rstrip("?.! ")
    stripped = _FILLER.sub("", q + " ").strip()
    return stripped if len(stripped.split()) >= 1 else q


def route_question(question: str) -> Route:
    q = question.strip()
    if m := _CODE_STRONG.search(q):
        return Route("code", f"code word '{m.group(0).strip()}'", q, CODE_SITES)
    weak = {w.lower() for w in _CODE_WEAK.findall(q)}
    if len(weak) >= 2:
        return Route("code", f"code words {sorted(weak)}", q, CODE_SITES)
    if m := _LIVE.search(q):
        return Route("web", f"live-data word '{m.group(1)}'", q)
    if m := _NEWS.search(q):
        return Route("news", f"news word '{m.group(1)}'", q, NEWS_SITES)
    if m := _FACT.search(q):
        return Route("wiki", f"fact pattern '{m.group(0).strip()}'", search_query(q))
    return Route("web", "no rule matched", q)


def forced_route(question: str, name: str) -> Route:
    """--route wiki/code/news/web: skip the rules, use this source."""
    if name == "wiki":
        return Route("wiki", "forced", search_query(question))
    if name == "code":
        return Route("code", "forced", question, CODE_SITES)
    if name == "news":
        return Route("news", "forced", question, NEWS_SITES)
    return Route("web", "forced", question)


def site_query(query: str, sites: tuple[str, ...], limit: int = 6) -> str:
    """Add `site:` filters so the search engine only returns allowed domains."""
    return query + " (" + " OR ".join(f"site:{s}" for s in sites[:limit]) + ")"


def on_allowed_site(url: str, sites: tuple[str, ...]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == s or host.endswith("." + s) for s in sites)


_EXAMPLES = [
    "Who designed the Eiffel Tower?",
    "When was the Taj Mahal built?",
    "What is photosynthesis?",
    "How do I reverse a list in Python?",
    "write a python code to read a csv file",
    "TypeError: 'NoneType' object is not subscriptable",
    "latest news on the Chandrayaan mission",
    "current flight price from Chennai to Delhi",
    "best laptop under 60000",
    "capital of Australia",
    "tell me about Alan Turing",
    "why is the sky blue",
    "where do penguins live",
    "what is the function of the liver",
    "how to import a csv file into a database",
    "NameError: name 'x' is not defined",
    "India vs Australia live score",
]

if __name__ == "__main__":
    qs = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else _EXAMPLES
    for q in qs:
        r = route_question(q)
        print(f"{r.name:5} | {q}\n      why: {r.why} | query: {r.query!r}")
