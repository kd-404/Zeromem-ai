"""Offline test of routing + fallback: no network, no model (both are faked).

    python zeromem/tests/test_routing.py      # run from the folder that contains zeromem/
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from collections import Counter
# stub heavy deps not needed for logic
sys.modules["tokenizers"] = types.SimpleNamespace(Tokenizer=object)
sys.modules["trafilatura"] = types.SimpleNamespace(extract=lambda *a, **k: None)
import zeromem.scraper.wikipedia as W
from zeromem.scraper.chunker import Chunk

# ---- wikipedia module with mocked HTTP ----
class R:
    def __init__(s, j): s.j = j
    def raise_for_status(s): pass
    def json(s): return s.j
art = ("The Eiffel Tower is a wrought-iron lattice tower in Paris. " * 10 + "\n== History ==\n"
       + "It was designed by the engineering company of Gustave Eiffel for the 1889 World's Fair. " * 8
       + "\n== References ==\n" + "Smith, J. (1999). Some book. Publisher. " * 20)
def fake_get(url, headers=None, timeout=None, params=None):
    assert "User-Agent" in headers
    if params.get("list") == "search":
        return R({"query": {"search": [{"title": "Eiffel Tower"}, {"title": "Gustave Eiffel"}]}})
    return R({"query": {"pages": [{"extract": art if params["titles"] == "Eiffel Tower" else ""}]}})
W.requests.get = fake_get
cs, ok, why = W.wiki_chunks("Eiffel Tower")
assert ok == 1 and cs, (ok, why)
assert all("Smith" not in c.text and "==" not in c.text for c in cs), "references/headings leaked"
assert cs[0].source_url == "https://en.wikipedia.org/wiki/Eiffel_Tower"
print("wikipedia: ok", ok, "article,", len(cs), "chunks, reasons", dict(why))

# ---- pipeline paths ----
import zeromem.pipeline as P
log = []
class FakeReader:
    def read(s, q, text):
        good = "Gustave Eiffel" in text or "Chennai" in text
        quote = next((x.strip() + "." for x in text.split(".") if "Gustave" in x or "Chennai" in x), None)
        return types.SimpleNamespace(verdict="KNOW" if good else "REFUSE", quote=quote, verified=good, seconds=0.0)
class FakeRerank:
    def predict(s, pairs, show_progress_bar=False): return [5.0 if ("Gustave" in t or "Chennai" in t) else -1.0 for _, t in pairs]
class FakeCache:
    def __init__(s): s.store = []
    def lookup(s, q, top_k, threshold): return [{"text": c.text, "source_url": c.source_url} for c in s.store] if "again" in q else []
    def store_chunks(s, cs): s.store += cs * 3
    def size(s): return len(s.store)

WEB = {"https://web.example/a": "Chennai " + "filler words here " * 20}
def fake_search(q, provider, n):
    log.append(("search", q))
    if "site:" in q:
        return [types.SimpleNamespace(url="https://stackoverflow.com/q/1"), types.SimpleNamespace(url="https://spam.biz/x")]
    return [types.SimpleNamespace(url="https://web.example/a")]
def fake_fetch(urls):
    log.append(("fetch", tuple(urls)))
    cs = [Chunk(WEB.get(u, "python reverse list docs " * 20), u, 0) for u in urls]
    return cs, len(urls), Counter()
P.search, P.fetch_and_chunk = fake_search, fake_fetch
wiki_mode = {"v": "good"}
def fake_wiki(q):
    log.append(("wiki", q))
    if wiki_mode["v"] == "empty": return [], 0, Counter({"no wikipedia results": 1})
    txt = ("It was designed by Gustave Eiffel for the fair. " if wiki_mode["v"] == "good" else "unrelated tower text ") * 10
    return [Chunk(txt, "https://en.wikipedia.org/wiki/Eiffel_Tower", 0)], 1, Counter()
P.wiki_chunks = fake_wiki

def mk(route="auto"):
    p = P.Pipeline(verbose=False, route=route)
    p._reader, p._reranker, p._cache = FakeReader(), FakeRerank(), FakeCache()
    return p

log.clear(); a = mk().ask("Who designed the Eiffel Tower?")
assert a.answered and a.route == "wiki" and not a.fell_back and not any(k == "search" for k, _ in log), (a, log)
print("wiki hit:", a.route, "|", a.text[:50], "| searches:", sum(k == "search" for k, _ in log))

wiki_mode["v"] = "empty"; log.clear(); a = mk().ask("Who designed the Eiffel Tower?")
assert a.route == "web" and a.fell_back and ("search", "Who designed the Eiffel Tower?") in log
print("wiki empty -> web fallback:", a.route, a.fell_back, a.answered)

wiki_mode["v"] = "bad"; log.clear(); a = mk().ask("Who designed the Eiffel Tower?")
assert a.fell_back and a.answered and a.source_url == "https://web.example/a"
print("wiki no verified answer -> web fallback, answered from:", a.source_url)

log.clear(); p = mk(); a = p.ask("How do I reverse a list in Python?")
fetched = [u for k, v in log if k == "fetch" for u in v]
assert "site:docs.python.org" in log[0][1] and "https://spam.biz/x" not in fetched and fetched[0] == "https://stackoverflow.com/q/1"
print("code route: site-filtered query, spam.biz dropped, fetched first:", fetched[0])

wiki_mode["v"] = "good"; p = mk(); p.ask("Who designed the Eiffel Tower?"); log.clear()
a = p.ask("Who designed the Eiffel Tower again?")
assert a.route == "cache" and not log
print("cache hit: route", a.route, "| network calls:", len(log))

log.clear(); a = mk("web").ask("Who designed the Eiffel Tower?")
assert not any(k == "wiki" for k, _ in log) and a.route == "web"
print("--route web: wikipedia skipped, old behaviour")
# ZeroMem refuses, but the page has the answer: the reranker's best sentence is shown as LOW confidence
class Refuser:
    def read(s, q, text): return types.SimpleNamespace(verdict="REFUSE", quote=None, verified=False, seconds=0.0)
wiki_mode["v"] = "good"; log.clear(); p = mk(); p._reader = Refuser()
a = p.ask("Who designed the Eiffel Tower?")
assert a.answered and a.confidence == "low" and a.picked_by == "reranker" and "Gustave" in a.text and a.route == "wiki"
assert not any(k == "search" for k, _ in log), "should not fall back to web when wiki has the answer"
print("ZeroMem refused -> reranker backup, LOW confidence:", a.text[:45])

# pages without the answer still give no answer (backup must score above SOFT_FLOOR)
wiki_mode["v"] = "empty"; log.clear(); p = mk(); p._reader = Refuser()
WEB["https://web.example/a"] = "Book cheap flights now. Great offers and deals today. " * 5
a = p.ask("current flight price from Chennai to Delhi")
assert not a.answered, a
print("irrelevant pages -> still no answer")

# ---- conversation: passages, "next", follow-ups ----
from zeromem.pipeline import merge_page, passage, resolve_followup, is_continue
words = ("How to make Rasam Step by Step 1.Soak 1 lemon sized tamarind in 1 cup water. 2.Pressure cook toor dal "
         "with turmeric until soft. 3.Grind pepper, cumin and garlic coarsely. 4.Boil the tamarind water with "
         "tomatoes and salt for 5 minutes. 5.Add the ground spices and the mashed dal. 6.Heat ghee, add mustard "
         "seeds and curry leaves, pour over the rasam. 7.Garnish with coriander and serve hot with rice. "
         "Tips for the best rasam: use ripe tomatoes. Do not boil rasam after adding dal for long. ").split() * 2
c0, c1, c2 = " ".join(words[0:60]), " ".join(words[45:105]), " ".join(words[90:150])  # 15-word overlaps
page = merge_page([(0, c0), (2, c2), (1, c1)][::1] and sorted([(0, c0), (1, c1), (2, c2)]))
assert page == " ".join(words[:150]), "merge_page must undo the overlap exactly"
assert merge_page([(0, "a b c."), (3, "x y z.")]) == "a b c. ... x y z."
t, end = passage(page, 0, words=40)
assert t.endswith(".") and len(t.split()) <= 40 and page.startswith(t)
t2, end2 = passage(page, end + 1, words=40)
assert t2.startswith("4.Boil") or t2.split()[0][0].isdigit(), t2[:30]
print("merge_page + passage: ok |", t[:60], "... | next:", t2[:30])

assert resolve_followup("when was he born", "who is rajinikanth") == "when was rajinikanth born"
assert resolve_followup("how long does it take", "how to make rasam") == "how long does rasam take"
assert resolve_followup("who designed the eiffel tower", "who is rajinikanth") is None  # no pronoun
assert resolve_followup("when was he born", None) is None
assert all(is_continue(x) for x in ["next step?", "next", "continue", "more", "tell me more", "what's next?"])
assert not any(is_continue(x) for x in ["next steps for india in space", "more about tokyo"])
print("follow-ups + continue detection: ok")

# full flow: rasam -> passage, then "next step?" -> the following steps, same page, no search
class RasamReader:
    def read(s, q, t): return types.SimpleNamespace(verdict="REFUSE", quote=None, verified=False, seconds=0.0)
class RasamRerank:
    def predict(s, pairs, show_progress_bar=False):
        return [9.0 if "1.Soak" in t else (3.0 if "Rasam" in t or "rasam" in t else -2.0) for _, t in pairs]
def rasam_fetch(urls):
    log.append(("fetch", tuple(urls)))
    return [Chunk(c0, "https://recipes.example/rasam", 0), Chunk(c1, "https://recipes.example/rasam", 1),
            Chunk(c2, "https://recipes.example/rasam", 2)], 1, Counter()
P.fetch_and_chunk = rasam_fetch
p = mk(); p._reader, p._reranker = RasamReader(), RasamRerank()
log.clear(); a = p.ask("how to make rasam")
assert a.answered and "1.Soak" in a.text and "3.Grind" in a.text and a.text in page, a.text
log.clear(); b = p.ask("next step?")
assert b.continued and b.answered and not log and b.text in page and "Soak" not in b.text, b
assert page.index(b.text) > page.index(a.text)
print("rasam:", len(a.text.split()), "words, steps 1-3+ | 'next step?':", b.text[:40], "| searches:", len(log))
c = p.ask("more")
assert c.continued and page.index(c.text) > page.index(b.text)
print("'more' keeps going:", c.text[:40])

# ---- code blocks survive fetching with their line breaks ----
from zeromem.scraper.fetch import code_blocks, code_chunk_text, is_code_chunk, CODE_INDEX
html = """<html><head><title>Simple Calculator in Python - Example</title></head><body><p>Intro</p>
<pre><code class="language-python">def add(x, y):
    return x + y

def sub(x, y):
    return x - y
print(add(2, 3))</code></pre><p>inline <code>x</code> and <code>$ pip install</code></p>
<div class="code"><code>first_number = 10<br>second_number = 20<br>print(first_number &lt; second_number)<br></code></div></body></html>"""
title, blocks = code_blocks(html)
assert title.startswith("Simple Calculator") and len(blocks) == 2 and "    return x + y" in blocks[0], blocks
assert blocks[1] == "first_number = 10\nsecond_number = 20\nprint(first_number < second_number)", blocks[1]
print("code_blocks: ok |", title, "|", len(blocks), "blocks, indentation and <br> line breaks kept")

# ---- code question: answer is the best code block; prose-only cache pages are not enough ----
CALC = code_chunk_text("Simple Calculator - GfG", "def add(x, y):\n    return x + y\nchoice = input('Select operation')")
class CodeRerank:
    def predict(s, pairs, show_progress_bar=False):
        return [7.0 if "def add" in t else 5.0 if "Calculator" in t else -5.0 for _, t in pairs]
class ProseCache(FakeCache):
    def lookup(s, q, top_k, threshold):
        return [{"text": "Using Python as a Calculator. Let's try some simple Python commands. " * 3,
                 "source_url": "https://docs.python.org/3/tutorial/introduction.html", "chunk_index": i} for i in range(3)]
def code_fetch(urls):
    log.append(("fetch", tuple(urls)))
    return [Chunk("Python calculator tutorial text. " * 10, urls[0], 0), Chunk(CALC, urls[0], CODE_INDEX)], 1, Counter()
P.fetch_and_chunk = code_fetch
p = mk(); p._reader, p._reranker, p._cache = RasamReader(), CodeRerank(), ProseCache()
log.clear(); a = p.ask("write a python code for simple calculator")
assert a.answered and is_code_chunk(a.text) and "def add" in a.text and a.route == "code", a
assert any(k == "search" for k, _ in log), "cache had only prose, so it must search fresh"
print("calculator: cache prose skipped -> searched ->", a.route, "| answer is a code block:", a.text.splitlines()[0])

# ---- wrong-topic cache (biriyani vs stored rasam pages) must search fresh, not give up ----
class RasamCache(FakeCache):
    def lookup(s, q, top_k, threshold):
        return [{"text": "Rasam is a tangy South Indian soup. Soak tamarind in water. " * 3,
                 "source_url": "https://recipes.example/rasam", "chunk_index": i} for i in range(3)]
class BiryaniRerank:
    def predict(s, pairs, show_progress_bar=False):
        return [8.0 if "Biryani" in t or "biryani" in t else -8.0 for _, t in pairs]
def biryani_fetch(urls):
    log.append(("fetch", tuple(urls)))
    return [Chunk("How to make Biryani. Wash and soak basmati rice for 30 minutes. Marinate the chicken. " * 3,
                  "https://recipes.example/biryani", 0)], 1, Counter()
P.fetch_and_chunk = biryani_fetch
p = mk(); p._reader, p._reranker, p._cache = RasamReader(), BiryaniRerank(), RasamCache()
log.clear(); a = p.ask("how to make biriyani")
assert a.answered and "biryani" in a.source_url and a.route == "web", a
print("biriyani: wrong-topic cache ignored -> searched ->", a.text[:50])

# ---- zeromem-only test mode: ZeroMem's own output is the answer, nothing else steps in ----
def zm(verdict, quote, raw=""):
    return types.SimpleNamespace(verdict=verdict, quote=quote, verified=False, raw=raw or f"<KNOW> [1] {quote}<DONE>", seconds=0.1)
class DriftReader:  # writes a sentence that is NOT in the page, like the real logs
    def read(s, q, t): return zm("KNOW", "Tokyo is a city in the Kanto area of Japan.")
wiki_mode["v"] = "good"; P.fetch_and_chunk = biryani_fetch
p = mk(); p.zeromem_only = True; p._reader, p._reranker = DriftReader(), FakeRerank()
a = p.ask("where is tokyo")
assert a.answered and a.text == "Tokyo is a city in the Kanto area of Japan." and a.confidence == "not-in-source", a
assert a.picked_by == "zeromem" and a.readings and all(r["verdict"] == "KNOW" for r in a.readings)
print("zeromem-only: drifted quote is SHOWN, marked not-in-source |", len(a.readings), "readings")
class GoodReader:
    def read(s, q, t): return zm("KNOW", "It was designed by Gustave Eiffel for the fair.")
p = mk(); p.zeromem_only = True; p._reader, p._reranker = GoodReader(), FakeRerank()
a = p.ask("Who designed the Eiffel Tower?")
assert a.confidence == "in-source" and a.route == "wiki", a
print("zeromem-only: real quote marked in-source, route", a.route)
p = mk(); p.zeromem_only = True; p._reader, p._reranker = RasamReader(), FakeRerank()
a = p.ask("Who designed the Eiffel Tower?")
assert not a.answered and a.readings and all(r["verdict"] == "REFUSE" for r in a.readings)
print("zeromem-only: all REFUSE -> no answer, and no reranker backup")

# ---- zeromem-only keeps the conversation: follow-ups and "next" (was broken: both modes must work) ----
VIJAY = ("Vijay is an Indian actor and politician who works in Tamil cinema. He is married to Sangeetha "
         "Sornalingam, a Sri Lankan Tamil. They have two children. Vijay founded a political party in 2024. " * 3)
def vijay_wiki(q):
    log.append(("wiki", q))
    return [Chunk(VIJAY, "https://en.wikipedia.org/wiki/C._Joseph_Vijay", 0)], 1, Counter()
class VijayReader:
    def read(s, q, t):
        want = "married" if "wife" in q else "Indian actor"
        sent = next(x.strip() + "." for x in t.split(".") if want in x)
        return types.SimpleNamespace(verdict="KNOW", quote=sent, verified=True, raw="", seconds=0.1)
P.wiki_chunks = vijay_wiki
p = mk(); p.zeromem_only = True; p._reader, p._reranker = VijayReader(), FakeRerank()
log.clear(); a = p.ask("who is actor vijay")
b = p.ask("who is his wife?")
assert b.resolved and "actor vijay" in b.resolved.lower(), b.resolved
assert any(k == "wiki" and "vijay" in q.lower() for k, q in log[1:]), log
assert "married" in b.text.lower() and "Vijay" in b.source_url, b.text
print("zeromem-only follow-up:", repr(b.resolved), "->", b.text[:45])
log.clear(); c = p.ask("next")
assert c.continued and c.answered and not log, c
print("zeromem-only 'next': continued the same page, no new search")

# ---- the reported Suriya conversation: topic kept across follow-ups, most confident pick wins ----
from zeromem.pipeline import subject_of, resolve_followup, clock_reply
assert subject_of("Who designed the Eiffel Tower?") == "the Eiffel Tower", subject_of("Who designed the Eiffel Tower?")
assert subject_of("who is he actually?") == "he"
assert resolve_followup("who build that?", "Who designed the Eiffel Tower?") == "who build the Eiffel Tower?"
SURIYA = ("They have two children: a daughter (Diya; born 2007) and a son (Dev; born 2010). "
          "Suriya was born as Saravanan on 23 July 1975 in Madras, to actor Sivakumar. "
          "Suriya is married to the actress Jyothika. He is among the highest paid actors in Tamil cinema. ") * 2
def suriya_wiki(q):
    log.append(("wiki", q))
    return [Chunk(SURIYA, "https://en.wikipedia.org/wiki/Suriya", 0),
            Chunk("Suriya appeared in 24 in 2016. " * 12, "https://en.wikipedia.org/wiki/Suriya_filmography", 0)], 2, Counter()
class ConfidentReader:  # chunk 1 (filmography) gets a weak pick, the real answer is in chunk 2 with high confidence
    def read_many(s, q, texts):
        out = []
        for t in texts:
            if "born" in q and "Saravanan" in t:
                out.append(types.SimpleNamespace(verdict="KNOW", quote="Suriya was born as Saravanan on 23 July 1975 in Madras, to actor Sivakumar.", verified=True, raw="", seconds=0.1, pick="B", prob=0.91))
            elif "wife" in q and "Jyothika" in t:
                out.append(types.SimpleNamespace(verdict="KNOW", quote="Suriya is married to the actress Jyothika.", verified=True, raw="", seconds=0.1, pick="C", prob=0.88))
            else:
                first = t.split(". ")[0].strip() + "."
                out.append(types.SimpleNamespace(verdict="KNOW", quote=first, verified=True, raw="", seconds=0.1, pick="A", prob=0.55))
        return out
class RankFilmFirst:  # the ranker puts the weaker chunk first, like in the log
    def predict(s, pairs, show_progress_bar=False): return [5.0 if "appeared in 24" in t else 1.0 for _, t in pairs]
P.wiki_chunks = suriya_wiki
p = mk(); p.zeromem_only = True; p._reader, p._reranker = ConfidentReader(), RankFilmFirst()
p.ask("who is actor suriya?")
b = p.ask("who is he actually?")
assert b.resolved == "who is actor suriya actually?", b.resolved
c = p.ask("who is his wife?")
assert c.resolved == "who is actor suriya's wife?" and "Jyothika" in c.text, (c.resolved, c.text)
d = p.ask("when was he born>")
assert d.resolved == "when was actor suriya born", d.resolved
assert "23 July 1975" in d.text, d.text
print("suriya chain:", repr(c.resolved), "->", c.text[:32], "|", repr(d.resolved), "->", d.text[:40])

a = clock_reply("what day is today/")
assert a and "Today is" in a, a
assert clock_reply("what day comes after monday in a week") is None
log.clear(); t = mk().ask("what day is today/")
assert t.smalltalk and "Today is" in t.text and not log, (t, log)
print("clock:", t.text[:40])
print("\nALL TESTS PASSED")
