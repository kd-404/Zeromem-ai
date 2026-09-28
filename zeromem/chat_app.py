"""ZeroMem chat: a local web page for the pipeline, with routing and live steps.

    python -m zeromem.chat_app                 # then open http://localhost:7860
    python -m zeromem.chat_app --port 8000 --route web --provider tavily

No extra packages: it uses Python's built-in web server. Each question streams the pipeline's
steps to the page as they happen (route, source, rerank, ZeroMem's verdict per chunk), then
the verified quote with its source link, or "no verified answer".

Runs on 127.0.0.1 only, so nothing on your network can reach it.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict
from html import escape as htmlesc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from zeromem.pipeline import Pipeline
from zeromem.scraper.router import ROUTES

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZeroMem Chat</title>
<style>
:root{--bg:#f6f6f4;--panel:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e4e3de;--me:#1d1d1b;--meink:#fff;
--wiki:#2f6fdb;--code:#7a4bd6;--news:#c0561c;--web:#4f6b57;--cache:#8a6d1c;--ok:#1f7a4a;--bad:#b3261e;--low:#a86b00}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1e1e1c;--ink:#ecebe6;--mute:#9a9992;--line:#2e2e2b;--me:#ecebe6;--meink:#141413;
--wiki:#7aa7ff;--code:#b394ff;--news:#ff9c63;--web:#8fb89a;--cache:#e0c25e;--ok:#5fd097;--bad:#ff8a80;--low:#f2b84b}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;display:flex;flex-direction:column}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:12px;background:var(--panel)}
header b{font-size:17px;letter-spacing:-.01em}header span{color:var(--mute);font-size:13px}
.modes{margin-left:auto;display:flex;border:1px solid var(--line);border-radius:99px;padding:2px;gap:2px}
.modes button{font:inherit;font-size:12.5px;font-weight:600;padding:5px 12px;border-radius:99px;background:none;color:var(--mute);border:0;cursor:pointer}
.modes button.on{background:var(--me);color:var(--meink)}
.about{max-width:560px;margin:14px auto 0;font-size:13px;line-height:1.55}.about a{color:var(--wiki)}
@media (max-width:640px){header{flex-wrap:wrap}.modes{margin-left:0}}
#log{flex:1;overflow-y:auto;padding:24px 16px}
.wrap{max-width:760px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
.me{align-self:flex-end;background:var(--me);color:var(--meink);padding:10px 14px;border-radius:16px 16px 4px 16px;max-width:80%}
.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--line);border-radius:16px 16px 16px 4px;padding:14px 16px;max-width:92%;width:100%}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.chip{font-size:11.5px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;padding:2px 8px;border-radius:99px;border:1px solid currentColor}
.chip.wiki{color:var(--wiki)}.chip.code{color:var(--code)}.chip.news{color:var(--news)}.chip.web{color:var(--web)}.chip.cache{color:var(--cache)}
.chip.fb{color:var(--mute)}.chip.ok{color:var(--ok)}.chip.low{color:var(--low)}.chip.continue{color:var(--mute)}.resolved{color:var(--mute);font-size:12.5px;margin:0 0 6px}.chip.no{color:var(--bad)}
.ans{font-size:16px;margin:4px 0 8px}.ans.none{color:var(--mute)}
.codeblock{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow-x:auto;white-space:pre;margin:6px 0}.ctitle{font-size:13px;color:var(--mute)}.bad{color:var(--bad)}.rd{font-size:13px;margin:8px 0 0;padding:0;list-style:none}.rd li{padding:6px 0;border-top:1px dashed var(--line)}.rd .dom{color:var(--mute);font-size:12px}.mode{color:var(--bad);font-weight:600}.note{color:var(--low);font-size:12.5px;margin:-2px 0 6px}.src a{color:var(--wiki);word-break:break-all;font-size:13px}
details{margin-top:10px;border-top:1px dashed var(--line);padding-top:8px}
summary{cursor:pointer;color:var(--mute);font-size:13px;user-select:none}
.steps{font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mute);white-space:pre-wrap;margin:8px 0 0}
.live .steps{margin:0}
.dots::after{content:"";animation:d 1.2s steps(4) infinite}@keyframes d{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}
form{padding:14px 16px 18px;border-top:1px solid var(--line);background:var(--panel)}
.bar{max-width:760px;margin:0 auto;display:flex;gap:8px}
input{flex:1;font:inherit;padding:12px 14px;border-radius:12px;border:1px solid var(--line);background:var(--bg);color:var(--ink);outline:none}
input:focus{border-color:var(--mute)}
button{font:inherit;font-weight:600;padding:0 18px;border-radius:12px;border:0;background:var(--me);color:var(--meink);cursor:pointer}
button:disabled{opacity:.4;cursor:default}
.hint{max-width:760px;margin:6px auto 0;color:var(--mute);font-size:12px}
.empty{color:var(--mute);text-align:center;margin-top:12vh}
.empty p{margin:6px}.ex{display:inline-block;margin:4px;padding:6px 10px;border:1px solid var(--line);border-radius:99px;cursor:pointer;font-size:13px;color:var(--ink);background:var(--panel)}
</style></head><body>
<header><b>ZeroMem</b><span id="tag"></span><div class="modes" id="modes"></div></header>
<div id="log"><div class="wrap" id="wrap">
 <div class="empty" id="empty"><p>Ask a question. ZeroMem picks where to look, reads, and quotes a verified sentence with its source.</p>
  <p class="about">__ABOUT__</p>
  <div id="exs"></div></div>
</div></div>
<form id="f"><div class="bar"><input id="q" autocomplete="off" placeholder="Ask anything factual..." autofocus><button id="b">Ask</button></div>
<div class="hint">Say <b>next</b> or <b>more</b> to keep reading the last source &middot; Routes: <b style="color:var(--wiki)">wiki</b> facts &middot; <b style="color:var(--code)">code</b> docs &amp; Q&amp;A &middot; <b style="color:var(--news)">news</b> trusted outlets &middot; <b style="color:var(--web)">web</b> everything else &middot; <b style="color:var(--cache)">cache</b> asked before</div></form>
<script>
const wrap=document.getElementById('wrap'),log=document.getElementById('log'),f=document.getElementById('f'),q=document.getElementById('q'),b=document.getElementById('b');
const MODES=__MODES__;let mode=MODES[0].id;const EXAMPLES=__EXAMPLES__;
const sid=Math.random().toString(36).slice(2,12);
const modesEl=document.getElementById('modes'),tagEl=document.getElementById('tag');
function setMode(id){mode=id;const m=MODES.find(x=>x.id===id);tagEl.innerHTML=m.tag;
 modesEl.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.id===id));}
if(MODES.length>1)MODES.forEach(m=>{const b=document.createElement('button');b.type='button';b.textContent=m.label;b.dataset.id=m.id;b.onclick=()=>setMode(m.id);modesEl.appendChild(b)});
else modesEl.remove();
setMode(mode);
const srcHref=u=>u.startsWith('doc://')?'/doc/'+u.slice(6):u;
const srcText=u=>{if(!u.startsWith('doc://'))return u;const[f,pg]=u.slice(6).split('#page=');const n=decodeURIComponent(f),p=decodeURIComponent(pg||'');return n+(p?(/^\d+$/.test(p)?' · page '+p:' · '+p):'');};
const srcHost=u=>u.startsWith('doc://')?srcText(u):new URL(u).hostname;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const down=()=>log.scrollTop=log.scrollHeight;
const stepfmt=t=>esc(t).replace(/\s(\d{1,2})\.(?=\s?[A-Z])/g,'<br>$1. ');
const fmt=t=>t.includes('```')?t.split('```').map((p,i)=>i%2?`<pre class="codeblock">${esc(p.replace(/^\n|\n$/g,''))}</pre>`:(p.trim()?`<div class="ctitle">${esc(p.trim())}</div>`:'')).join(''):`&ldquo;${stepfmt(t)}&rdquo;`;
document.getElementById('exs').innerHTML=EXAMPLES.map(x=>`<span class="ex">${esc(x)}</span>`).join('');
document.querySelectorAll('.ex').forEach(e=>e.onclick=()=>{q.value=e.textContent;f.requestSubmit()});
f.onsubmit=e=>{e.preventDefault();const text=q.value.trim();if(!text||b.disabled)return;
 document.getElementById('empty')?.remove();
 wrap.insertAdjacentHTML('beforeend',`<div class="me">${esc(text)}</div>`);
 const bot=document.createElement('div');bot.className='bot live';
 bot.innerHTML=`<div class="chips"><span class="chip fb dots">working</span></div><pre class="steps"></pre>`;
 wrap.appendChild(bot);down();q.value='';b.disabled=true;
 const steps=[];const pre=bot.querySelector('.steps');
 const es=new EventSource('/ask?q='+encodeURIComponent(text)+'&mode='+mode+'&sid='+sid);
 es.addEventListener('step',ev=>{const line=JSON.parse(ev.data);steps.push(line);pre.textContent=steps.join('\n');
   const m=line.match(/route: (\w+)/);if(m)bot.querySelector('.chips').innerHTML=`<span class="chip ${m[1].toLowerCase()}">${m[1].toLowerCase()}</span><span class="chip fb dots">working</span>`;down()});
 es.addEventListener('answer',ev=>{es.close();const a=JSON.parse(ev.data);bot.classList.remove('live');
   let chips='';
   if(a.smalltalk)chips=`<span class="chip fb">no search</span>`;
   else{chips=`<span class="chip ${a.route}">${a.continued?'continued':a.route}</span>`+(a.fell_back?`<span class="chip fb">fell back to web</span>`:'')+
     (a.answered?(a.confidence==='in-source'?`<span class="chip ok">in source</span><span class="chip fb">written by zeromem</span>`:a.confidence==='not-in-source'?`<span class="chip no">not in source</span><span class="chip fb">written by zeromem</span>`:a.confidence==='verified'?`<span class="chip ok">verified</span>`:`<span class="chip low">low confidence</span><span class="chip fb">picked by ${a.picked_by}</span>`):`<span class="chip no">no verified answer</span>`)+`<span class="chip fb">${a.seconds.toFixed(1)}s</span>`}
   let body;
   if(a.smalltalk)body=`<div class="ans">${esc(a.text)}</div>`;
   else if(a.answered)body=(a.resolved?`<div class="resolved">understood as: <i>${esc(a.resolved)}</i></div>`:'')+`<div class="ans">${fmt(a.text)}</div>${a.confidence==='low'?`<div class="note">${a.text.includes('```')?'Code copied as-is from the source page. Read it before running it.':'Copied word for word from the source, but ZeroMem did not confirm it answers your question. Check the link.'}</div>`:''}<div class="src"><a href="${esc(srcHref(a.source_url))}" target="_blank" rel="noopener">${esc(srcText(a.source_url))}</a></div>`;
   else if(a.readings&&a.readings.length)body=`<div class="ans none">ZeroMem said REFUSE for every chunk it read.</div>`;
   else body=`<div class="ans none">I couldn't verify an answer in the sources I read${a.considered.length?'':' (no usable pages were retrieved)'}.</div>`+
     (a.considered.length?`<div class="src">${[...new Set(a.considered.map(c=>c[0]))].map(u=>`<a href="${esc(srcHref(u))}" target="_blank" rel="noopener">${esc(srcText(u))}</a>`).join('<br>')}</div>`:'');
   if(a.confidence==='not-in-source')body=body.replace('<div class="src">','<div class="note bad">ZeroMem wrote this, but this exact sentence is not in the page. It may be reworded or made up.</div><div class="src">');
   if(a.readings&&a.readings.length)body+=`<details open><summary>What ZeroMem wrote for each chunk (${a.readings.length})</summary><ul class="rd">${a.readings.map(r=>`<li><span class="chip ${r.verdict==='REFUSE'?'fb':r.in_source?'ok':'no'}">${r.verdict==='REFUSE'?'refuse':r.in_source?'in source':r.verdict==='KNOW'?'not in source':'malformed'}</span> <span class="dom">${esc(srcHost(r.url))}</span>${r.verdict==='REFUSE'?'':`<div>${esc(r.text)}</div>`}</li>`).join('')}</ul></details>`;
   bot.innerHTML=`<div class="chips">${chips}</div>${body}`+(steps.length?`<details><summary>How I got this (${steps.length} steps)</summary><pre class="steps">${esc(steps.join('\n'))}</pre></details>`:'');
   b.disabled=false;q.focus();down()});
 es.addEventListener('fail',ev=>{es.close();bot.classList.remove('live');bot.innerHTML=`<div class="chips"><span class="chip no">error</span></div><div class="ans none">${esc(JSON.parse(ev.data))}</div>`;b.disabled=false});
 es.onerror=()=>{if(b.disabled){es.close();bot.classList.remove('live');bot.querySelector('.chips').innerHTML='<span class="chip no">connection lost</span>';b.disabled=false}};
};
</script></body></html>"""


class ChatServer:
    """Runs questions one at a time (the models are not thread-safe). Each visitor (sid) keeps
    their own conversation memory per mode, so "next" and follow-ups never mix between people."""

    def __init__(self, pipelines: dict[str, Pipeline], per_minute: int = 0, max_waiting: int = 0):
        self.pipes = pipelines
        self.lock = threading.Lock()
        self.states: OrderedDict = OrderedDict()   # (sid, mode) -> that visitor's last answer
        self.per_minute, self.max_waiting = per_minute, max_waiting  # 0 = no limit (local use)
        self.hits: dict[str, deque] = defaultdict(deque)
        self.waiting = 0
        self.meta = threading.Lock()
        self.docs = None  # DocIndex when answering from documents (serves /doc/<file>)

    def admit(self, ip: str) -> str | None:
        """None if this request may run, else the reason it can't (public mode only)."""
        with self.meta:
            if self.per_minute:
                h, now = self.hits[ip], time.time()
                while h and now - h[0] > 60:
                    h.popleft()
                if len(h) >= self.per_minute:
                    return f"Slow down a little: {self.per_minute} questions per minute. Try again in {int(61 - (now - h[0]))}s."
                h.append(now)
            if self.max_waiting and self.waiting >= self.max_waiting:
                return "ZeroMem is busy answering other people. Try again in a few seconds."
            self.waiting += 1
            return None

    def ask(self, question: str, mode: str, sid: str, emit) -> None:
        p = self.pipes.get(mode) or next(iter(self.pipes.values()))
        try:
            with self.lock:
                key = (sid, mode)
                p.last = self.states.get(key)
                p._say = lambda msg: [emit("step", line) for line in msg.strip("\n").splitlines() if line.strip()]
                try:
                    ans = p.ask(question)
                    emit("answer", asdict(ans))
                except Exception as e:  # noqa: BLE001 - show the error in the chat instead of hanging
                    emit("fail", f"{type(e).__name__}: {e}")
                finally:
                    self.states[key] = p.last
                    self.states.move_to_end(key)
                    while len(self.states) > 2000:
                        self.states.popitem(last=False)
        finally:
            with self.meta:
                self.waiting -= 1


def make_handler(server: ChatServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal for pipeline output
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if url.path.startswith("/doc/") and server.docs is not None:
                path = server.docs.path_of(unquote(url.path[5:]))  # only files that were loaded
                if not path:
                    self.send_error(404)
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if url.path == "/health":
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(body)
                return
            if url.path != "/ask":
                self.send_error(404)
                return
            qs = parse_qs(url.query)
            question = (qs.get("q") or [""])[0].strip()[:500]
            mode = (qs.get("mode") or [""])[0]
            sid = re.sub(r"[^a-z0-9]", "", (qs.get("sid") or ["anon"])[0].lower())[:32] or "anon"
            # behind a proxy (Hugging Face, Render) the visitor's address is the first X-Forwarded-For entry
            ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")  # stream through proxies instead of buffering
            self.end_headers()
            refused = server.admit(ip)
            if refused:
                self.wfile.write(f"event: fail\ndata: {json.dumps(refused)}\n\n".encode())
                return
            events: queue.Queue = queue.Queue()
            threading.Thread(target=server.ask, args=(question, mode, sid, lambda k, v: events.put((k, v))),
                             daemon=True).start()
            while True:
                kind, data = events.get()
                try:
                    self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return  # browser closed the tab; the pipeline thread finishes on its own
                if kind in ("answer", "fail"):
                    return

    return Handler


TAG_ZM = '<span class="mode">ZEROMEM ONLY</span> &middot; the model\'s own picks, nothing filtered'
TAG_FULL = "answers only from sources &middot; every quote is copied word for word"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default=None, help="default 127.0.0.1 (this computer only); --public uses 0.0.0.0")
    ap.add_argument("--public", action="store_true",
                    help="internet-facing: listen on 0.0.0.0, rate-limit per visitor, ZeroMem-only/full toggle, "
                         "no local cache, models loaded at startup")
    ap.add_argument("--lite", action="store_true",
                    help="--public for small servers (Render free, 512 MB): ZeroMem-only mode, keyword ranking "
                         "instead of the reranker model, 2 chunks per question")
    ap.add_argument("--per-minute", type=int, default=12, help="--public: questions per visitor per minute")
    ap.add_argument("--max-waiting", type=int, default=8, help="--public: questions allowed in line at once")
    ap.add_argument("--provider", default="ddgs")
    ap.add_argument("--route", default="auto", choices=("auto",) + ROUTES)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--reader", default="auto", choices=("auto", "pointer", "copy"))
    ap.add_argument("--min-know", type=float, default=float(os.environ.get("MIN_KNOW", 0.5)),
                    help="pointer model: answer only when P(KNOW) >= this (higher = refuses more, fewer wrong answers)")
    ap.add_argument("--zeromem-only", action="store_true",
                    help="TEST MODE: scraped chunks go into ZeroMem and whatever it writes is shown")
    ap.add_argument("--docs", nargs="+", default=None, metavar="PATH",
                    help="answer ONLY from these files/folders (PDF, scans and photos via OCR, Word, PowerPoint, "
                         "Excel, CSV, text, HTML); no web search")
    args = ap.parse_args()
    if args.lite:
        args.public = True
        args.k = min(args.k, 2)
        import torch
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

    use_cache = not (args.no_cache or args.public)  # a public server keeps no local cache
    docs = None
    if args.docs:
        from zeromem.docs.index import DocIndex
        print("Reading your documents ...", flush=True)
        docs = DocIndex(args.docs)
        print(docs.summary(), flush=True)
        if not docs.chunks:
            raise SystemExit("No readable text found in --docs. Check the paths and the messages above.")
    make = lambda zm: Pipeline(args.provider, args.ckpt, args.device, use_cache=use_cache, k=args.k, route=args.route,
                               zeromem_only=zm, reader=args.reader, min_know=args.min_know,
                               ranker="bm25" if args.lite else "cross-encoder", docs=docs)
    if args.lite:
        zm = make(True)
        print("Loading ZeroMem ...", flush=True)
        zm._say = lambda msg: None
        zm.reader, zm.reranker
        pipes = {"zeromem": zm}
        modes = [{"id": "zeromem", "label": "", "tag": TAG_ZM}]
        server = ChatServer(pipes, args.per_minute, args.max_waiting)
    elif args.public:
        zm, full = make(True), make(False)
        print("Loading models ...", flush=True)
        zm._say = full._say = lambda msg: None
        zm.reader, zm.reranker  # load once at startup ...
        full._reader, full._reranker = zm._reader, zm._reranker  # ... and share them between both modes
        pipes = {"zeromem": zm, "full": full}
        modes = [{"id": "zeromem", "label": "ZeroMem only", "tag": TAG_ZM},
                 {"id": "full", "label": "Full system", "tag": TAG_FULL}]
        server = ChatServer(pipes, args.per_minute, args.max_waiting)
    else:
        mid = "zeromem" if args.zeromem_only else "full"
        pipes = {mid: make(args.zeromem_only)}
        modes = [{"id": mid, "label": "", "tag": TAG_ZM if args.zeromem_only else TAG_FULL}]
        server = ChatServer(pipes)

    server.docs = docs
    examples = ["Who designed the Eiffel Tower?", "How do I reverse a list in Python?", "latest news on Chandrayaan",
                "how to make rasam"]
    if docs is not None:
        n_files = len(docs.files)
        where = f"answers only from your documents ({n_files} file{'s' if n_files != 1 else ''}), no web search"
        for m in modes:
            m["tag"] = (TAG_ZM.split(" &middot; ")[0] + " &middot; " + where) if m["id"] == "zeromem" else where
        examples = ["Who is the CEO?", "How many days of paid leave do employees get?",
                    "What was the revenue in FY 2025-26?", "How long must passwords be?"]
    repo = os.environ.get("REPO_URL", "").strip()
    about = ("ZeroMem is a small language model (about 34M parameters) trained from scratch on a laptop. "
             "It never answers from memory: it reads web pages and points at the sentence that answers, "
             "or refuses." + (f' <a href="{repo}" target="_blank" rel="noopener">Source code</a>' if repo else ""))
    global PAGE
    if docs is not None:
        about = htmlesc("ZeroMem is reading only your documents: " + ", ".join(f.name for f in docs.files[:4])
                        + (" and more" if len(docs.files) > 4 else "") + ". If they don't answer a question, it says so.")
    PAGE = PAGE.replace("__MODES__", json.dumps(modes)).replace("__ABOUT__", about).replace("__EXAMPLES__", json.dumps(examples))
    host = args.host or ("0.0.0.0" if args.public else "127.0.0.1")
    httpd = ThreadingHTTPServer((host, args.port), make_handler(server))
    print(f"ZeroMem chat on http://{'localhost' if host == '127.0.0.1' else host}:{args.port}  "
          f"(modes: {', '.join(pipes)}; route={args.route}, provider={args.provider}, min-know {args.min_know}"
          f"{f', DOCUMENTS ONLY ({len(docs.files)} files)' if docs is not None else ''}"
          f"{', PUBLIC: rate-limited, no cache' if args.public else ''}{', LITE' if args.lite else ''})  Ctrl+C to stop",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
