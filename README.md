# ZeroMem

A Perplexity-style answer engine where the "brain" is a transformer I wrote from scratch — about 34M parameters, trained and run on one 8GB M2 Pro Mac. No GPT, no Claude, no API model anywhere in the loop.

Ask it something. It searches the web, scrapes a few pages, and gives you back one sentence copied **word for word** from a source, plus the link. If nothing it found actually answers the question, it says so instead of guessing.

```
$ python -m zeromem.pipeline "Who designed the Eiffel Tower?"

"The design of the Eiffel Tower was originated by Maurice Koechlin and Emile
Nouguier, who had discussed ideas for a centrepiece for the 1889 Exposition
Universelle."
https://en.wikipedia.org/wiki/Gustave_Eiffel
```

## Why

Every RAG demo I'd seen "solves hallucination" by handing a found passage to GPT-4 and trusting it to paraphrase faithfully. That's still a model generating free-form text — it can drift, blend two sources, or just make something up with a citation stapled on.

ZeroMem can't do that, because generating isn't an option. Its only job is to **point at a sentence that's already there**, verbatim, or refuse. It cannot free-generate an answer even if it wanted to.

## How it actually works

```
question
  1. cache     check ChromaDB — have we already fetched something relevant?
  2. route     facts -> Wikipedia · code -> docs/Q&A sites · news -> trusted outlets · else -> open web
  3. search    ddgs / brave / tavily / exa -> fetch pages in parallel -> clean with trafilatura -> 120-word chunks
  4. store     embed + cache the new chunks (the cache grows over time)
  5. rerank    a MiniLM cross-encoder scores chunks against the question
  6. read      ZeroMem reads the top chunks and emits either
                 <KNOW> [sentence] <DONE>   or   <REFUSE> <DONE>
  7. verify    the quote must be (a) found verbatim inside the chunk it came from, AND
               (b) scored relevant by the cross-encoder — otherwise it's thrown out
  8. answer    the verified quote + its URL, or "no verified answer"
```

If the routed source comes back empty, or ZeroMem can't verify anything in it, the pipeline falls back to an open web search once before giving up. A wrong route costs a few seconds — never the answer.

## The numbers, straight

I'd rather show the actual eval than a cherry-picked demo:

| What's being measured | Result |
|---|---|
| Reranker: picks the right sentence, top-1 | 86.6% |
| Correctly refuses when it should refuse | 35.8% |
| ...but when it *does* refuse, it's right to | 85.5% |
| Answers correctly when it answers | 77.8% |

Read that middle row carefully — it's the honest problem right now. ZeroMem is conservative about *when* it refuses (it's right most of the time it does refuse), but it still misses a lot of cases it should have refused on. The retrieval/reranking stack is doing a lot of the heavy lifting; the model's own judgment of "is this actually answering the question" is the piece still being trained.

Full write-up of what I tried, what didn't work, and why: [`BLUEPRINT.md`](BLUEPRINT.md).

## What this doesn't solve

Worth being upfront about, since people keep asking the right questions:

- **It can still quote the wrong source.** It can't invent a sentence, but it can faithfully quote a real sentence that happens to be irrelevant or wrong — e.g. two sources disagreeing, or a stale page.
- **No source trust/provenance scoring yet.** Every document is weighted equally going into the reranker. A convincing but wrong (or planted) document can still win.
- **No "I'm not sure" middle state.** Right now it's a binary answer/refuse. A calibrated uncertain state is on the list, not built.
- **Verbatim ≠ true.** Verifying a quote appears in the source doesn't verify the source itself is correct.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m zeromem.pipeline "your question here"
python -m zeromem.pipeline --route wiki "force a specific source"
python -m zeromem.pipeline            # interactive mode
```

Point it at your own documents instead of the web:

```bash
python -m zeromem.chat_app --docs ./my_pdfs_and_docs/
```

Or run the web UI:

```bash
uvicorn zeromem.api.server:app --reload --port 8000
```

## Project layout

```
zeromem/
  model/          the transformer itself (RoPE, RMSNorm, SwiGLU, chunk-isolation attention)
  tokenizer/       byte-BPE, 16k vocab, 8 reserved control tokens
  train/           stage-1 (TinyStories fluency) -> reading practice (Wikipedia) -> pointer fine-tune
  scraper/         search providers, fetch, chunking, routing
  cache/           embedding + ChromaDB vector store
  docs/            your-own-documents mode (PDF/DOCX/XLSX/scanned images)
  eval/            held-out evaluation harness
  api/             FastAPI server + minimal web UI
```

## Status

Actively being trained and iterated on — see [`BLUEPRINT.md`](BLUEPRINT.md) for the running log of what's done, what broke, and what's next.
