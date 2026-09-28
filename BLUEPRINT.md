# ZeroMem — Blueprint

A Perplexity-style answer engine that runs on one 8GB Mac, where a small transformer
written from scratch (ZeroMem, ~34M params) is the only "brain". No LLM, no API model.
Ask a question; it searches the web, scrapes pages, and returns a sentence quoted from a
source with its URL, or says it could not verify an answer.

Research backing this doc: [`reports/ZeroMem feasibility and prior art.md`](reports/ZeroMem%20feasibility%20and%20prior%20art.md)

## What is and isn't novel (after the prior-art check)

| Piece | Status |
|---|---|
| Zero-parametric-knowledge reader | Prior art: *Knowledgeless Language Models* (arXiv:2607.12831). We follow the idea; general pretraining still stores some facts (unavoidable for learning to read). |
| Chunk-isolation attention | Prior art: Block-Attention (arXiv:2409.15355) and Fusion-in-Decoder. Implemented in `my_transformer.py`. |
| KNOW / REFUSE / DONE control tokens | Not found published in this combination. |
| Verify-or-refuse output rule (below) | Our design. |

## Hard rules

1. **Facts enter only through the context window.** Scraped text is never used to train weights.
   Weights encode skill (reading, copying, refusing), not answers.
2. **Verify or refuse.** A quote is shown only if (a) it is verbatim inside the chunk it was read
   from, and (b) a cross-encoder scores it relevant to the question (>= 0; calibrated on held-out
   data: keeps 84% of correct sentences, lets through 8% of wrong ones). Otherwise: "no verified answer".
3. **Sanitize scraped text.** Literal control-token text (`<REFUSE>`, `<KNOW>`...) on a web page is
   stripped before tokenizing, so a page cannot inject control tokens.

## Pipeline (as built: `python -m zeromem.pipeline "question"`)

```
question
  1. cache     bge-small + ChromaDB: reuse chunks we already fetched
  2. search    ddgs (no key) | brave | tavily | exa          zeromem/scraper/search.py
  3. scrape    parallel fetch, trafilatura, 120-word chunks   zeromem/scraper/fetch.py
  4. store     new chunks embedded into the cache (grows over time)
  5. rerank    MiniLM cross-encoder picks the top-k chunks
  6. read      ZeroMem reads each top chunk: <KNOW>[1] sentence<DONE> or <REFUSE><DONE>
                 -> quote must be verbatim in the chunk AND relevant to the question
  7. answer    verified quote + source URL, or "no verified answer"
```

## Status

| Component | Path | State |
|---|---|---|
| Transformer (RoPE, RMSNorm, SwiGLU, chunk masks) | `zeromem/model/my_transformer.py` | done, tested |
| Tokenizer (byte BPE, 16k, 8 reserved control tokens) | `zeromem/tokenizer/` | done |
| Stage 1: TinyStories (fluent English) | `zeromem/train/train_stage1.py` | done, val loss 1.59 |
| Reading practice: Simple Wikipedia (`--init-from`) | same script | done, held-out loss 8.24 -> 2.11 |
| "Arrange" task dataset (SQuAD 2.0 reshaped) | `zeromem/data/build_arrange_dataset.py` | done, labels verified |
| Arrange fine-tune | `zeromem/train/finetune_arrange.py` | in progress (see findings) |
| Search, scrape, cache, rerank, reader, pipeline | `zeromem/scraper/`, `cache/`, `reader.py`, `pipeline.py` | built and run end to end |
| Evaluation harness (generation-based, per question) | `zeromem/eval/` | not built yet |
| API / UI | | not started |

## Measured findings (do not lose these)

- **Story-only ZeroMem cannot do the job.** After arrange fine-tuning it answered every question with
  the same sentence, and its quotes were corrupted copies. Loss looked fine; behavior did not.
  Always judge with `python -m zeromem.train.ask_arrange --demo`, not with loss alone.
- **Reading practice fixed copying**: quotes became exact verbatim sentences.
- **Still open: matching the question to the right sentence.** The pretrained MiniLM reranker gets
  86.6% top-1 sentence selection on our validation set; ZeroMem's teacher-forced metrics were at
  chance. The pipeline's relevance check protects against wrong-but-real quotes in the meantime.
- Summary metrics can mislead: "quote-start accuracy" mostly measured the habit of starting with "The".
- A model can use shortcuts: with short chunks it refused everything (it learned "short = junk page").

## Machine lessons (8GB M2 Pro, MPS)

- Keep batch_size x seq_len under ~8,192 tokens/step: above it throughput collapses ~36x.
- Every batch must have one fixed tensor shape, or the process balloons in memory and thrashes.
- Explicit-mask attention at ~1,000 tokens is memory-heavy; use small micro-batches + accumulation.
- Closing the lid sleeps the Mac regardless of `caffeinate` (needs an external display).
- Quit heavy apps (VS Code held ~4GB) before long runs; the run slows down under memory pressure.

## Removed

The Qwen "agent brain" was dropped from the plan. The 4-bit weights remain in the Hugging Face
cache (~1.7GB) and `mlx-lm` is no longer a requirement.

## Next

1. Finish the one-chunk arrange fine-tune from the reading-practice weights; judge it with the demo prompts.
2. Build a generation-based evaluation: over N held-out questions, how often does ZeroMem's quote equal
   the gold sentence, how often is it verified, relevant, and refused correctly (vs. the reranker alone).
3. If matching is learned: extend to multi-chunk. If not: more reading practice, then a bigger model.
4. UNSURE token data (partial evidence), API and UI.
