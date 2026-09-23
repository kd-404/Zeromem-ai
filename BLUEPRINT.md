# ZeroMem — Blueprint

Research backing this doc: [`reports/ZeroMem feasibility and prior art.md`](reports/ZeroMem%20feasibility%20and%20prior%20art.md)

## What this actually is (after the prior-art check)

Two of the four originally-claimed "novel" pieces already exist in published work.
We keep them anyway because they're the right engineering choice — we just don't
claim to have invented them.

| Piece | Status | We're using |
|---|---|---|
| Zero-parametric-knowledge reader | Prior art: *Knowledgeless Language Models* (arXiv:2607.12831) | The idea, reused honestly |
| Chunk-native attention | Prior art: **Block-Attention** (arXiv:2409.15355, ICLR 2025) | Block-Attention's actual masking scheme |
| KNOW / UNSURE / REFUSE / DONE tokens | Not found published — genuine contribution | Built from scratch |
| Hallucination-rate-vs-quantization-level benchmark, specifically for RAG citation groundedness | Open gap in the literature | Our headline result |

## The one hard rule that keeps this project honest

**Facts only ever enter through the context window. Weights only ever encode skill.**

| Event | Allowed to touch weights? | Allowed to touch the cache? |
|---|---|---|
| A new chunk is fetched from the internet | No | Yes — stored raw, embedded, indexed |
| A query is answered using cached/fetched chunks | No | Read-only lookup |
| Periodic behavior fine-tune (citation format, refusal calibration) | Yes | Reads accumulated (question, chunks, correct-answer) examples, but the *topic* is incidental — only the *skill* is the training target |

If this rule is ever violated (i.e. we fine-tune on "new facts" directly), the project
stops being ZeroMem and becomes a normal small LM with a stale knowledge cutoff —
exactly what it's designed to avoid. Any PR that blurs this gets rejected.

## Pipeline

```
User query
   │
   ▼
Qwen2.5-3B-Q4 (agent brain, mlx-lm)  ─── decides: answer directly / call ZeroMem
   │
   ▼
ZeroMem tool call
   │
   ├─▶ 1. Embed query (bge-small-en) → similarity search against local ChromaDB cache
   │
   ├─▶ 2. Cache hit (similar-enough past query) → skip web, use stored chunks
   │
   ├─▶ 3. Cache miss → Brave Search API → fetch pages (trafilatura) → chunk →
   │       embed → STORE in ChromaDB (permanent) → use these chunks
   │
   ▼
ZeroMem reader (our from-scratch transformer)
   - reads top-K reranked chunks (cross-encoder rerank first)
   - chunk-native attention: chunks can't attend to each other, only the
     query/answer segment attends to all of them (Block-Attention scheme)
   - emits KNOW / UNSURE / REFUSE token + cited answer
   │
   ▼
Qwen2.5-3B synthesizes final response with citations
   │
   ▼
User sees answer + sources
```

## Components and where they live

| Component | Path | Status |
|---|---|---|
| Transformer architecture | `zeromem/model/my_transformer.py` | building now |
| Tokenizer (BPE, custom special tokens) | `zeromem/tokenizer/train_tokenizer.py` | building now |
| Web fetch + chunk | `zeromem/scraper/fetch.py` | building now |
| Semantic cache (ChromaDB) | `zeromem/cache/vector_store.py` | building now |
| Stage-1 training (TinyStories, general fluency) | `zeromem/train/train_stage1.py` | next |
| Stage-2 fine-tune (citation/refusal skill) | `zeromem/train/train_stage2.py` | later — needs synthetic data pipeline |
| Agent orchestration (Qwen brain + tool calling) | `zeromem/agent/` | later |
| Eval harness (hallucination vs. quantization) | `zeromem/eval/` | later |
| API / UI | `zeromem/api/` | later |

## Special tokens (reserved in tokenizer at training time — never resize embeddings later)

`<PAD>` `<BOS>` `<EOS>` `<CHUNK_SEP>` `<KNOW>` `<UNSURE>` `<REFUSE>` `<DONE>`

## Model config (v1 target — tune after first training run)

- vocab_size: 16384
- d_model: 512
- n_layers: 8
- n_heads: 8 (head_dim 64)
- d_ff: 1408 (SwiGLU)
- max_seq_len: 2048
- rope_theta: 10000.0
- ~34M params, tied embeddings

## Known risks carried forward from research (see full report for detail)

1. Refusal-following behavior at 30M params is untested in the literature at this scale — validate early with a small synthetic refusal set before committing to the full pipeline.
2. Chunk-native attention masking may need fine-tuning to avoid accuracy loss (Block-Attention saw ~20pt drop pre-tuning) — budget for this, don't assume zero-shot masking works.
3. DONE-token stopping alone is flagged unreliable in 2026 literature — pair it with a max-hops safety cap in the orchestrator, never trust DONE alone.
4. MPS training memory/time is unbenchmarked at this exact scale — expect to tune batch size empirically, don't pre-commit to a training-time estimate.
5. Cache contamination: a wrong fetched chunk is cheap to fix (delete from cache); a wrong chunk baked into a stage-2 fine-tune is not — always keep a manual review/filter step before stage-2 data goes into training.
