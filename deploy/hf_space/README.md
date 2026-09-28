---
title: ZeroMem
emoji: 🔎
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: A 34M from-scratch model that answers only from sources
---

# ZeroMem

A small language model (about 34M parameters) trained from scratch on a laptop, built to
**never answer from memory**. For each question it:

1. picks where to look (Wikipedia for facts, docs for code, news sites, or the open web),
2. fetches and splits the pages into chunks,
3. reads each chunk with its sentences tagged `[A] [B] [C] ...` and answers `<KNOW>[C]`
   (sentence C answers it) or `<REFUSE>`.

ZeroMem points at a sentence instead of retyping it, so every answer it shows is a real
sentence from a real page, with the link.

**Two modes** (toggle in the top right):
- **ZeroMem only**: the model's own pick, nothing filtered.
- **Full system**: ZeroMem plus a relevance check and a reranker backup.

Held-out results (500 questions, stricter setting): ZeroMem shows a wrong answer on 13.8% of
questions vs 18.4% for an off-the-shelf MiniLM reranker, and refuses correctly 78% vs 62%;
it answers fewer answerable questions (52% vs 76%). About 17 ms per chunk on CPU.

__REPO_LINE__
