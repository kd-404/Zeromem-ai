"""Download Simple English Wikipedia as "general reading practice" text for ZeroMem.

Why: stage 1 only read children's stories, so ZeroMem never learned to match a
question to a sentence in encyclopedic text (measured: it answers every question with
the same sentence). Simple Wikipedia is real factual prose in short, plain sentences: a
good bridge between TinyStories and full web text.

Note on the "zero memorized knowledge" rule: any general pretraining stores some facts in
the weights. That is unavoidable for learning to read. The rule that still holds is that
ANSWERS come only from retrieved chunks, and every quote is verified against its chunk.

Output: one article per blank-line-separated block (same layout the tokenizer/training
code already reads), plus a ~1% held-out validation file of whole articles.

Usage:
    python -m zeromem.data.download_simplewiki
"""

from __future__ import annotations

import argparse
import os
import random

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="zeromem/data/raw")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--min-words", type=int, default=60, help="skip stubs shorter than this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading wikimedia/wikipedia 20231101.simple ...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    print(f"{len(ds):,} articles", flush=True)

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tr_path = os.path.join(args.out_dir, "simplewiki_train.txt")
    va_path = os.path.join(args.out_dir, "simplewiki_val.txt")
    n_tr = n_va = words = 0
    with open(tr_path, "w", encoding="utf-8") as ftr, open(va_path, "w", encoding="utf-8") as fva:
        for row in ds:
            text = row["text"].strip()
            if len(text.split()) < args.min_words:
                continue
            # One article = one block: collapse blank lines so the block splitter keeps it whole.
            article = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if rng.random() < args.val_frac:
                fva.write(article + "\n\n"); n_va += 1
            else:
                ftr.write(article + "\n\n"); n_tr += 1
                words += len(article.split())
    print(f"train: {n_tr:,} articles, ~{words / 1e6:.1f}M words -> {tr_path} ({os.path.getsize(tr_path) / 1e6:.0f} MB)")
    print(f"val:   {n_va:,} articles -> {va_path}")


if __name__ == "__main__":
    main()
