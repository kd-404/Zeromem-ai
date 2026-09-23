"""Download the real TinyStories corpus from its correct, canonical source.

Source: https://huggingface.co/datasets/roneneldan/TinyStories
(the dataset from Eldan & Li, "TinyStories: How Small Can Language Models
Be and Still Speak Coherent English?", arXiv:2305.07759 — cited in
BLUEPRINT.md as the precedent for what a ~30M model can learn from this
corpus). This is stage-1 data: general English fluency, zero citation
behavior — that comes later from our own synthetic stage-2 data.

Usage:
    python -m zeromem.data.download_tinystories --limit 50000
    python -m zeromem.data.download_tinystories --full   # ~2M stories, ~1.9GB text
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def download(out_path: str, split: str, limit: int | None) -> None:
    print(f"Loading roneneldan/TinyStories [{split}] from Hugging Face Hub...")
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=limit is not None)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            f.write(text + "\n\n")
            n += 1
            if limit is not None and n >= limit:
                break
            if n % 10000 == 0:
                print(f"  ...{n} stories written")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done: {n} stories -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=50_000,
                         help="Number of stories to pull. Ignored if --full is passed.")
    parser.add_argument("--full", action="store_true",
                         help="Download the entire split (~2M stories for train, ~1.9GB text).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    limit = None if args.full else args.limit
    out = args.out or f"zeromem/data/raw/tinystories_{args.split}.txt"
    download(out, args.split, limit)
