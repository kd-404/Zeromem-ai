"""Train a byte-level BPE tokenizer for ZeroMem.

Why BPE, and why train our own instead of reusing Qwen's or GPT-2's:
  - A 16k vocab (vs. Qwen's ~150k) keeps the embedding table small, which
    matters directly for a 30-50M param budget — at 32k vocab the
    embeddings alone would eat ~16M params.
  - Byte-level BPE means there is no "unknown token" case, ever: any
    string, including a word invented five seconds ago, decomposes into
    known byte/subword pieces. This is why new internet content never
    requires retraining just to be readable (see BLUEPRINT.md).
  - Special tokens are reserved as ids 0-7 up front (see special_tokens.py)
    and never touched again, so we never have to resize embeddings later.

Usage:
    python -m zeromem.tokenizer.train_tokenizer --input data/raw/*.txt --out zeromem/tokenizer/trained
"""

from __future__ import annotations

import argparse
import glob
import os

from tokenizers import ByteLevelBPETokenizer

from zeromem.tokenizer.special_tokens import SPECIAL_TOKENS


def train(input_globs: list[str], out_dir: str, vocab_size: int) -> None:
    files = []
    for pattern in input_globs:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matched {input_globs}. Point --input at your training corpus "
            "(e.g. the TinyStories text dump) before training the tokenizer."
        )

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,  # reserved as the first N ids, in this exact order
    )
    tokenizer.save_model(out_dir)

    # Also save a single-file tokenizer.json for fast loading at inference time.
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

    print(f"Trained tokenizer with vocab_size={tokenizer.get_vocab_size()} -> {out_dir}")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:14s} -> id {tokenizer.token_to_id(tok)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Glob pattern(s) for training text files, e.g. 'zeromem/data/raw/*.txt'",
    )
    parser.add_argument("--out", default="zeromem/tokenizer/trained")
    parser.add_argument("--vocab-size", type=int, default=16384)
    args = parser.parse_args()

    train(args.input, args.out, args.vocab_size)
