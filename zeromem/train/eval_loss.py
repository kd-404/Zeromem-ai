"""Measure loss/perplexity on held-out data vs. a sample of training data.

Validation = the official TinyStories validation split, which was never
part of training. The train-sample number is computed the same way (eval
mode, same window size), so the gap between the two is a clean
overfitting signal: val loss far above train loss means memorization.

Usage:
    python -m zeromem.train.eval_loss
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from zeromem.train.dataset import BlockDataset, tokenize_corpus_to_ids
from zeromem.train.generate import load_model
from zeromem.train.train_stage1 import get_device


@torch.no_grad()
def mean_loss(model, dataset, device, batch_size: int, max_batches: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        _, loss = model(x.to(device), targets=y.to(device))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints/stage1/latest.pt")
    parser.add_argument("--tokenizer", default="zeromem/tokenizer/trained/tokenizer.json")
    parser.add_argument("--val-data", default="zeromem/data/raw/tinystories_validation.txt")
    parser.add_argument("--train-cache", default="zeromem/data/raw/tinystories_train.txt.tokenized.npy")
    parser.add_argument("--seq-len", type=int, default=256)
    # 16 * 256 = 4096 tokens/step, safely under the MPS 8192-token cliff.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=400)
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.ckpt, device)

    val_ids = tokenize_corpus_to_ids(args.val_data, args.tokenizer)
    val_ds = BlockDataset(val_ids, args.seq_len)
    print(f"Validation: {len(val_ids):,} tokens, {len(val_ds):,} windows")

    train_ids = np.load(args.train_cache, mmap_mode="r")
    train_ds = BlockDataset(train_ids, args.seq_len)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_ds), size=args.max_batches * args.batch_size, replace=False)
    train_sample = Subset(train_ds, idx.tolist())

    val_loss = mean_loss(model, val_ds, device, args.batch_size, args.max_batches)
    train_loss = mean_loss(model, train_sample, device, args.batch_size, args.max_batches)

    print(f"\ntrain loss (sampled, eval mode): {train_loss:.4f}  perplexity {math.exp(train_loss):.2f}")
    print(f"val   loss (never seen):         {val_loss:.4f}  perplexity {math.exp(val_loss):.2f}")
    print(f"gap (val - train):               {val_loss - train_loss:+.4f}")


if __name__ == "__main__":
    main()
