"""Stage-1 pretraining: general English fluency on TinyStories.

This is ONLY step 2 of the build order in BLUEPRINT.md — plain next-token
prediction, no chunks, no citation behavior, no confidence tokens yet.
Those come from stage 2's synthetic fine-tune, on top of these weights.

Resumable: stop anytime with Ctrl+C (or `kill`), and restart with
--resume pointing at the last checkpoint (see check_progress.py for the
easiest way to find it). Model weights, optimizer state (momentum, etc.),
and the exact step count are all restored, so a resumed run continues
the learning-rate schedule and optimizer state as if it never stopped —
not a cold restart.

Usage:
    # first run
    python -m zeromem.train.train_stage1 \
        --data zeromem/data/raw/tinystories_train.txt \
        --tokenizer zeromem/tokenizer/trained/tokenizer.json

    # after stopping, resume from the latest checkpoint
    python -m zeromem.train.train_stage1 \
        --data zeromem/data/raw/tinystories_train.txt \
        --tokenizer zeromem/tokenizer/trained/tokenizer.json \
        --resume checkpoints/stage1/latest.pt

Check progress from another terminal at any time with:
    python -m zeromem.train.check_progress
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from zeromem.model.config import ZeroMemConfig
from zeromem.model.my_transformer import ZeroMem
from zeromem.train.dataset import BlockDataset, tokenize_corpus_to_ids


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_or_build_token_cache(data_path: str, tokenizer_path: str) -> np.ndarray:
    """Tokenizing the full 1.8GB corpus takes real time — cache the result
    next to the source file so a resumed/restarted run doesn't pay that
    cost again."""
    cache_path = data_path + ".tokenized.npy"
    if os.path.exists(cache_path):
        print(f"Loading cached tokenized corpus from {cache_path} ...")
        return np.load(cache_path)

    print("Tokenizing corpus (one-time cost, cached to disk for future runs)...")
    token_ids = tokenize_corpus_to_ids(data_path, tokenizer_path)
    np.save(cache_path, token_ids)
    print(f"Cached tokenized corpus -> {cache_path}")
    return token_ids


def _checkpoint_dict(model: ZeroMem, optimizer: torch.optim.Optimizer, cfg: ZeroMemConfig, step: int) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "step": step,
    }


def save_latest(out_dir: str, model: ZeroMem, optimizer: torch.optim.Optimizer, cfg: ZeroMemConfig, step: int) -> None:
    """The frequent, cheap safety net: always overwrites the same file, so
    disk usage never grows no matter how often this is called."""
    torch.save(_checkpoint_dict(model, optimizer, cfg, step), os.path.join(out_dir, "latest.pt"))


def save_milestone(out_dir: str, model: ZeroMem, optimizer: torch.optim.Optimizer, cfg: ZeroMemConfig, step: int) -> None:
    """A permanent, uniquely-named snapshot — called much less often than
    save_latest, since each one is a new ~400MB file that's never deleted."""
    path = os.path.join(out_dir, f"step_{step}.pt")
    torch.save(_checkpoint_dict(model, optimizer, cfg, step), path)


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    cfg = ZeroMemConfig()
    model = ZeroMem(cfg).to(device)
    print(f"Model: {model.num_params():,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1, betas=(0.9, 0.95))

    if args.init_from:
        # Continued pretraining: start from existing WEIGHTS only (fresh optimizer, step 0, new LR schedule).
        # Different from --resume, which restores optimizer state and the step counter of the same run.
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model"])
        print(f"Initialised weights from {args.init_from} (trained to step {init['step']:,}); fresh optimizer", flush=True)

    start_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # weights_only=False: our checkpoints embed a ZeroMemConfig object, not just
        # tensors, and this is always our own local file, never an untrusted download.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed at step {start_step}")

    token_ids = get_or_build_token_cache(args.data, args.tokenizer)
    print(f"Corpus: {len(token_ids):,} tokens")

    dataset = BlockDataset(token_ids, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset):,} training windows of {args.seq_len} tokens, "
          f"{len(loader):,} steps/epoch")

    total_steps = args.epochs * len(loader)
    if start_step >= total_steps:
        print(f"Checkpoint step {start_step} already >= total_steps {total_steps}. Nothing to do — "
              f"increase --epochs if you want to keep training.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Flat step loop across all epochs (rather than nested epoch loops) so
    # resuming is just "start the counter at start_step" — no need to
    # reconstruct which epoch/batch-within-epoch we were on.
    def infinite_loader():
        while True:
            yield from loader

    infinite_batches = infinite_loader()

    t0 = time.time()
    step = start_step
    prev_logged_loss = None  # loss at the last print, so we can show how much it dropped since then

    for x, y in infinite_batches:
        if step >= total_steps:
            break

        x, y = x.to(device), y.to(device)

        lr = cosine_lr(step, total_steps, args.warmup_steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            current_loss = loss.item()
            elapsed = time.time() - t0
            steps_done_this_run = step - start_step + 1
            rate = steps_done_this_run / elapsed if elapsed > 0 else 0.0
            eta_s = (total_steps - step) / rate if rate > 0 else float("inf")
            pct = 100.0 * step / total_steps

            if prev_logged_loss is None:
                delta_str = "n/a"
            else:
                delta = current_loss - prev_logged_loss  # negative = loss dropping (good)
                arrow = "v" if delta < 0 else ("^" if delta > 0 else "=")
                delta_str = f"{arrow}{abs(delta):.4f}"
            prev_logged_loss = current_loss

            print(f"[{pct:5.1f}%] step {step:>7,}/{total_steps:,} "
                  f"loss {current_loss:.4f} (Dloss {delta_str}) "
                  f"lr {lr:.2e} {rate*args.batch_size*args.seq_len:,.0f} tok/s "
                  f"elapsed {elapsed/3600:.2f}h eta {eta_s / 3600:.2f}h")

        if step % args.ckpt_every == 0 and step > start_step:
            save_latest(args.out_dir, model, optimizer, cfg, step)
            print(f"  saved latest.pt (resume point) at step {step}")

        if step % args.milestone_every == 0 and step > start_step:
            save_milestone(args.out_dir, model, optimizer, cfg, step)
            print(f"  saved permanent snapshot -> step_{step}.pt")

        step += 1

    save_latest(args.out_dir, model, optimizer, cfg, step)
    save_milestone(args.out_dir, model, optimizer, cfg, step)
    print(f"Training done. Final checkpoint -> {args.out_dir}/step_{step}.pt (and latest.pt)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to raw text corpus")
    parser.add_argument("--tokenizer", required=True, help="Path to trained tokenizer.json")
    parser.add_argument("--out-dir", default="checkpoints/stage1")
    parser.add_argument("--init-from", default=None,
                        help="start from these weights but with a fresh optimizer/step (continued pretraining on new data)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from (e.g. checkpoints/stage1/latest.pt)")
    # seq_len * batch_size MUST stay under 8192 on an 8GB M2 Pro (MPS) —
    # measured directly: 4096 tokens/step -> ~6,250-6,590 tok/s consistently;
    # 8192 tokens/step -> collapses to ~172-179 tok/s, a 36x cliff, almost
    # certainly MPS's internal memory "high watermark" eviction kicking in.
    # This isn't a gradual slowdown, it's a hard threshold — stay under it.
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    # ~0.75s/step measured -> every 200 steps is ~2.5 minutes of progress at
    # risk at any given time, instead of the ~12 minutes at the old default
    # of 1000. This only touches latest.pt (single file, overwritten each
    # time), so frequent saves don't cost extra disk space.
    parser.add_argument("--ckpt-every", type=int, default=200)
    # Permanent, uniquely-named snapshots (step_N.pt) are separate and much
    # rarer, since each one is a new ~400MB file that's kept forever —
    # these are rollback points, not the resume safety net.
    parser.add_argument("--milestone-every", type=int, default=5000)
    args = parser.parse_args()

    train(args)
