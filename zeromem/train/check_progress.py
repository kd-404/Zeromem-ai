"""Check on a running (or stopped) stage-1 training run from any terminal,
without touching the training process itself.

Usage:
    python -m zeromem.train.check_progress
    python -m zeromem.train.check_progress --log logs/stage1_train.log --ckpt-dir checkpoints/stage1
"""

from __future__ import annotations

import argparse
import glob
import os
import re

LOG_LINE_RE = re.compile(
    r"\[\s*([\d.]+)%\] step\s+([\d,]+)/([\d,]+) loss ([\d.]+) \(Dloss ([^)]+)\) "
    r"lr ([\d.eE+-]+) ([\d,]+) tok/s elapsed ([\d.]+)h eta ([\d.]+)h"
)


def find_last_progress_line(log_path: str) -> dict | None:
    if not os.path.exists(log_path):
        return None

    last_match = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if m:
                last_match = m

    if last_match is None:
        return None

    pct, step, total_steps, loss, dloss, lr, tok_s, elapsed_h, eta_h = last_match.groups()
    return {
        "pct": float(pct),
        "step": int(step.replace(",", "")),
        "total_steps": int(total_steps.replace(",", "")),
        "loss": float(loss),
        "dloss": dloss,
        "lr": float(lr),
        "tok_s": int(tok_s.replace(",", "")),
        "elapsed_h": float(elapsed_h),
        "eta_h": float(eta_h),
    }


def list_checkpoints(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/stage1_train.log")
    parser.add_argument("--ckpt-dir", default="checkpoints/stage1")
    args = parser.parse_args()

    print(f"Log file:        {args.log}")
    print(f"Checkpoint dir:  {args.ckpt_dir}")
    print()

    progress = find_last_progress_line(args.log)
    if progress is None:
        print("No progress lines found yet. If training just started, it may still be "
              "tokenizing the corpus (one-time cost on first run; cached after that) — "
              "check back in a few minutes.")
    else:
        print(f"Step:            {progress['step']:,} / {progress['total_steps']:,}  ({progress['pct']:.1f}%)")
        print(f"Loss:            {progress['loss']:.4f}  (Dloss {progress['dloss']})")
        print(f"Learning rate:   {progress['lr']:.2e}")
        print(f"Throughput:      {progress['tok_s']:,} tok/s")
        print(f"Elapsed:         {progress['elapsed_h']:.2f}h (this run)")
        print(f"Estimated left:  {progress['eta_h']:.2f}h")

    print()
    ckpts = list_checkpoints(args.ckpt_dir)
    latest = os.path.join(args.ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        print(f"latest.pt exists -> safe to resume with:")
        print(f"  python -m zeromem.train.train_stage1 --data <...> --tokenizer <...> --resume {latest}")
    elif ckpts:
        print(f"{len(ckpts)} step checkpoint(s) found, most recent: {ckpts[-1]}")
        print(f"  python -m zeromem.train.train_stage1 --data <...> --tokenizer <...> --resume {ckpts[-1]}")
    else:
        print("No checkpoints saved yet.")


if __name__ == "__main__":
    main()
