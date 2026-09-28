"""Side experiment: fine-tune the stage-1 model on simple everyday conversations.

NOT part of the ZeroMem reader pipeline (see BLUEPRINT.md's hard rule: chat
data teaches answering from memory, the opposite of the reader's job).
This exists to make the mini-LLM respond to "hi" and to practice
supervised fine-tuning mechanics (loss masking, held-out validation, best-
checkpoint saving) that stage 2 will reuse.

Format (no new tokens needed, embeddings can't be resized anyway):
    User: <text>\\n
    AI: <text>\\n
    ...<EOS>
Loss is computed ONLY on the AI's reply tokens; user turns and the "AI:"
prefix are masked with the PAD id, which the model's cross-entropy ignores.

Usage:
    python -m zeromem.train.finetune_chat
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from zeromem.tokenizer.special_tokens import EOS, PAD
from zeromem.train.generate import load_model
from zeromem.train.train_stage1 import cosine_lr, get_device

DATASET = "HuggingFaceTB/everyday-conversations-llama3.1-2k"


def encode_conversation(messages: list[dict], tokenizer: Tokenizer, max_len: int) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        text = m["content"].strip()
        if m["role"] == "user":
            toks = tokenizer.encode(f"User: {text}\n").ids
            ids += toks
            labels += [PAD] * len(toks)
        else:
            prefix = tokenizer.encode("AI:").ids
            body = tokenizer.encode(f" {text}\n").ids
            ids += prefix + body
            labels += [PAD] * len(prefix) + body
    ids.append(EOS)
    labels.append(EOS)
    return ids[:max_len], labels[:max_len]


def make_batches(examples: list[tuple[list[int], list[int]]], batch_size: int, shuffle: bool, max_len: int = 384):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i:i + batch_size]]
        # EVERY batch is exactly (batch_size, max_len - 1): padded rows and columns are all PAD,
        # which the loss ignores. Varying shapes made the MPS backend cache a kernel set per
        # shape and the process grew to 6.8GB of an 8GB machine and thrashed (measured). Stage 1
        # never had this problem because its batches were always one fixed shape.
        width = max_len
        x = torch.full((batch_size, width - 1), PAD, dtype=torch.long)
        y = torch.full((batch_size, width - 1), PAD, dtype=torch.long)
        for r, (ids, labels) in enumerate(chunk):
            n = len(ids) - 1
            x[r, :n] = torch.tensor(ids[:-1])
            y[r, :n] = torch.tensor(labels[1:])
        yield x, y


@torch.no_grad()
def eval_loss(model, examples, device, batch_size: int, max_len: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in make_batches(examples, batch_size, shuffle=False, max_len=max_len):
        _, loss = model(x.to(device), targets=y.to(device))
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints/stage1/latest.pt")
    parser.add_argument("--tokenizer", default="zeromem/tokenizer/trained/tokenizer.json")
    parser.add_argument("--out-dir", default="checkpoints/chat")
    parser.add_argument("--epochs", type=int, default=3)
    # 8 * 384 = 3072 tokens/step, under the ~8192 MPS cliff measured in stage 1.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    tokenizer = Tokenizer.from_file(args.tokenizer)

    ds = load_dataset(DATASET)
    train_rows, val_rows = ds["train_sft"], ds["test_sft"]
    train_ex = [encode_conversation(r["messages"], tokenizer, args.max_len) for r in train_rows]
    val_ex = [encode_conversation(r["messages"], tokenizer, args.max_len) for r in val_rows]
    print(f"train {len(train_ex):,} convos, val {len(val_ex):,} convos, "
          f"avg {sum(len(i) for i, _ in train_ex) / len(train_ex):.0f} tokens each")

    model = load_model(args.ckpt, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_ex) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup = 20
    os.makedirs(args.out_dir, exist_ok=True)

    base_val = eval_loss(model, val_ex, device, args.batch_size, args.max_len)
    print(f"before fine-tuning: val loss {base_val:.4f}  (stage-1 model has never seen this format)")

    # Only save when we beat the checkpoint we started from (so resuming from best.pt can't
    # overwrite it with something worse).
    best_val, step, t0 = base_val, 0, time.time()
    for epoch in range(args.epochs):
        run_loss, run_n = 0.0, 0
        for x, y in make_batches(train_ex, args.batch_size, shuffle=True, max_len=args.max_len):
            lr = cosine_lr(step, total_steps, warmup, args.lr, args.lr * 0.1)
            for g in optimizer.param_groups:
                g["lr"] = lr
            _, loss = model(x.to(device), targets=y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run_loss += loss.item()
            run_n += 1
            step += 1
            if step % 50 == 0 and device.type == "mps":
                torch.mps.empty_cache()  # belt and braces against allocator growth
            if step % 25 == 0:
                print(f"  step {step}/{total_steps} train loss {run_loss / run_n:.4f} "
                      f"elapsed {(time.time() - t0) / 60:.1f}m", flush=True)

        val = eval_loss(model, val_ex, device, args.batch_size, args.max_len)
        print(f"epoch {epoch + 1}/{args.epochs}: train {run_loss / run_n:.4f}  val {val:.4f}"
              f"  (val ppl {math.exp(val):.2f})", flush=True)
        if val < best_val:
            best_val = val
            torch.save({"model": model.state_dict(), "config": model.cfg, "step": step, "val_loss": val},
                       os.path.join(args.out_dir, "best.pt"))
            print(f"  new best val -> saved {args.out_dir}/best.pt", flush=True)

    print(f"done. best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
