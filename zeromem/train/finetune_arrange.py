"""Fine-tune ZeroMem on the "arrange" task (format: zeromem/data/arrange_format.py).

Starts from the stage-1 weights, trains ONLY on the target tokens (the
<KNOW>[n] sentence<DONE> / <REFUSE><DONE> part) with the chunk-isolation
attention mask on: chunk i's tokens cannot see other chunks, only the
question and answer see everything.

Built on measured lessons from this machine (8GB M2 Pro, MPS):
  * EVERY batch has one fixed tensor shape. Varying shapes made the GPU backend
    cache kernels per shape and the process ballooned to 6.8GB and thrashed.
  * micro-batch x length stays well under the 8192 tokens/step MPS cliff, and
    gradient accumulation gives a bigger effective batch.
  * attention with an explicit mask is memory-heavy at ~1000 tokens, so the
    micro-batch is small (2).

Checkpoints (resumable, stop with Ctrl+C or kill any time):
  latest.pt  every --ckpt-every optimizer steps: model + optimizer + step
  best.pt    whenever validation target-loss improves (model only)

Validation runs cheap teacher-forced checks every --eval-every steps:
  target loss, refuse-vs-answer decision accuracy (+ refusal precision/recall),
  and chunk-pick accuracy. Exact-quote copy accuracy needs generation and is
  measured separately after training.

Usage:
    python -m zeromem.train.finetune_arrange
    python -m zeromem.train.finetune_arrange --resume checkpoints/arrange/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from zeromem.data.arrange_format import encode_example
from zeromem.tokenizer.special_tokens import KNOW, PAD, REFUSE
from zeromem.train.generate import load_model
from zeromem.train.train_stage1 import cosine_lr, get_device


def load_examples(path: str, tok: Tokenizer, max_len: int, limit: int | None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            ids, labels, cids = encode_example(tok, r["chunks"], r["question"], r["target"])
            if len(ids) <= max_len:
                out.append((ids, labels, cids, r["answerable"]))
    return out


def make_batch(examples, idxs, T: int, key_weight: float = 1.0):
    """Fixed shape (len(idxs), T). Padding: token PAD, label PAD (ignored), chunk id 0.
    Also returns per-token loss weights: the two decision tokens (<KNOW> vs <REFUSE>, and the
    chunk number) get `key_weight`; everything else 1. Those two tokens are 1 of ~35 target
    tokens, so unweighted they barely reach the gradient, and the model sat at the wrong
    answer-vs-refuse split without learning to use the evidence."""
    B = len(idxs)
    x = torch.full((B, T), PAD, dtype=torch.long)
    y = torch.full((B, T), PAD, dtype=torch.long)
    c = torch.zeros((B, T), dtype=torch.long)
    w = torch.ones((B, T))
    for r, i in enumerate(idxs):
        ids, labels, cids, _ = examples[i]
        n = len(ids) - 1
        x[r, :n] = torch.tensor(ids[:-1])
        y[r, :n] = torch.tensor(labels[1:])
        c[r, :n] = torch.tensor(cids[:-1])
        s = next(k for k, l in enumerate(labels) if l != PAD)  # index in ids of the first target token
        w[r, s - 1] = key_weight            # y[s-1] = the verdict token
        if s + 1 < n:
            w[r, s + 1] = key_weight        # y[s+1] = the chunk-number token (answerable examples)
    return x, y, c, w


def weighted_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    V = logits.size(-1)
    per_tok = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD, reduction="none").view_as(y)
    weights = w * (y != PAD)
    return (per_tok * weights).sum() / weights.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, val, device, mb: int, T: int, n: int, digit_ids: list[int]) -> dict:
    model.eval()
    n = (min(n, len(val)) // mb) * mb
    loss_sum = tok_count = 0.0
    tp = fp = fn = tn = 0  # "positive" = the model says REFUSE
    chunk_ok = chunk_n = 0
    first_ok = first_n = 0  # top-1 accuracy of the FIRST word of the quoted sentence
    for start in range(0, n, mb):
        idxs = list(range(start, start + mb))
        x, y, c, _ = make_batch(val, idxs, T)
        logits, _ = model(x.to(device), chunk_ids=c.to(device))
        logits = logits.float().cpu()
        for r, i in enumerate(idxs):
            ids, labels, _, answerable = val[i]
            s = next(k for k, l in enumerate(labels) if l != PAD)  # first target token index in ids
            # per-token target loss
            lp = torch.log_softmax(logits[r, s - 1:len(ids) - 1], dim=-1)
            tgt = torch.tensor(ids[s:len(ids)])
            loss_sum += -lp[torch.arange(len(tgt)), tgt].sum().item()
            tok_count += len(tgt)
            # refuse-vs-answer decision at the first target position
            pred_refuse = logits[r, s - 1, REFUSE] > logits[r, s - 1, KNOW]
            if answerable:
                fp += int(pred_refuse); tn += int(not pred_refuse)
            else:
                tp += int(pred_refuse); fn += int(not pred_refuse)
            # chunk pick: the digit after "<KNOW>["
            if answerable and s + 2 < len(ids) and ids[s + 2] in digit_ids:
                pick = digit_ids[int(torch.argmax(logits[r, s + 1, digit_ids]))]
                chunk_ok += int(pick == ids[s + 2]); chunk_n += 1
            # first token of the quote: after that, copying is easy, so this is the pure
            # 'did it find the right sentence' signal (chance level is low but not zero)
            if answerable and s + 4 < len(ids):
                first_ok += int(int(torch.argmax(logits[r, s + 3])) == ids[s + 4]); first_n += 1
    model.train()
    total = tp + fp + fn + tn
    return {
        "val_loss": loss_sum / max(tok_count, 1),
        "decision_acc": (tp + tn) / max(total, 1),
        "refuse_recall": tp / max(tp + fn, 1),
        "refuse_precision": tp / max(tp + fp, 1),
        "chunk_acc": chunk_ok / max(chunk_n, 1),
        "quote_start_acc": first_ok / max(first_n, 1),
        "n": total,
    }


def save_latest(path, model, optimizer, step, best_val):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": model.cfg,
                "step": step, "best_val": best_val}, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/stage1/latest.pt", help="starting weights (stage 1)")
    ap.add_argument("--resume", default=None, help="resume from a latest.pt of THIS script")
    ap.add_argument("--tokenizer", default="zeromem/tokenizer/trained/tokenizer.json")
    ap.add_argument("--train", default="zeromem/data/processed/arrange_train.jsonl")
    ap.add_argument("--val", default="zeromem/data/processed/arrange_val.jsonl")
    ap.add_argument("--out-dir", default="checkpoints/arrange")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training rows (pilot)")
    ap.add_argument("--key-weight", type=float, default=5.0, help="loss weight on the verdict and chunk-number tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    tok = Tokenizer.from_file(args.tokenizer)
    T = args.max_len - 1

    print("Encoding data ...", flush=True)
    train = load_examples(args.train, tok, args.max_len, args.max_train)
    val = load_examples(args.val, tok, args.max_len, None)
    print(f"train {len(train):,} examples, val {len(val):,} | avg train len "
          f"{sum(len(e[0]) for e in train) / len(train):.0f} tokens", flush=True)
    digit_ids = [tok.encode(f"[{n}]").ids[1] for n in range(1, 5)]

    model = load_model(args.resume or args.ckpt, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    step, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        opt.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck.get("best_val", float("inf"))
        print(f"Resumed at optimizer step {step} (best val loss so far {best_val:.4f})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    eff = args.micro_batch * args.accum
    spe = len(train) // eff
    total = args.epochs * spe
    print(f"device {device} | micro-batch {args.micro_batch} x accum {args.accum} = {eff} examples/step | "
          f"{spe:,} steps/epoch x {args.epochs} epochs = {total:,} steps | fixed shape ({args.micro_batch}, {T})",
          flush=True)

    if step == 0:
        m = evaluate(model, val, device, args.micro_batch, T, args.eval_n, digit_ids)
        print(f"EVAL step 0 (before fine-tuning) | val loss {m['val_loss']:.4f} | decision acc "
              f"{m['decision_acc']:.1%} | chunk acc {m['chunk_acc']:.1%} | quote-start acc {m['quote_start_acc']:.1%}", flush=True)

    t0, run_start = time.time(), step
    last_loss = 0.0
    while step < total:
        epoch, off = divmod(step, spe)
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)  # deterministic, so resume lands on the same data
        for s in range(off, spe):
            if step >= total:
                break
            lr = cosine_lr(step, total, args.warmup, args.lr, args.lr * 0.1)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = order[s * eff:(s + 1) * eff]
            loss_acc = 0.0
            for mi in range(args.accum):
                idxs = batch[mi * args.micro_batch:(mi + 1) * args.micro_batch]
                x, y, c, w = make_batch(train, idxs, T, args.key_weight)
                logits, _ = model(x.to(device), chunk_ids=c.to(device))
                loss = weighted_loss(logits, y.to(device), w.to(device))
                (loss / args.accum).backward()
                loss_acc += loss.item() / args.accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()

            if step % args.log_every == 0:
                el = time.time() - t0
                sps = (step - run_start) / el
                eta_h = (total - step) / sps / 3600
                d = "" if last_loss == 0 else f" (D {loss_acc - last_loss:+.3f})"
                print(f"[{100 * step / total:5.1f}%] step {step:>5}/{total} epoch {epoch + 1}/{args.epochs} "
                      f"loss {loss_acc:.4f}{d} lr {lr:.2e} {1 / sps:.1f}s/step elapsed {el / 3600:.2f}h "
                      f"eta {eta_h:.2f}h", flush=True)
                last_loss = loss_acc

            if step % args.ckpt_every == 0:
                save_latest(os.path.join(args.out_dir, "latest.pt"), model, opt, step, best_val)
                print(f"  saved latest.pt (resume point) at step {step}", flush=True)

            if step % args.eval_every == 0 or step == total:
                m = evaluate(model, val, device, args.micro_batch, T, args.eval_n, digit_ids)
                print(f"EVAL step {step} | val loss {m['val_loss']:.4f} | decision acc {m['decision_acc']:.1%} "
                      f"(refuse recall {m['refuse_recall']:.1%}, precision {m['refuse_precision']:.1%}) | "
                      f"chunk-pick acc {m['chunk_acc']:.1%} | quote-start acc {m['quote_start_acc']:.1%} | n={m['n']}", flush=True)
                with open(os.path.join(args.out_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, **m}) + "\n")
                if m["val_loss"] < best_val:
                    best_val = m["val_loss"]
                    torch.save({"model": model.state_dict(), "config": model.cfg, "step": step,
                                "val_loss": best_val}, os.path.join(args.out_dir, "best.pt"))
                    print(f"  new best val loss {best_val:.4f} -> saved best.pt", flush=True)
                save_latest(os.path.join(args.out_dir, "latest.pt"), model, opt, step, best_val)

    save_latest(os.path.join(args.out_dir, "latest.pt"), model, opt, step, best_val)
    print(f"DONE. best val loss {best_val:.4f}. Weights: {args.out_dir}/best.pt", flush=True)


if __name__ == "__main__":
    main()
