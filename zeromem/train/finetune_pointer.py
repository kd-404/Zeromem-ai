"""Fine-tune ZeroMem on the POINTER task (format: zeromem/data/pointer_format.py).

ZeroMem reads a question + one chunk whose sentences are tagged [A] [B] [C] ... and learns to
answer <KNOW>[C]<DONE> (sentence C answers it) or <REFUSE><DONE>. It never retypes text, so
it can't corrupt a quote; its whole job is the decision: answer or refuse, and which sentence.

Starts from the copy-trained reader (checkpoints/arrange_1chunk_rp/best.pt), which already
knows how to read a chunk against a question. Loss is only on the answer tokens, with extra
weight on the two decisions (verdict + letter). Same memory rules as finetune_arrange.py: one
fixed batch shape, micro-batch x length under the MPS cliff.

LIVE VIEW in the terminal:
  * a progress bar that updates every step: loss, pick accuracy and refuse accuracy on the
    training batches just seen, speed, ETA
  * every --eval-every steps: a results table on held-out data + a few real questions showing
    which sentence ZeroMem picked vs the right one
  * Ctrl+C saves a resume point and prints the command to continue

Usage (run from the folder that contains zeromem/):
    python -m zeromem.data.build_pointer_dataset                 # once: make the data
    python -m zeromem.train.finetune_pointer --max-train 64 --epochs 1 --eval-every 4 --eval-n 32   # 2-min smoke test
    python -m zeromem.train.finetune_pointer                     # the real run
    python -m zeromem.train.finetune_pointer --resume checkpoints/pointer/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
from tokenizers import Tokenizer

from zeromem.data.arrange_format import encode_example
from zeromem.data.pointer_format import MARKERS
from zeromem.tokenizer.special_tokens import DONE, KNOW, PAD, REFUSE
from zeromem.train.finetune_arrange import make_batch, save_latest, weighted_loss
from zeromem.train.generate import load_model
from zeromem.train.train_stage1 import cosine_lr, get_device

TTY = sys.stdout.isatty()
BAR = 28


def marker_ids(tok: Tokenizer) -> list[int]:
    ids = [tok.token_to_id(m) for m in MARKERS]
    assert all(i is not None for i in ids), "tokenizer is missing a single-letter token A..Z"
    t = tok.encode("<KNOW>[C]<DONE>").ids
    assert t == [KNOW, tok.token_to_id("["), ids[2], tok.token_to_id("]"), DONE], \
        f"'<KNOW>[C]<DONE>' must be 5 tokens (verdict, [, letter, ], done); got {t}"
    return ids


def load_rows(path: str, tok: Tokenizer, max_len: int, limit: int | None):
    """-> (examples for make_batch, meta). examples[i] = (ids, labels, chunk_ids, answerable);
    meta[i] = {n: sentences in the chunk, gold: gold index or None, q, sentences}."""
    ex, meta, too_long = [], [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            ids, labels, cids = encode_example(tok, r["chunks"], r["question"], r["target"])
            if len(ids) > max_len:
                too_long += 1
                continue
            ex.append((ids, labels, cids, r["answerable"]))
            meta.append({"n": len(r["sentences"]), "gold": r["gold_index"], "q": r["question"],
                         "sentences": r["sentences"], "kind": r["kind"]})
    return ex, meta, too_long


def decisions(logits: torch.Tensor, ex, meta, idxs, mids: list[int]):
    """For each row: (said_know, picked_index or None). Reads the verdict at the position before
    the first answer token and the letter at the position after '['. Letters past the chunk's
    last sentence are masked, exactly as at inference. Only these few logits leave the GPU
    (copying the whole (B, T, vocab) tensor every step would cost ~270 MB of transfers)."""
    rows = torch.arange(len(idxs), device=logits.device)
    starts = [next(k for k, l in enumerate(ex[i][1]) if l != PAD) for i in idxs]
    s = torch.tensor(starts, device=logits.device)
    verdict = logits[rows, s - 1][:, [KNOW, REFUSE]].float().cpu()                 # (B, 2)
    letters = logits[rows, (s + 1).clamp(max=logits.size(1) - 1)][:, mids].float().cpu()  # (B, 26)
    out = []
    for r, i in enumerate(idxs):
        know = bool(verdict[r, 0] > verdict[r, 1])
        pick = None
        if ex[i][3]:
            n = max(1, meta[i]["n"])
            pick = int(torch.argmax(letters[r, :n]))
        out.append((know, pick))
    return out


@torch.no_grad()
def evaluate(model, val, vmeta, device, mb, T, n, mids):
    model.eval()
    n = (min(n, len(val)) // mb) * mb
    c = dict(ans=0, ref=0, know_on_ans=0, pick_ok=0, correct_shown=0, wrong_shown=0, refused_ok=0, said_refuse=0,
             loss=0.0, toks=0)
    for start in range(0, n, mb):
        idxs = list(range(start, start + mb))
        x, y, cc, _ = make_batch(val, idxs, T)
        logits, _ = model(x.to(device), chunk_ids=cc.to(device))
        yd = y.to(device)
        per = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yd.reshape(-1),
                                                ignore_index=PAD, reduction="sum")
        c["loss"] += per.item(); c["toks"] += int((y != PAD).sum())
        for (know, pick), i in zip(decisions(logits, val, vmeta, idxs, mids), idxs):
            if val[i][3]:
                c["ans"] += 1; c["know_on_ans"] += know
                ok = pick == vmeta[i]["gold"]
                c["pick_ok"] += ok
                c["correct_shown"] += know and ok
                c["wrong_shown"] += know and not ok
            else:
                c["ref"] += 1; c["refused_ok"] += not know
                c["wrong_shown"] += know
            c["said_refuse"] += not know
    model.train()
    A, R = max(c["ans"], 1), max(c["ref"], 1)
    return {
        "val_loss": c["loss"] / max(c["toks"], 1),
        "decision_acc": (c["know_on_ans"] + c["refused_ok"]) / max(c["ans"] + c["ref"], 1),
        "pick_acc": c["pick_ok"] / A,                    # right sentence, given it answers
        "correct_shown": c["correct_shown"] / A,         # end to end: answers AND picks right
        "refuse_recall": c["refused_ok"] / R,            # refuses when it should
        "refuse_precision": c["refused_ok"] / max(c["said_refuse"], 1),
        "wrong_shown": c["wrong_shown"] / max(c["ans"] + c["ref"], 1),  # harmful: shows a wrong answer
        "chance_pick": sum(1 / max(m["n"], 1) for m, e in zip(vmeta[:n], val[:n]) if e[3]) / A,
        "n": c["ans"] + c["ref"],
    }


@torch.no_grad()
def show_examples(model, val, vmeta, device, T, mids, k, rng):
    """Print k held-out questions with ZeroMem's pick next to the right answer."""
    model.eval()
    idxs = rng.sample(range(len(val)), min(k, len(val)))
    x, _, cc, _ = make_batch(val, idxs, T)
    logits, _ = model(x.to(device), chunk_ids=cc.to(device))
    for (know, pick), i in zip(decisions(logits, val, vmeta, idxs, mids), idxs):
        m = vmeta[i]
        short = lambda s: (s[:95] + "...") if len(s) > 98 else s
        print(f"   Q: {short(m['q'])}")
        if not val[i][3]:
            verdict = "REFUSE  ✓" if not know else "answered ✗ (should refuse)"
            print(f"      should refuse ({m['kind']}) -> ZeroMem: {verdict}")
        elif not know:
            print(f"      ZeroMem: REFUSE ✗   right answer [{MARKERS[m['gold']]}] {short(m['sentences'][m['gold']])}")
        elif pick == m["gold"]:
            print(f"      ZeroMem: [{MARKERS[pick]}] {short(m['sentences'][pick])}  ✓")
        else:
            print(f"      ZeroMem: [{MARKERS[pick]}] {short(m['sentences'][pick])}  ✗")
            print(f"      right:   [{MARKERS[m['gold']]}] {short(m['sentences'][m['gold']])}")
    model.train()


def print_eval(step, m, best_val, prev):
    arrow = lambda key, higher=True: "" if prev is None else (
        " ▲" if (m[key] > prev[key]) == higher and m[key] != prev[key] else " ▼" if m[key] != prev[key] else "")
    print(f"\n{'─' * 22} EVAL @ step {step} ({m['n']} held-out examples) {'─' * 22}")
    print(f"  answers correctly (right sentence)   {m['correct_shown']:6.1%}{arrow('correct_shown')}")
    print(f"  picks the right sentence             {m['pick_acc']:6.1%}{arrow('pick_acc')}   (random guess: {m['chance_pick']:.0%})")
    print(f"  answer-or-refuse decision            {m['decision_acc']:6.1%}{arrow('decision_acc')}")
    print(f"  refuses when it should               {m['refuse_recall']:6.1%}{arrow('refuse_recall')}   "
          f"(its refusals that were right: {m['refuse_precision']:.1%})")
    print(f"  shows a WRONG answer                 {m['wrong_shown']:6.1%}{arrow('wrong_shown', higher=False)}   <- lower is better")
    print(f"  val loss                             {m['val_loss']:.4f}" + ("   new best -> best.pt" if m["val_loss"] < best_val else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/arrange_1chunk_rp/best.pt", help="starting weights")
    ap.add_argument("--resume", default=None, help="continue from a latest.pt of THIS script")
    ap.add_argument("--tokenizer", default="zeromem/tokenizer/trained/tokenizer.json")
    ap.add_argument("--train", default="zeromem/data/processed/pointer1_train.jsonl")
    ap.add_argument("--val", default="zeromem/data/processed/pointer1_val.jsonl")
    ap.add_argument("--out-dir", default="checkpoints/pointer")
    ap.add_argument("--max-len", type=int, default=512, help="one-chunk examples are ~350 tokens")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--examples", type=int, default=3, help="sample questions printed at each eval")
    ap.add_argument("--max-train", type=int, default=None, help="first N training rows only (smoke test)")
    ap.add_argument("--key-weight", type=float, default=5.0, help="loss weight on the verdict and letter tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    tok = Tokenizer.from_file(args.tokenizer)
    mids = marker_ids(tok)
    T = args.max_len - 1

    print("Loading data ...", flush=True)
    train, tmeta, tl1 = load_rows(args.train, tok, args.max_len, args.max_train)
    val, vmeta, tl2 = load_rows(args.val, tok, args.max_len, None)
    print(f"  train {len(train):,} | val {len(val):,} | skipped as too long: {tl1 + tl2} | "
          f"avg length {sum(len(e[0]) for e in train) / len(train):.0f} tokens", flush=True)
    eff = args.micro_batch * args.accum
    if len(train) < eff:
        sys.exit(f"need at least {eff} training rows (micro-batch x accum); got {len(train)}")

    model = load_model(args.resume or args.ckpt, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    step, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        opt.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck.get("best_val", float("inf"))
        print(f"Resumed at step {step} (best val loss {best_val:.4f})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    spe = len(train) // eff
    total = args.epochs * spe
    print(f"device {device} | {eff} examples/step ({args.micro_batch} x {args.accum}) | {spe:,} steps/epoch x "
          f"{args.epochs} = {total:,} steps | batch shape ({args.micro_batch}, {T})", flush=True)
    rng = random.Random(args.seed)
    prev = None
    if step == 0:
        prev = evaluate(model, val, vmeta, device, args.micro_batch, T, args.eval_n, mids)
        print("\nBEFORE fine-tuning (the copy-trained model has never seen letter markers):")
        print_eval(0, prev, float("inf"), None)
        show_examples(model, val, vmeta, device, T, mids, args.examples, rng)
        print()

    t0, run_start = time.time(), step
    win = dict(loss=0.0, n=0, pick_ok=0, pick_n=0, dec_ok=0, dec_n=0)  # rolling window for the live bar
    try:
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
                    with torch.no_grad():  # live accuracy on what it just trained on
                        for (know, pick), i in zip(decisions(logits.detach(), train, tmeta, idxs, mids), idxs):
                            win["dec_n"] += 1
                            win["dec_ok"] += know == train[i][3]
                            if train[i][3]:
                                win["pick_n"] += 1; win["pick_ok"] += pick == tmeta[i]["gold"]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                win["loss"] += loss_acc; win["n"] += 1
                if device.type == "mps" and step % 50 == 0:
                    torch.mps.empty_cache()

                # ---- live progress bar ----
                el = time.time() - t0
                sps = (step - run_start) / max(el, 1e-9)
                eta = (total - step) / max(sps, 1e-9)
                done = int(BAR * step / total)
                line = (f"[{'█' * done}{'░' * (BAR - done)}] {100 * step / total:5.1f}% | step {step}/{total} "
                        f"ep {epoch + 1}/{args.epochs} | loss {win['loss'] / win['n']:.3f} | "
                        f"picks right {win['pick_ok'] / max(win['pick_n'], 1):4.0%} | "
                        f"answer/refuse right {win['dec_ok'] / max(win['dec_n'], 1):4.0%} | "
                        f"{1 / max(sps, 1e-9):.1f}s/step | eta {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m")
                if TTY:
                    print("\r" + line, end="", flush=True)
                if step % 25 == 0:  # keep a permanent line every 25 steps (the only lines in a log file)
                    print() if TTY else print(line, flush=True)
                    win = dict(loss=0.0, n=0, pick_ok=0, pick_n=0, dec_ok=0, dec_n=0)

                if step % args.ckpt_every == 0:
                    save_latest(os.path.join(args.out_dir, "latest.pt"), model, opt, step, best_val)

                if step % args.eval_every == 0 or step == total:
                    if TTY:
                        print()
                    m = evaluate(model, val, vmeta, device, args.micro_batch, T, args.eval_n, mids)
                    print_eval(step, m, best_val, prev)
                    show_examples(model, val, vmeta, device, T, mids, args.examples, rng)
                    prev = m
                    with open(os.path.join(args.out_dir, "metrics.jsonl"), "a") as f:
                        f.write(json.dumps({"step": step, **m}) + "\n")
                    if m["val_loss"] < best_val:
                        best_val = m["val_loss"]
                        torch.save({"model": model.state_dict(), "config": model.cfg, "step": step,
                                    "val_loss": best_val, "format": "pointer"}, os.path.join(args.out_dir, "best.pt"))
                    save_latest(os.path.join(args.out_dir, "latest.pt"), model, opt, step, best_val)
                    print(flush=True)
    except KeyboardInterrupt:
        save_latest(os.path.join(args.out_dir, "latest.pt"), model, opt, step, best_val)
        print(f"\n\nStopped at step {step}. Resume point saved. Continue with:\n"
              f"  python -m zeromem.train.finetune_pointer --resume {args.out_dir}/latest.pt\n")
        return

    save_latest(os.path.join(args.out_dir, "latest.pt"), model, opt, step, best_val)
    print(f"\nDONE. best val loss {best_val:.4f} -> {args.out_dir}/best.pt\n"
          f"Next: python -m zeromem.eval.eval_pointer   (ZeroMem vs the reranker on held-out questions)")


if __name__ == "__main__":
    main()
