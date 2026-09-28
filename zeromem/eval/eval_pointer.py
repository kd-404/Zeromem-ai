"""Scoreboard: pointer-ZeroMem vs the off-the-shelf reranker, on held-out questions.

Same chunks, same sentences, same gold answers for both:
    ZeroMem   reads the lettered chunk -> <KNOW>[C] or <REFUSE>
    reranker  cross-encoder/ms-marco-MiniLM-L-6-v2 scores every sentence; shows the top one
              if its score >= threshold (the pipeline's relevance bar), else refuses

What matters most is "shows a WRONG answer" (lower is better) together with "answers
correctly" (higher is better). If ZeroMem beats the reranker on both, your model is doing a
job an off-the-shelf model can't; that is the result to report.

    python -m zeromem.eval.eval_pointer                       # 500 held-out examples
    python -m zeromem.eval.eval_pointer --n 1978 --min-know 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

from zeromem.data.pointer_format import MARKERS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/pointer/best.pt")
    ap.add_argument("--val", default="zeromem/data/processed/pointer1_val.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-know", type=float, default=0.5, help="ZeroMem answers when P(KNOW) >= this")
    ap.add_argument("--rerank-threshold", type=float, nargs="+", default=[0.0, 2.0])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder

    from zeromem.reader import PointerReader
    reader = PointerReader(args.ckpt, device=args.device, min_know=args.min_know)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    rows = [json.loads(l) for l in open(args.val, encoding="utf-8")][: args.n]

    z, per_kind = Counter(), defaultdict(Counter)
    ref = {t: Counter() for t in args.rerank_threshold}
    t0 = time.time(); zsecs = 0.0
    for b in range(0, len(rows), args.batch):
        batch = rows[b:b + args.batch]
        for r in batch:
            t = time.time()
            out = reader.read_sentences(r["question"], [r["sentences"]])[0]  # exactly the training sentences
            zsecs += time.time() - t
            gold = r["gold_index"]
            pick = MARKERS.index(out.pick) if out.pick else None
            if r["answerable"]:
                z["ans"] += 1
                z["correct"] += out.verdict == "KNOW" and pick == gold
                z["wrong"] += out.verdict == "KNOW" and pick != gold
                z["refused_answerable"] += out.verdict != "KNOW"
            else:
                z["ref"] += 1; per_kind[r["kind"]]["n"] += 1
                ok = out.verdict != "KNOW"
                z["refused_ok"] += ok; per_kind[r["kind"]]["ok"] += ok
                z["wrong"] += not ok
            # reranker alone
            scores = ce.predict([(r["question"], s) for s in r["sentences"]], show_progress_bar=False)
            top = int(max(range(len(scores)), key=lambda i: scores[i]))
            for th, c in ref.items():
                shown = float(scores[top]) >= th
                if r["answerable"]:
                    c["correct"] += shown and top == gold
                    c["wrong"] += shown and top != gold
                else:
                    c["refused_ok"] += not shown
                    c["wrong"] += shown
        print(f"\r  {min(b + args.batch, len(rows))}/{len(rows)} examples ({time.time() - t0:.0f}s)", end="", flush=True)

    A, R, N = max(z["ans"], 1), max(z["ref"], 1), max(z["ans"] + z["ref"], 1)
    pc = lambda a, d: f"{100 * a / d:5.1f}%"
    print(f"\n\n=== {args.ckpt} | {len(rows)} held-out examples ({z['ans']} answerable, {z['ref']} should-refuse) ===\n")
    head = f"{'':34s}{'ZeroMem':>10s}" + "".join(f"{'reranker≥' + format(t, 'g'):>14s}" for t in args.rerank_threshold)
    print(head); print("─" * len(head))
    row = lambda name, zv, rv: print(f"{name:34s}{zv:>10s}" + "".join(f"{v:>14s}" for v in rv))
    row("answers correctly (of answerable)", pc(z["correct"], A), [pc(c["correct"], A) for c in ref.values()])
    row("refuses when it should", pc(z["refused_ok"], R), [pc(c["refused_ok"], R) for c in ref.values()])
    row("shows a WRONG answer (of all)", pc(z["wrong"], N), [pc(c["wrong"], N) for c in ref.values()])
    print("─" * len(head))
    print(f"ZeroMem refusals by kind: " + ", ".join(f"{k} {pc(c['ok'], max(c['n'], 1))}" for k, c in sorted(per_kind.items())))
    print(f"ZeroMem speed: {1000 * zsecs / len(rows):.0f} ms per chunk (one forward pass, no text generation)")
    print("\nZeroMem wins if it answers correctly MORE and shows wrong answers LESS than the reranker columns.")


if __name__ == "__main__":
    main()
