"""Generation-based evaluation of ZeroMem as a reader.

The teacher-forced metrics printed during training are cheap but can mislead (for example
"quote-start accuracy" mostly measured the habit of starting with "The"). This script does
what the pipeline does: ZeroMem GENERATES an answer for each held-out question, then we score
the actual output.

Per answerable example (the chunk contains the answer):
    exact     ZeroMem's quote equals the gold sentence
    shown     the pipeline would show it (verbatim in chunk AND relevance >= threshold)
    wrong     shown but NOT the gold sentence   <- the harmful case
Per refusable example (the chunk does not answer it):
    refused   the pipeline would NOT show an answer, by kind (near-miss / off-topic / junk page)
Reference: the pretrained cross-encoder reranker alone (top-scoring sentence), same examples.

Usage:
    python -m zeromem.eval.eval_reader --ckpt checkpoints/arrange_1chunk_rp/best.pt --n 150
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict

import numpy as np

from zeromem.data.build_arrange_dataset import SENT_SPLIT
from zeromem.reader import ZeroMemReader, norm


def split_sentences(chunk: str) -> list[str]:
    out = []
    for line in chunk.split("\n"):
        st = [0] + [m.end() for m in SENT_SPLIT.finditer(line)]
        en = st[1:] + [len(line)]
        out += [norm(line[a:b]) for a, b in zip(st, en) if len(line[a:b].strip()) > 3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/arrange_1chunk_rp/best.pt")
    ap.add_argument("--val", default="zeromem/data/processed/arrange1_val.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-relevance", type=float, default=0.0)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reader = ZeroMemReader(args.ckpt, device=args.device)

    rows = [json.loads(l) for l in open(args.val)][: args.n]
    z = Counter()                       # ZeroMem outcomes
    zk: dict[str, Counter] = defaultdict(Counter)
    ref = Counter()                     # reranker-alone outcomes
    n_ans = n_ref = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        chunk, q = r["chunks"][0], r["question"]
        out = reader.read(q, chunk)
        sc_quote = None
        if out.verdict == "KNOW" and out.verified:
            sc_quote = float(ce.predict([(q, out.quote)], show_progress_bar=False)[0])
        shown = sc_quote is not None and sc_quote >= args.min_relevance

        sents = split_sentences(chunk)
        scores = ce.predict([(q, s) for s in sents], show_progress_bar=False) if sents else [-99]
        top_i = int(np.argmax(scores))
        ref_shown = float(max(scores)) >= args.min_relevance

        if r["answerable"]:
            n_ans += 1
            gold = norm(re.match(r"<KNOW>\[1\] (.*)<DONE>$", r["target"], re.S).group(1))
            z["ans_said_know"] += out.verdict == "KNOW"
            z["ans_verbatim"] += out.verdict == "KNOW" and out.verified
            z["ans_exact"] += out.quote is not None and norm(out.quote) == gold
            z["ans_shown"] += shown
            z["ans_shown_correct"] += shown and norm(out.quote) == gold
            z["ans_shown_wrong"] += shown and norm(out.quote) != gold
            ref["ans_top1_exact"] += bool(sents) and sents[top_i] == gold
            ref["ans_shown_correct"] += ref_shown and bool(sents) and sents[top_i] == gold
            ref["ans_shown_wrong"] += ref_shown and not (bool(sents) and sents[top_i] == gold)
        else:
            n_ref += 1
            zk[r["kind"]]["n"] += 1
            zk[r["kind"]]["refused"] += not shown
            z["ref_refused"] += not shown
            ref["ref_refused"] += not ref_shown
        if i % 25 == 0:
            print(f"  {i}/{len(rows)} done ({time.time() - t0:.0f}s)", flush=True)

    pa = lambda c, d: f"{100 * c / max(d, 1):5.1f}%"
    print(f"\n=== {args.ckpt} | {len(rows)} held-out examples ({n_ans} answerable, {n_ref} should-refuse) ===")
    print("ZeroMem, answerable questions:")
    print(f"  says KNOW                          {pa(z['ans_said_know'], n_ans)}")
    print(f"  quote is verbatim in the chunk     {pa(z['ans_verbatim'], n_ans)}")
    print(f"  quote == the gold sentence         {pa(z['ans_exact'], n_ans)}")
    print(f"  pipeline SHOWS a correct answer    {pa(z['ans_shown_correct'], n_ans)}")
    print(f"  pipeline SHOWS a WRONG answer      {pa(z['ans_shown_wrong'], n_ans)}   <- harmful")
    print("ZeroMem, should-refuse questions (pipeline shows no answer):")
    print(f"  overall                            {pa(z['ref_refused'], n_ref)}")
    for kind, c in sorted(zk.items()):
        print(f"    {kind:12s}                     {pa(c['refused'], c['n'])}  (n={c['n']})")
    print("Reference, cross-encoder reranker alone (top sentence, same threshold):")
    print(f"  shows the correct answer           {pa(ref['ans_shown_correct'], n_ans)}")
    print(f"  shows a WRONG answer               {pa(ref['ans_shown_wrong'], n_ans)}")
    print(f"  refuses when it should             {pa(ref['ref_refused'], n_ref)}")


if __name__ == "__main__":
    main()
