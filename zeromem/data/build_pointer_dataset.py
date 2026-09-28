"""Convert the one-chunk arrange dataset into the pointer format (data/pointer_format.py).

No new scraping and no new labels: every arrange example already stores its gold sentence,
so the pointer target is just "which letter is that sentence". Refusal examples stay refusals.

    python -m zeromem.data.build_pointer_dataset
    -> zeromem/data/processed/pointer1_train.jsonl, pointer1_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from zeromem.data.pointer_format import MAX_SENTENCES, find_gold, pointer_target, render_chunk, split_sentences

_KNOW = re.compile(r"^<KNOW>\[\d+\] (.*)<DONE>$", re.S)


def convert(src: str, dst: str) -> Counter:
    stats: Counter = Counter()
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            stats["read"] += 1
            chunk = r["chunks"][0]
            sents = split_sentences(chunk)
            if not sents:
                stats["skip: no sentences"] += 1
                continue
            if r["answerable"]:
                m = _KNOW.match(r["target"])
                gi = find_gold(sents, m.group(1)) if m else None
                if gi is None:
                    stats["skip: gold sentence not found after splitting"] += 1
                    continue
                stats["answerable"] += 1
                stats[f"gold position {min(gi, 9)}{'+' if gi >= 9 else ''}"] += 1
            else:
                gi = None
                stats[f"refuse ({r['kind']})"] += 1
            stats["sentences total"] += len(sents)
            out.write(json.dumps({
                "id": r["id"], "kind": r["kind"], "question": r["question"],
                "chunks": [render_chunk(sents)], "sentences": sents,
                "target": pointer_target(gi), "answerable": r["answerable"], "gold_index": gi,
            }, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default="zeromem/data/processed")
    ap.add_argument("--src-prefix", default="arrange1")
    ap.add_argument("--dst-prefix", default="pointer1")
    args = ap.parse_args()
    for split in ("train", "val"):
        src = f"{args.src_dir}/{args.src_prefix}_{split}.jsonl"
        dst = f"{args.src_dir}/{args.dst_prefix}_{split}.jsonl"
        s = convert(src, dst)
        print(f"\n{split}: {src} -> {dst}")
        print(f"  read {s['read']:,} | written {s['written']:,} | answerable {s['answerable']:,} | "
              f"avg sentences/chunk {s['sentences total'] / max(s['written'], 1):.1f} (max {MAX_SENTENCES})")
        for k in sorted(k for k in s if k.startswith(("skip", "refuse"))):
            print(f"  {k:48s} {s[k]:,}")
        pos = {k: s[k] for k in sorted(k for k in s if k.startswith("gold position"))}
        print("  where the answer sits:", ", ".join(f"{k[14:]}: {v}" for k, v in pos.items()))


if __name__ == "__main__":
    main()
