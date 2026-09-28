"""Year 4 data: pointer data + HARD NEGATIVES (the gold sentence removed -> REFUSE).

Live tests showed the pointer model's main error: it picks the sentence NEXT to the answer
("12 days of sick leave" for "annual leave"). Here every answerable example also appears once
without its gold sentence, so only look-alike neighbours remain and the target is <REFUSE>.
No new downloads: built from pointer1_*.jsonl.

    python -m zeromem.data.build_pointer4_dataset   -> pointer4_train.jsonl, pointer4_val.jsonl
"""
import json
import random

from zeromem.data.pointer_format import REFUSE_TARGET, render_chunk

D = "zeromem/data/processed"


def build(split: str, rate: float, seed: int = 4) -> None:
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(f"{D}/pointer1_{split}.jsonl", encoding="utf-8")]
    out, neg = [], 0
    for r in rows:
        out.append(r)
        if r["answerable"] and len(r["sentences"]) >= 3 and rng.random() < rate:
            sents = [s for i, s in enumerate(r["sentences"]) if i != r["gold_index"]]
            out.append({**r, "id": r["id"] + "_nogold", "kind": "gold_removed", "chunks": [render_chunk(sents)],
                        "sentences": sents, "target": REFUSE_TARGET, "answerable": False, "gold_index": None})
            neg += 1
    rng.shuffle(out)
    with open(f"{D}/pointer4_{split}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print(f"{split}: {len(rows):,} original + {neg:,} hard negatives = {len(out):,}")


if __name__ == "__main__":
    build("train", rate=0.35)
    build("val", rate=0.35)
