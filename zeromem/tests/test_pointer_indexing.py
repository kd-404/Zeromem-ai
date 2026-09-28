"""No torch or tokenizers needed: run `python zeromem/tests/test_pointer_indexing.py`.

Check the pointer index math with a NumPy stand-in for torch and a fake model that puts the
right answer ONLY at the exact positions the code should read (s-1 / s+1 in training,
n-1 / n+1 at inference) and decoys one position off. Any off-by-one shows up as a wrong pick."""
import json, os, sys, types
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # folder that contains zeromem/
sys.path.insert(0, ROOT)


# ---------------- numpy-backed torch subset ----------------
class TT(np.ndarray):
    def float(self): return self.astype(np.float32).view(TT)
    def cpu(self): return self
    def to(self, *a, **k): return self
    def detach(self): return self
    def clone(self): return self.copy().view(TT)
    def clamp(self, min=None, max=None): return np.clip(self, min, max).view(TT)
    def size(self, d=None): return self.shape if d is None else self.shape[d]
    def item(self): return self.tolist()


def _t(a, dtype=None): return np.asarray(a, dtype=dtype).view(TT)


torch = types.ModuleType("torch")
torch.long = np.int64
torch.Tensor = TT
torch.tensor = lambda a, dtype=None, device=None: _t(a, dtype)
torch.full = lambda shape, v, dtype=None: _t(np.full(shape, v, dtype=dtype))
torch.zeros = lambda shape, dtype=None: _t(np.zeros(shape, dtype=dtype or np.float32))
torch.ones = lambda shape, dtype=None: _t(np.ones(shape, dtype=dtype or np.float32))
torch.arange = lambda n, device=None: _t(np.arange(n))
torch.argmax = lambda a: _t(np.argmax(a))
torch.softmax = lambda a, dim=-1: _t(np.exp(a - a.max(axis=dim, keepdims=True)) / np.exp(a - a.max(axis=dim, keepdims=True)).sum(axis=dim, keepdims=True))
torch.no_grad = lambda: (lambda f: f)
torch.device = lambda name: types.SimpleNamespace(type=name)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
nn = types.ModuleType("torch.nn"); F = types.ModuleType("torch.nn.functional"); nn.functional = F; torch.nn = nn
sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": F})

# ---------------- fake char-level tokenizer (special tokens recognized) ----------------
from zeromem.tokenizer.special_tokens import SPECIAL_TOKENS, KNOW, REFUSE, DONE, PAD
import re
_SPEC = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")
V = 400


class Tok:
    @staticmethod
    def from_file(p): return Tok()
    def token_to_id(self, t): return SPECIAL_TOKENS.index(t) if t in SPECIAL_TOKENS else (ord(t) if len(t) == 1 and ord(t) < V else None)
    def encode(self, text):
        ids = []
        for part in _SPEC.split(text):
            if part in SPECIAL_TOKENS: ids.append(SPECIAL_TOKENS.index(part))
            elif part: ids += [ord(ch) if ord(ch) < V else ord("?") for ch in part]
        return types.SimpleNamespace(ids=ids)


tokmod = types.ModuleType("tokenizers"); tokmod.Tokenizer = Tok; sys.modules["tokenizers"] = tokmod
for name in ("zeromem.train.generate", "zeromem.train.train_stage1"):
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["zeromem.train.train_stage1"].cosine_lr = lambda *a: 0.0
sys.modules["zeromem.train.train_stage1"].get_device = lambda: torch.device("cpu")

from zeromem.data.pointer_format import MARKERS
tok = Tok()
LB = tok.token_to_id("[")
MIDS = [tok.token_to_id(m) for m in MARKERS]


class OracleModel:
    """Puts the verdict at `vpos(b)` and the letter at `lpos(b)`; decoys one position either side."""
    def __init__(self, plan): self.plan = plan  # list per row: (vpos, know:bool, lpos, letter_index)
    def __call__(self, x, chunk_ids=None):
        B, L = x.shape
        lg = np.zeros((B, L, V), dtype=np.float32)
        for b, (vpos, know, lpos, li) in enumerate(self.plan[:B]):
            for off in (-1, 1):  # decoys: the OPPOSITE verdict / a wrong letter, one position off
                if 0 <= vpos + off < L: lg[b, vpos + off, REFUSE if know else KNOW] = 9
                if 0 <= lpos + off < L: lg[b, lpos + off, MIDS[(li + 1) % 26]] = 9
            lg[b, vpos, KNOW if know else REFUSE] = 5
            lg[b, lpos, MIDS[li]] = 5
        self.plan = self.plan[B:]
        return lg.view(TT), None


# ================= 1. training-side: decisions() =================
sys.modules["zeromem.train.generate"].load_model = lambda *a, **k: None
import zeromem.train.finetune_pointer as FP
rows = [json.loads(l) for l in open(os.path.join(ROOT, "zeromem/data/processed/pointer1_val.jsonl"))][:64]
ex, meta = [], []
from zeromem.data.arrange_format import encode_example
for r in rows:
    ids, labels, cids = encode_example(tok, r["chunks"], r["question"], r["target"])
    ex.append((ids, labels, cids, r["answerable"]))
    meta.append({"n": len(r["sentences"]), "gold": r["gold_index"], "q": r["question"], "sentences": r["sentences"], "kind": r["kind"]})
T = max(len(e[0]) for e in ex) + 5
# the target really is 5 tokens: KNOW [ letter ] DONE
for e, r in zip(ex, rows):
    t = [l for l in e[1] if l != PAD]
    assert (t == [KNOW, LB, MIDS[r["gold_index"]], ord("]"), DONE]) if r["answerable"] else (t == [REFUSE, DONE]), t
idxs = list(range(len(ex)))
x, y, c, w = FP.make_batch(ex, idxs, T, 5.0)
plan = []
for i in idxs:
    s = next(k for k, l in enumerate(ex[i][1]) if l != PAD)
    plan.append((s - 1, ex[i][3], s + 1, meta[i]["gold"] or 0))
    # loss weights land on the verdict and the letter targets
    assert float(w[i, s - 1]) == 5.0 and y[i, s - 1] == ex[i][1][s]
    if ex[i][3]:
        assert float(w[i, s + 1]) == 5.0 and y[i, s + 1] == MIDS[meta[i]["gold"]], (y[i, s + 1], MIDS[meta[i]['gold']])
logits, _ = OracleModel(plan)(x)
got = FP.decisions(logits, ex, meta, idxs, MIDS)
for (know, pick), i in zip(got, idxs):
    assert know == ex[i][3], (i, know)
    if ex[i][3]: assert pick == meta[i]["gold"], (i, pick, meta[i]["gold"])
print(f"training decisions(): {len(idxs)} rows, verdict + letter read at the right positions, loss weights on the right targets")

# ================= 2. inference-side: PointerReader.read_sentences() =================
import zeromem.reader as R
from zeromem.data.arrange_format import encode_prompt
from zeromem.data.pointer_format import render_chunk
reader = R.PointerReader.__new__(R.PointerReader)
reader.device, reader.tok, reader.min_know = torch.device("cpu"), tok, 0.5
reader.KNOW, reader.REFUSE, reader.PAD, reader.LB, reader.mids = KNOW, REFUSE, PAD, LB, MIDS
q = "Who designed the Eiffel Tower?"
lists = [["The tower is in Paris.", "It is made of iron.", "Gustave Eiffel's company designed it.", "It opened in 1889."],
         ["Rain is water falling from clouds.", "It is common in monsoon season."],  # REFUSE row
         ["One.", "Two sentences only here."],                                         # letter C would be out of range
         []]                                                                           # empty chunk
ns = [len(encode_prompt(tok, [render_chunk(s)], q)[0]) if s else 0 for s in lists]
plan = [(ns[0] - 1, True, ns[0] + 1, 2), (ns[1] - 1, False, ns[1] + 1, 0), (ns[2] - 1, True, ns[2] + 1, 2)]
reader.model = OracleModel(plan)
out = reader.read_sentences(q, lists)
assert out[0].verdict == "KNOW" and out[0].pick == "C" and out[0].quote == lists[0][2] and out[0].verified, out[0]
assert out[1].verdict == "REFUSE" and out[1].quote is None, out[1]
assert out[2].verdict == "KNOW" and out[2].pick in ("A", "B") and out[2].quote in lists[2], out[2]
assert out[3].verdict == "REFUSE", out[3]
# a 70+ word "sentence" (a flattened list) can never be the answer, even if the model prefers it
longs = [["Short opener here.", "Second short line.", " ".join(["name"] * 90) + "."]]
nl = len(encode_prompt(tok, [render_chunk(longs[0])], q)[0])
reader.model = OracleModel([(nl - 1, True, nl + 1, 2)])
lo = reader.read_sentences(q, longs)[0]
assert lo.verdict == "KNOW" and lo.pick in ("A", "B"), lo
print(f"reader: picks [C] -> {out[0].quote!r} (p_know {out[0].prob:.2f}) | refuses row 2 | "
      f"out-of-range letter masked -> [{out[2].pick}] | empty chunk -> REFUSE")
print("\nALL POINTER INDEX TESTS PASSED")

# ================= 3. preview the EVAL block the terminal will show =================
def ce(logits2d, y, ignore_index=0, reduction="sum"):
    l = np.asarray(logits2d, dtype=np.float64); y = np.asarray(y)
    m = y != ignore_index
    lse = np.log(np.exp(l[m] - l[m].max(1, keepdims=True)).sum(1)) + l[m].max(1)
    return _t(np.array((lse - l[m][np.arange(m.sum()), y[m]]).sum()))
F.cross_entropy = ce
rng = np.random.default_rng(0)
class HalfTrained:  # right ~70% of the time
    def eval(self): pass
    def train(self): pass
    def __call__(self, x, chunk_ids=None):
        B, L = x.shape
        lg = np.zeros((B, L, V), dtype=np.float32)
        for b in range(B):
            row = [int(t) for t in np.asarray(x[b])]
            # find where the answer starts: first position after 'Answer:' text
            s = len(row) - 1 - row[::-1].index(ord(":"))  # last ':' = end of 'Answer:'
            nxt = row[s + 1] if s + 1 < L else PAD
            know = nxt == KNOW if rng.random() < .8 else nxt != KNOW
            lg[b, s, KNOW if know else REFUSE] = 3
            if s + 3 < L and row[s + 3] in MIDS:
                gold = MIDS.index(row[s + 3]); pick = gold if rng.random() < .7 else (gold + 1) % 3
                lg[b, s + 2, MIDS[pick]] = 3
        return lg.view(TT), None
m = FP.evaluate(HalfTrained(), ex, meta, "cpu", 8, T, 64, MIDS)
FP.print_eval(500, m, 9.9, None)
import random
FP.show_examples(HalfTrained(), ex, meta, "cpu", T, MIDS, 3, random.Random(1))
