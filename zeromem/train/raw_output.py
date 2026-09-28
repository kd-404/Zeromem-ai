"""Raw output viewer: type ANY text, no chunks, and see what the model writes back as plain text.

Output looks like the chat script:

    You: what is red
    AI: <the model's raw reply>

Nothing is wrapped or formatted for you (unless you pass --as-question). The text you type goes
into the model as-is. The reply is shown as text; the model's own control markers (<KNOW>,
<REFUSE>, <DONE>) appear inside it because they are part of what it wrote. Add --hide-special to
remove them. Token ids and probabilities are hidden unless you ask (--debug, --show-tokens).

By default it reads checkpoints/arrange_1chunk_rp/best.pt (the finalized fine-tune). Point --ckpt
at another checkpoint to compare, e.g. checkpoints/stage1/latest.pt (stories only) or
checkpoints/readpractice/latest.pt (after Wikipedia reading practice).

Usage:
    python -m zeromem.train.raw_output "Once upon a time"
    python -m zeromem.train.raw_output --as-question "Who designed the Eiffel Tower?"
    python -m zeromem.train.raw_output "Question: hi\\nAnswer:"          # \\n means a newline
    python -m zeromem.train.raw_output --temperature 0.8 "The capital of France is"
    python -m zeromem.train.raw_output                                    # interactive
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from zeromem.tokenizer.special_tokens import DONE, EOS, SPECIAL_TOKENS
from zeromem.train.generate import load_model

DEFAULT_CKPT = "checkpoints/arrange_1chunk_rp/best.pt"
DEFAULT_TOKENIZER = "zeromem/tokenizer/trained/tokenizer.json"


def show(tok: Tokenizer, i: int) -> str:
    return SPECIAL_TOKENS[i] if i < len(SPECIAL_TOKENS) else repr(tok.decode([i]))


@torch.no_grad()
def generate_text(model, tok: Tokenizer, prompt: str, device: torch.device, max_new: int,
                  temperature: float, top_k: int, debug: bool, show_tokens: bool, hide_special: bool) -> str:
    ids = tok.encode(prompt).ids
    max_ctx = model.cfg.max_seq_len
    out: list[int] = []
    if debug:
        print(f"[debug] prompt is {len(ids)} tokens")

    for step in range(max_new):
        x = torch.tensor([(ids + out)[-max_ctx:]], device=device)
        logits, _ = model(x)                       # plain causal path: no chunk structure at all
        logits = logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)

        if debug and step == 0:
            top = torch.topk(probs, 5)
            print("[debug] first-token top 5: " + "  |  ".join(
                f"{show(tok, int(i))} {float(p):.2f}" for p, i in zip(top.values, top.indices)))

        if temperature <= 0:
            nxt = int(torch.argmax(logits))        # greedy: same input always gives the same output
        else:
            scaled = logits / temperature
            if top_k > 0:
                kth = torch.topk(scaled, min(top_k, scaled.numel())).values[-1]
                scaled[scaled < kth] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(scaled, dim=-1), 1))

        if show_tokens:
            alts = torch.topk(probs, 3)
            print(f"[token {step:>3}] {show(tok, nxt):<14} p={float(probs[nxt]):.3f}   top3: " + ", ".join(
                f"{show(tok, int(i))} {float(p):.2f}" for p, i in zip(alts.values, alts.indices)))
        out.append(nxt)
        if nxt in (DONE, EOS):
            break

    if debug:
        print("[debug] token ids:", out)
    return tok.decode(out, skip_special_tokens=hide_special).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="*", help="text to feed the model as-is (omit for interactive mode)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--as-question", action="store_true",
                    help='wrap the text as "Question: <text>\\nAnswer:" (still no chunks)')
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (repeatable); try 0.8 for variety")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hide-special", action="store_true", help="remove <KNOW>/<REFUSE>/<DONE> markers from the text")
    ap.add_argument("--debug", action="store_true", help="also show token ids and the first-token top 5")
    ap.add_argument("--show-tokens", action="store_true", help="also print every generated token with its probability")
    ap.add_argument("--device", default=None, help="mps | cpu (default: mps if available)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    tok = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.ckpt, device)

    def go(typed: str) -> None:
        text = typed.replace("\\n", "\n")           # let you type \n for a newline in one line
        if args.as_question:
            text = f"Question: {text}\nAnswer:"
        reply = generate_text(model, tok, text, device, args.max_new, args.temperature, args.top_k,
                              args.debug, args.show_tokens, args.hide_special)
        print(f"You: {typed}\nAI: {reply}")

    if args.prompt:
        go(" ".join(args.prompt))
        return
    print("Raw output mode. Type text and press Enter (empty line to quit).")
    while True:
        try:
            line = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            break
        go(line)


if __name__ == "__main__":
    main()
