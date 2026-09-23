"""Talk to the chat-fine-tuned mini model.

This is a toy: it has no world knowledge (its only reading so far is
children's stories plus ~2k small-talk conversations), so factual answers
will be made up. Its value is showing the fine-tune worked and the mechanics
are sound.

Usage:
    python -m zeromem.train.chat                # interactive
    python -m zeromem.train.chat --once "hi"    # single reply, then exit
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from zeromem.tokenizer.special_tokens import EOS
from zeromem.train.generate import load_model
from zeromem.train.train_stage1 import get_device


@torch.no_grad()
def generate_reply(model, tokenizer, history_ids: list[int], device, max_new: int = 100,
                   temperature: float = 0.7, top_k: int = 40) -> list[int]:
    """Continue after 'AI:' until the model emits a newline (end of its turn) or <EOS>."""
    ids = list(history_ids)
    new: list[int] = []
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.max_seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[0, -1] / max(temperature, 1e-6)
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(next_id)
        new.append(next_id)
        if next_id == EOS or "\n" in tokenizer.decode([next_id]):
            break
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints/chat/best.pt")
    parser.add_argument("--tokenizer", default="zeromem/tokenizer/trained/tokenizer.json")
    parser.add_argument("--once", default=None, help="Send one message and exit")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    history: list[int] = []

    def turn(user_text: str) -> str:
        nonlocal history
        history += tokenizer.encode(f"User: {user_text.strip()}\n").ids
        history += tokenizer.encode("AI:").ids
        new = generate_reply(model, tokenizer, history, device, temperature=args.temperature)
        history += new
        return tokenizer.decode(new).strip()

    if args.once is not None:
        print(f"You: {args.once}\nAI: {turn(args.once)}")
        return

    print("Chat with the mini model (Ctrl+C or empty line to quit).")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            break
        print(f"AI: {turn(msg)}")


if __name__ == "__main__":
    main()
