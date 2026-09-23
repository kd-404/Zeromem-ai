"""Sample text from a trained ZeroMem checkpoint.

Stage-1 is a plain next-token completion model, not a chat model: whatever
prompt you give it is treated as the start of a document and continued in
the style of its training data (TinyStories). This script is for sanity-
checking that stage-1 actually learned fluent English.

Usage:
    python -m zeromem.train.generate --prompt "hi"
    python -m zeromem.train.generate --prompt "Once upon a time" --max-new-tokens 150 --temperature 0.7
"""

from __future__ import annotations

import argparse

import torch
from tokenizers import Tokenizer

from zeromem.model.my_transformer import ZeroMem
from zeromem.train.train_stage1 import get_device


def load_model(ckpt_path: str, device: torch.device) -> ZeroMem:
    # weights_only=False: checkpoint embeds a ZeroMemConfig object; it's our own local file.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ZeroMem(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (trained to step {ckpt['step']:,})")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints/stage1/latest.pt")
    parser.add_argument("--tokenizer", default="zeromem/tokenizer/trained/tokenizer.json")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # No <BOS> prepended: stage-1 training data only separated documents with <EOS>.
    prompt_ids = tokenizer.encode(args.prompt).ids
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        out = model.generate(x, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k)
        text = tokenizer.decode(out[0].tolist())
        print(f"\n--- sample {i + 1} ---\n{text}")


if __name__ == "__main__":
    main()
