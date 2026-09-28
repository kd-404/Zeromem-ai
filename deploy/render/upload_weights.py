"""Put ZeroMem's weights + tokenizer in a Hugging Face MODEL repo (free), for Render to download.

    python deploy/render/upload_weights.py                      # -> balaji9043/zeromem-pointer (public)
    python deploy/render/upload_weights.py --private            # then add HF_TOKEN in Render's settings
    python deploy/render/upload_weights.py --repo you/name

Uses your existing `hf auth login`.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {"best.pt": ROOT / "checkpoints/pointer/best.pt",
         "tokenizer.json": ROOT / "zeromem/tokenizer/trained/tokenizer.json"}
CARD = """---
tags: [zeromem, from-scratch, rag, extractive-qa]
---
# ZeroMem pointer reader

About 34M parameters, trained from scratch on a laptop. Reads a question plus a chunk whose
sentences are tagged [A] [B] [C] ... and answers `<KNOW>[C]` or `<REFUSE>`, so it never
retypes (or invents) text. `best.pt` is a PyTorch checkpoint for the ZeroMem code (not a
transformers model); `tokenizer.json` is its byte-level BPE tokenizer.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="balaji9043/zeromem-pointer")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    for name, p in FILES.items():
        if not p.exists():
            sys.exit(f"missing {p.relative_to(ROOT)}")
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"logged in as {api.whoami()['name']}; uploading to model repo {args.repo} ...")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for name, p in FILES.items():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=args.repo, repo_type="model")
        print(f"  {name} ({p.stat().st_size / 1e6:.0f} MB)")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=args.repo, repo_type="model")
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
