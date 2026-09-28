"""Build and publish ZeroMem as a Hugging Face Space (Docker, free CPU tier).

One-time setup on your Mac (inside the project's .venv):
    pip install "huggingface_hub>=1.5,<2.0"
    hf auth login            # paste a token with WRITE access: huggingface.co/settings/tokens

Then, from the project root (the folder that contains zeromem/ and checkpoints/):
    python deploy/hf_space/deploy.py --space YOUR_HF_USERNAME/zeromem

Options:
    --private            make the Space private
    --min-know 0.6       how sure ZeroMem must be before it answers (default 0.5)
    --repo-url URL       add a 'Source code' link (only once your GitHub repo is public)
    --build-only         only assemble build/hf-space/ (to inspect), don't upload

What gets uploaded: the zeromem/ Python code (no datasets, tests or caches), the tokenizer,
checkpoints/pointer/best.pt (~145 MB, sent via Git LFS automatically), the Dockerfile,
requirements and the Space's README. Hugging Face then builds the image (~5-10 min) and
serves it at https://huggingface.co/spaces/YOUR_HF_USERNAME/zeromem.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / "build" / "hf-space"
SKIP_DIRS = {"__pycache__", "tests", "processed", "raw", "trained", "api"}
WEIGHTS = ROOT / "checkpoints" / "pointer" / "best.pt"
TOKENIZER = ROOT / "zeromem" / "tokenizer" / "trained" / "tokenizer.json"


def build(min_know: float, repo_url: str) -> Path:
    for need in (WEIGHTS, TOKENIZER):
        if not need.exists():
            sys.exit(f"missing {need.relative_to(ROOT)}: train the pointer model first (see README).")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    n = 0
    for src in (ROOT / "zeromem").rglob("*.py"):
        rel = src.relative_to(ROOT)
        if SKIP_DIRS & set(rel.parts):
            continue
        dst = BUILD / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    (BUILD / TOKENIZER.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TOKENIZER, BUILD / TOKENIZER.relative_to(ROOT))
    (BUILD / WEIGHTS.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEIGHTS, BUILD / WEIGHTS.relative_to(ROOT))
    shutil.copy2(HERE / "requirements.txt", BUILD / "requirements.txt")
    (BUILD / "Dockerfile").write_text((HERE / "Dockerfile").read_text()
                                      .replace("__MIN_KNOW__", str(min_know))
                                      .replace("__REPO_URL__", repo_url or '""'))
    (BUILD / "README.md").write_text((HERE / "README.md").read_text()
                                     .replace("__REPO_LINE__", f"Source code: {repo_url}" if repo_url else ""))
    (BUILD / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n*.json filter=lfs diff=lfs merge=lfs -text\n")
    size = sum(f.stat().st_size for f in BUILD.rglob("*") if f.is_file()) / 1e6
    print(f"built {BUILD.relative_to(ROOT)}: {n} code files + tokenizer + weights ({size:.0f} MB total)")
    return BUILD


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="YOUR_HF_USERNAME/space-name, e.g. balajikd/zeromem")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--min-know", type=float, default=0.5)
    ap.add_argument("--repo-url", default="",
                    help="shown as 'Source code' on the page (only if the GitHub repo is PUBLIC), "
                         "e.g. https://github.com/kd-404/Zeromem-ai")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    folder = build(args.min_know, args.repo_url)
    if args.build_only:
        return
    if not args.space or "/" not in args.space:
        sys.exit("pass --space YOUR_HF_USERNAME/zeromem")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install "huggingface_hub>=1.5,<2.0"   (then: hf auth login)")
    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:  # noqa: BLE001
        sys.exit("not logged in to Hugging Face: run  hf auth login  with a WRITE token")
    print(f"logged in as {who}; creating/updating Space {args.space} ...")
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy ZeroMem (pointer reader)", delete_patterns=["*"])
    print(f"\nUploaded. Hugging Face is building it now (about 5-10 minutes).\n"
          f"  App:  https://huggingface.co/spaces/{args.space}\n"
          f"  Logs: https://huggingface.co/spaces/{args.space}?logs=build")


if __name__ == "__main__":
    main()
