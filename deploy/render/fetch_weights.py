"""Download ZeroMem's weights + tokenizer at image build time (they are not in git).

Reads WEIGHTS_REPO (default balaji9043/zeromem-pointer) and, for a private repo, HF_TOKEN.
Puts them where the app expects them:
    checkpoints/pointer/best.pt
    zeromem/tokenizer/trained/tokenizer.json
"""
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo = os.environ.get("WEIGHTS_REPO") or "balaji9043/zeromem-pointer"
token = os.environ.get("HF_TOKEN") or None
for name, dest in (("best.pt", "checkpoints/pointer/best.pt"),
                   ("tokenizer.json", "zeromem/tokenizer/trained/tokenizer.json")):
    path = hf_hub_download(repo, name, token=token)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, dest)
    print(f"{repo}/{name} -> {dest} ({Path(dest).stat().st_size / 1e6:.0f} MB)")
