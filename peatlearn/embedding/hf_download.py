#!/usr/bin/env python3
"""
Download embeddings from a Hugging Face dataset repository into local folder.

Usage:
  python embedding/download_from_hf.py --repo abanwild/peatlearn-embeddings --dest embedding/vectors

Environment variables (preferred):
  HF_TOKEN            # Optional for public repos; required for private
  HF_DATASET_REPO     # e.g., abanwild/peatlearn-embeddings
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ALLOWED_EXTS = {".npy", ".pkl", ".json", ".zip"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download embeddings from HF Datasets repo")
    p.add_argument("--repo", default=None, help="Repo id (owner/name). Overrides HF_DATASET_REPO")
    p.add_argument("--dest", default="embedding/vectors", help="Destination folder")
    p.add_argument("--branch", default="main", help="Branch / revision")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    repo_id = args.repo or os.getenv("HF_DATASET_REPO")
    if not repo_id:
        raise SystemExit("HF_DATASET_REPO env or --repo is required")

    token = os.getenv("HF_TOKEN")  # optional for public

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading snapshot from hf://{repo_id} to {dest} ...")
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.branch,
        token=token,
    )

    # Copy allowed files to dest
    src = Path(local_path)
    count = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix in ALLOWED_EXTS:
            rel = p.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = p.read_bytes()
            target.write_bytes(data)
            count += 1

    print(f"Downloaded {count} files to {dest}")


if __name__ == "__main__":
    main()
