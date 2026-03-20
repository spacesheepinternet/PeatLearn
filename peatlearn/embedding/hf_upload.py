#!/usr/bin/env python3
"""
Upload local embeddings to a Hugging Face dataset repository.

Usage:
  python embedding/upload_to_hf.py \
    --repo abanwild/peatlearn-embeddings \
    --path embedding/vectors \
    --private false

Environment variables (preferred):
  HF_TOKEN=hf_...                # Hugging Face token with write access
  HF_DATASET_REPO=owner/name     # e.g., abanwild/peatlearn-embeddings

Notes:
- Only files matching patterns (npy/pkl/json/zip) are uploaded.
- Existing files on the repo will be updated on subsequent runs.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder

ALLOWED_PATTERNS = ["*.npy", "*.pkl", "*.json", "*.zip", "*.pkl alias"]
IGNORED_PATTERNS = [".DS_Store", "*.tmp", "*.log"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload embeddings to Hugging Face Datasets repo")
    p.add_argument("--repo", default=None, help="Target repo id (owner/name). Overrides HF_DATASET_REPO")
    p.add_argument("--path", default="embedding/vectors", help="Local folder with embeddings")
    p.add_argument("--private", default=None, choices=["true", "false"], help="Create repo as private if created")
    p.add_argument("--branch", default="main", help="Target branch on HF")
    return p.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()
    repo_id = args.repo or os.getenv("HF_DATASET_REPO")
    if not repo_id:
        raise SystemExit("HF_DATASET_REPO env or --repo is required")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in environment")

    folder_path = Path(args.path)
    if not folder_path.exists():
        raise SystemExit(f"Folder not found: {folder_path}")

    api = HfApi(token=token)

    # Ensure repo exists
    private = (args.private or os.getenv("HF_DATASET_PRIVATE", "false")).lower() == "true"
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    # Upload folder
    print(f"Uploading {folder_path} to hf://{repo_id} (repo_type=dataset, branch={args.branch}) ...")
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder_path),
        token=token,
        revision=args.branch,
        commit_message="Upload/update embeddings",
        allow_patterns=ALLOWED_PATTERNS,
        ignore_patterns=IGNORED_PATTERNS,
    )

    print("Done. View at: https://huggingface.co/datasets/" + repo_id)


if __name__ == "__main__":
    main()
