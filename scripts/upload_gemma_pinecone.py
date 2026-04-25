#!/usr/bin/env python3
"""
Upload fine-tuned EmbeddingGemma vectors to a new Pinecone index.

Creates 'ray-peat-corpus-v2' (768-dim, cosine, serverless) and uploads
all 26,431 vectors from the re-embedding step.

Usage:
    python scripts/upload_gemma_pinecone.py
    python scripts/upload_gemma_pinecone.py --batch-size 100 --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# --- Config ---
VECTORS_DIR = PROJECT_ROOT / "data" / "embeddings" / "vectors"
INDEX_NAME = "ray-peat-corpus-v2"
DIMENSION = 768
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def sanitize_vector_id(raw_id: str, index: int = 0) -> str:
    """Return a Pinecone-safe ASCII ID."""
    if not isinstance(raw_id, str):
        raw_id = str(raw_id)
    normalized = unicodedata.normalize("NFKD", raw_id)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = re.sub(r"\s+", "_", ascii_only)
    ascii_only = re.sub(r"[^A-Za-z0-9\._:/\+\-=]", "-", ascii_only)
    ascii_only = ascii_only.strip("-_.:/+=")
    if not ascii_only:
        suffix = hashlib.sha1(f"vector-{index}".encode()).hexdigest()[:8]
        ascii_only = f"vector_{suffix}"
    if len(ascii_only) > 512:
        digest = hashlib.sha1(raw_id.encode()).hexdigest()[:10]
        ascii_only = f"{ascii_only[:501]}-{digest}"
    return ascii_only


def find_latest_gemma_files() -> tuple[Path, Path]:
    """Find the latest gemma_ft embedding and metadata files."""
    npy_files = sorted(VECTORS_DIR.glob("embeddings_gemma_ft_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    json_files = sorted(VECTORS_DIR.glob("metadata_gemma_ft_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not npy_files or not json_files:
        raise FileNotFoundError("No gemma_ft embedding files found in vectors/")
    return npy_files[0], json_files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Prepare but don't upload")
    args = parser.parse_args()

    from pinecone import Pinecone, ServerlessSpec

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set")
        sys.exit(1)

    # Load vectors
    npy_path, meta_path = find_latest_gemma_files()
    print(f"Loading vectors: {npy_path.name}")
    print(f"Loading metadata: {meta_path.name}")

    vectors = np.load(str(npy_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Vectors: {vectors.shape}")
    assert vectors.shape[1] == DIMENSION, f"Expected {DIMENSION}-dim, got {vectors.shape[1]}"
    assert len(vectors) == len(metadata), "Vector/metadata count mismatch"

    # Prepare vectors
    print(f"\nPreparing {len(vectors)} vectors...")
    prepared = []
    seen_ids = set()
    for i, (vec, meta) in enumerate(zip(vectors, metadata)):
        raw_id = meta.get("id", f"vector_{i}")
        vec_id = sanitize_vector_id(raw_id, index=i)
        if vec_id in seen_ids:
            disambig = hashlib.sha1(str(raw_id).encode()).hexdigest()[:6]
            vec_id = f"{vec_id}-{disambig}"[:512]
        seen_ids.add(vec_id)

        pinecone_meta = {
            "context": meta.get("context", "")[:1000],
            "source_file": meta.get("source_file", ""),
            "tokens": meta.get("tokens", 0),
            "original_id": meta.get("id", ""),
        }
        response = meta.get("ray_peat_response", "")
        if len(response) <= 40000:
            pinecone_meta["ray_peat_response"] = response
        else:
            pinecone_meta["ray_peat_response"] = response[:39000] + "... [truncated]"

        prepared.append((vec_id, vec.tolist(), pinecone_meta))

    print(f"Prepared {len(prepared)} vectors (unique IDs: {len(seen_ids)})")

    if args.dry_run:
        print("\n[DRY RUN] Would create index and upload. Exiting.")
        return

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)

    # Create index if needed
    existing = pc.list_indexes().names()
    if INDEX_NAME in existing:
        print(f"\nIndex '{INDEX_NAME}' already exists — will upsert into it.")
    else:
        print(f"\nCreating index '{INDEX_NAME}' ({DIMENSION}-dim, {METRIC}, {CLOUD}/{REGION})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(10)

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"Index stats before upload: {stats.get('total_vector_count', 0)} vectors")

    # Upload in batches
    print(f"\nUploading {len(prepared)} vectors (batch_size={args.batch_size})...")
    success = 0
    failed = 0
    start = time.time()

    for i in range(0, len(prepared), args.batch_size):
        batch = prepared[i : i + args.batch_size]
        try:
            index.upsert(vectors=batch)
            success += len(batch)
        except Exception as e:
            print(f"  Batch {i // args.batch_size + 1} failed: {e}")
            failed += len(batch)

        if (i // args.batch_size + 1) % 50 == 0:
            pct = success / len(prepared) * 100
            print(f"  Progress: {success}/{len(prepared)} ({pct:.0f}%)")

    elapsed = time.time() - start
    print(f"\nUpload complete in {elapsed:.1f}s")
    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")

    # Verify
    time.sleep(5)
    stats = index.describe_index_stats()
    final_count = stats.get("total_vector_count", 0)
    print(f"\nFinal index count: {final_count}")
    if final_count >= len(prepared):
        print("Upload verified.")
    else:
        print(f"WARNING: Expected {len(prepared)}, got {final_count}. May need time to index.")


if __name__ == "__main__":
    main()
