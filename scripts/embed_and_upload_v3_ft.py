#!/usr/bin/env python3
"""
Embed & Upload V3 (fine-tuned) — local peat-embeddinggemma-ft → Pinecone ray-peat-corpus-v3-ft.

Reads the v3 JSONL, embeds with fine-tuned EmbeddingGemma (768-dim, local GPU),
creates the Pinecone index if needed, and upserts all vectors.

Features:
  - Checkpoint / resume: saves progress every 500 vectors, safe to Ctrl-C and restart
  - Dry run: validate JSONL without embedding
  - GPU batch size configurable (default 128)

Usage:
    python scripts/embed_and_upload_v3_ft.py
    python scripts/embed_and_upload_v3_ft.py --jsonl data/corpus_v3/corpus_v3_<timestamp>.jsonl
    python scripts/embed_and_upload_v3_ft.py --dry-run
    python scripts/embed_and_upload_v3_ft.py --resume
    python scripts/embed_and_upload_v3_ft.py --batch-size 64
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

CORPUS_V3_DIR = PROJECT_ROOT / "data" / "corpus_v3"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "embeddings" / "peat-embeddinggemma-ft"

PINECONE_INDEX = "ray-peat-corpus-v3-ft"
PINECONE_DIMENSION = 768
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_BATCH = 100


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> set[str]:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_path: Path, uploaded_ids: set[str]):
    with open(checkpoint_path, "w") as f:
        json.dump(list(uploaded_ids), f)


# ── Pinecone upsert ───────────────────────────────────────────────────────────

def upsert_batch(index, vectors: list[tuple], retries: int = 5):
    delay = 1.0
    for attempt in range(retries):
        try:
            index.upsert(vectors=vectors)
            return
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Pinecone upsert error (attempt {attempt+1}/{retries}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                raise


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, help="Path to corpus_v3 JSONL (default: latest in data/corpus_v3/)")
    parser.add_argument("--dry-run", action="store_true", help="Validate JSONL without embedding or uploading")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already in Pinecone (via checkpoint)")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU embedding batch size")
    args = parser.parse_args()

    # Resolve JSONL path
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.is_absolute():
            jsonl_path = PROJECT_ROOT / jsonl_path
    else:
        jsonl_files = sorted(CORPUS_V3_DIR.glob("corpus_v3_*.jsonl"))
        if not jsonl_files:
            print("ERROR: No corpus_v3_*.jsonl files found. Run build_corpus_v3.py first.")
            sys.exit(1)
        jsonl_path = jsonl_files[-1]

    print(f"\n{'='*70}")
    print(f"  EMBED & UPLOAD V3 — fine-tuned EmbeddingGemma")
    print(f"{'='*70}")
    print(f"  Input:       {jsonl_path.relative_to(PROJECT_ROOT)}")
    print(f"  Model:       {MODEL_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Index:       {PINECONE_INDEX}")
    print(f"  Dimensions:  {PINECONE_DIMENSION}")
    print(f"  Batch size:  {args.batch_size}\n")

    # Load records
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records):,} records from JSONL")

    if args.dry_run:
        print("\n  DRY RUN — no embedding or upload.")
        missing_embed_text = sum(1 for r in records if not r.get("embed_text"))
        print(f"  Records missing embed_text: {missing_embed_text}")
        return

    # Load fine-tuned model
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading model from {MODEL_DIR.name}...")
    model = SentenceTransformer(str(MODEL_DIR), trust_remote_code=True, device=device)
    actual_dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded. Embedding dim: {actual_dim}")
    if actual_dim != PINECONE_DIMENSION:
        print(f"  ERROR: model dim {actual_dim} != expected {PINECONE_DIMENSION}")
        sys.exit(1)

    # Pinecone setup
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("  ERROR: PINECONE_API_KEY not set")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"\n  Creating index '{PINECONE_INDEX}' ({PINECONE_DIMENSION}-dim, {PINECONE_METRIC})...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        time.sleep(5)
    else:
        print(f"\n  Index '{PINECONE_INDEX}' already exists.")

    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()
    print(f"  Current vector count: {stats.total_vector_count:,}")

    # Checkpoint / resume
    checkpoint_path = CHECKPOINT_DIR / f"v3_ft_upload_checkpoint.json"
    uploaded_ids: set[str] = set()
    if args.resume:
        uploaded_ids = load_checkpoint(checkpoint_path)
        print(f"  Resuming from checkpoint: {len(uploaded_ids):,} already uploaded")

    pending = [r for r in records if r["id"] not in uploaded_ids]
    print(f"  Pending: {len(pending):,} records to embed and upload\n")

    # Embed + upload
    total = len(pending)
    texts = [r.get("embed_text") or r.get("text", "") for r in pending]

    t0 = time.time()
    all_embeddings = []

    print(f"  Embedding {total:,} texts in batches of {args.batch_size}...")
    for i in range(0, total, args.batch_size):
        batch_texts = texts[i:i + args.batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_embeddings.extend(batch_embeddings.tolist())
        done = min(i + args.batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done/total*100:5.1f}%] {done:6,}/{total:,} embedded  {rate:.0f} rec/s  ETA {eta/60:.1f}min", end="\r")

    print(f"\n  Embedding complete in {(time.time()-t0)/60:.1f}min")

    # Upload to Pinecone in batches
    print(f"\n  Uploading {total:,} vectors to Pinecone...")
    t1 = time.time()
    upsert_count = 0

    for i in range(0, total, PINECONE_BATCH):
        batch_records = pending[i:i + PINECONE_BATCH]
        batch_embeddings = all_embeddings[i:i + PINECONE_BATCH]

        vectors = []
        for rec, emb in zip(batch_records, batch_embeddings):
            metadata = {
                "context": (rec.get("context") or "")[:1000],
                "text": (rec.get("text") or "")[:1000],
                "source_file": rec.get("source_file", ""),
                "source_type": rec.get("source_type", ""),
                "title": rec.get("title", ""),
                "primary_topic": rec.get("primary_topic", ""),
                "topics": rec.get("topics", []),
                "tokens": rec.get("tokens", 0),
                "chunk_index": rec.get("chunk_index", 0),
            }
            vectors.append((rec["id"], emb, metadata))

        upsert_batch(index, vectors)
        upsert_count += len(vectors)

        for rec in batch_records:
            uploaded_ids.add(rec["id"])

        # Checkpoint every 500
        if upsert_count % 500 == 0:
            save_checkpoint(checkpoint_path, uploaded_ids)

        done = min(i + PINECONE_BATCH, total)
        elapsed = time.time() - t1
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done/total*100:5.1f}%] {done:6,}/{total:,} uploaded  {rate:.0f} rec/s  ETA {eta/60:.1f}min", end="\r")

    save_checkpoint(checkpoint_path, uploaded_ids)
    total_time = (time.time() - t0) / 60
    final_stats = index.describe_index_stats()

    print(f"\n\n{'='*70}")
    print(f"  DONE")
    print(f"  Vectors uploaded: {upsert_count:,}")
    print(f"  Total in index:   {final_stats.total_vector_count:,}")
    print(f"  Total time:       {total_time:.1f}min")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
