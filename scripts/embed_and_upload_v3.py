#!/usr/bin/env python3
"""
Embed & Upload V3 — Generate Gemini embeddings and upload to Pinecone ray-peat-corpus-v3.

Reads the clean JSONL from build_corpus_v3.py, embeds each chunk with
gemini-embedding-001 (3072-dim), creates the Pinecone index if needed,
and upserts all vectors with rich topic metadata.

Features:
  - Checkpoint / resume: saves progress every 500 vectors, safe to Ctrl-C and restart
  - Dry run: validate JSONL without making API calls
  - Batch sizing: Gemini batch size 20, Pinecone upsert batch 100
  - Rate limiting: exponential backoff on 429s

Usage:
    python scripts/embed_and_upload_v3.py
    python scripts/embed_and_upload_v3.py --jsonl data/corpus_v3/corpus_v3_<timestamp>.jsonl
    python scripts/embed_and_upload_v3.py --dry-run
    python scripts/embed_and_upload_v3.py --resume          # skip already-uploaded IDs
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

CORPUS_V3_DIR = PROJECT_ROOT / "data" / "corpus_v3"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
PINECONE_INDEX = "ray-peat-corpus-v3"
PINECONE_DIMENSION = 3072
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

GEMINI_BATCH = 20       # texts per Gemini batch request
PINECONE_BATCH = 100    # vectors per Pinecone upsert


# ── Gemini Embedding ─────────────────────────────────────────────────────────

async def embed_batch(texts: list[str], session: aiohttp.ClientSession, api_key: str) -> list[list[float] | None]:
    """Embed a batch of texts with gemini-embedding-001. Returns list of vectors (or None on failure)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL}:batchEmbedContents"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "requests": [
            {
                "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": "RETRIEVAL_DOCUMENT",
            }
            for t in texts
        ]
    }

    retry_delay = 2.0
    for attempt in range(6):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = []
                    for emb in data.get("embeddings", []):
                        vals = emb.get("values", [])
                        results.append(vals if vals else None)
                    return results
                elif resp.status == 429:
                    wait = retry_delay * (2 ** attempt)
                    print(f"      Rate limit (429), waiting {wait:.0f}s (attempt {attempt+1}/6)...")
                    await asyncio.sleep(wait)
                    continue
                else:
                    err = await resp.text()
                    print(f"      Gemini error {resp.status}: {err[:200]}")
                    return [None] * len(texts)
        except Exception as e:
            wait = retry_delay * (2 ** attempt)
            print(f"      Gemini request failed: {e} — retry in {wait:.0f}s (attempt {attempt+1}/6)")
            await asyncio.sleep(wait)

    return [None] * len(texts)


# ── Pinecone ─────────────────────────────────────────────────────────────────

def create_pinecone_index(pc, index_name: str):
    """Create the Pinecone index if it doesn't exist."""
    from pinecone import ServerlessSpec

    existing = pc.list_indexes().names()
    if index_name in existing:
        print(f"  Index '{index_name}' already exists.")
        idx = pc.Index(index_name)
        stats = idx.describe_index_stats()
        print(f"  Current vector count: {stats.get('total_vector_count', 0):,}")
        return idx

    print(f"  Creating index '{index_name}' ({PINECONE_DIMENSION}-dim, {PINECONE_METRIC})...")
    pc.create_index(
        name=index_name,
        dimension=PINECONE_DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status.get("ready", False):
        print("  Waiting for index to be ready...")
        time.sleep(5)
    print(f"  Index '{index_name}' created and ready.")
    return pc.Index(index_name)


def upsert_batch(index, vectors: list[dict]):
    """Upsert a batch of vectors to Pinecone with retry."""
    for attempt in range(4):
        try:
            index.upsert(vectors=vectors)
            return True
        except Exception as e:
            if attempt < 3:
                wait = 2 ** attempt
                print(f"      Pinecone upsert failed: {e} — retry in {wait}s")
                time.sleep(wait)
            else:
                print(f"      Pinecone upsert failed after 4 attempts: {e}")
                return False
    return False


# ── Checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load set of already-uploaded IDs from checkpoint file."""
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path) as f:
        return set(json.load(f))


def save_checkpoint(checkpoint_path: Path, uploaded_ids: set[str]):
    """Save checkpoint of uploaded IDs."""
    with open(checkpoint_path, "w") as f:
        json.dump(list(uploaded_ids), f)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main_async(args):
    # Find latest JSONL if not specified
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.is_absolute():
            jsonl_path = PROJECT_ROOT / jsonl_path
    else:
        jsonl_files = sorted(CORPUS_V3_DIR.glob("corpus_v3_*.jsonl"))
        if not jsonl_files:
            print("ERROR: No corpus_v3_*.jsonl files found. Run build_corpus_v3.py first.")
            sys.exit(1)
        jsonl_path = jsonl_files[-1]  # latest

    print(f"\n{'='*70}")
    print(f"  EMBED & UPLOAD V3")
    print(f"{'='*70}")
    print(f"  Input:       {jsonl_path.relative_to(PROJECT_ROOT)}")
    print(f"  Index:       {PINECONE_INDEX}")
    print(f"  Dimensions:  {PINECONE_DIMENSION}")
    print(f"  Model:       {GEMINI_EMBEDDING_MODEL}\n")

    # Load records
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records):,} records from JSONL")

    if args.dry_run:
        print("\n  [DRY RUN] Validating records...")
        missing_fields = []
        for i, r in enumerate(records[:10]):
            for field in ["id", "text", "context", "source_file", "primary_topic", "embed_text"]:
                if field not in r:
                    missing_fields.append(f"Record {i} missing '{field}'")
        if missing_fields:
            print("  ERRORS:")
            for e in missing_fields:
                print(f"    {e}")
        else:
            print("  All records valid.")
        print(f"\n  Sample record:")
        r = records[0]
        print(f"    id:            {r['id']}")
        print(f"    source_type:   {r['source_type']}")
        print(f"    primary_topic: {r['primary_topic']}")
        print(f"    tokens:        {r['tokens']}")
        print(f"    embed_text[:100]: {r['embed_text'][:100]}...")
        total_tokens = sum(r["tokens"] for r in records)
        print(f"\n  Est. Gemini cost: ${total_tokens / 1_000_000 * 0.15:.3f} for {total_tokens:,} tokens")
        return

    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)
    if not pinecone_key:
        print("ERROR: PINECONE_API_KEY not set in .env")
        sys.exit(1)

    # Init Pinecone
    from pinecone import Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index = create_pinecone_index(pc, PINECONE_INDEX)

    # Load checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"embed_v3_{jsonl_path.stem}.json"
    uploaded_ids = load_checkpoint(checkpoint_path)
    if uploaded_ids:
        print(f"\n  Resuming from checkpoint: {len(uploaded_ids):,} already uploaded")

    # Filter to pending records
    pending = [r for r in records if r["id"] not in uploaded_ids]
    print(f"  Pending: {len(pending):,} records to embed and upload\n")

    if not pending:
        print("  All records already uploaded!")
        return

    # Embed + upload
    start_time = time.time()
    total_uploaded = len(uploaded_ids)
    total_failed = 0
    pinecone_buffer = []

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(pending), GEMINI_BATCH):
            batch = pending[batch_start:batch_start + GEMINI_BATCH]
            texts = [r["embed_text"] for r in batch]

            # Embed
            embeddings = await embed_batch(texts, session, gemini_key)

            # Build Pinecone vectors
            for record, embedding in zip(batch, embeddings):
                if embedding is None:
                    print(f"    SKIP {record['id']}: embedding failed")
                    total_failed += 1
                    continue

                # Pinecone metadata (keep values short — 40KB limit per vector)
                metadata = {
                    "context": record.get("context", "")[:500],
                    "ray_peat_response": record.get("text", "")[:2000],
                    "source_file": record.get("source_file", "")[:200],
                    "source_type": record.get("source_type", ""),
                    "title": record.get("title", "")[:100],
                    "primary_topic": record.get("primary_topic", ""),
                    "topics": record.get("topics", [])[:3],
                    "tokens": record.get("tokens", 0),
                    "chunk_index": record.get("chunk_index", 0),
                }

                pinecone_buffer.append({
                    "id": record["id"],
                    "values": embedding,
                    "metadata": metadata,
                })
                uploaded_ids.add(record["id"])

            # Upsert when buffer is full
            while len(pinecone_buffer) >= PINECONE_BATCH:
                batch_to_upload = pinecone_buffer[:PINECONE_BATCH]
                pinecone_buffer = pinecone_buffer[PINECONE_BATCH:]
                if upsert_batch(index, batch_to_upload):
                    total_uploaded += len(batch_to_upload)

            # Save checkpoint every 500 records
            if total_uploaded % 500 < GEMINI_BATCH:
                save_checkpoint(checkpoint_path, uploaded_ids)

            # Progress
            pct = (batch_start + len(batch)) / len(pending) * 100
            elapsed = time.time() - start_time
            rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0
            eta = (len(pending) - batch_start - len(batch)) / rate if rate > 0 else 0
            print(f"  [{pct:5.1f}%] {batch_start + len(batch):>6}/{len(pending):>6} embedded"
                  f"  {rate:.0f} rec/s  ETA {eta/60:.1f}min", end="\r", flush=True)

    # Flush remaining buffer
    if pinecone_buffer:
        if upsert_batch(index, pinecone_buffer):
            total_uploaded += len(pinecone_buffer)

    save_checkpoint(checkpoint_path, uploaded_ids)
    print()  # newline after progress

    # Final stats
    elapsed = time.time() - start_time
    stats = index.describe_index_stats()
    final_count = stats.get("total_vector_count", 0)

    print(f"\n{'='*70}")
    print(f"  UPLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"  Uploaded:    {total_uploaded:,} vectors")
    print(f"  Failed:      {total_failed:,}")
    print(f"  Index total: {final_count:,} vectors")
    print(f"  Time:        {elapsed/60:.1f} minutes")
    print(f"  Checkpoint:  {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print(f"\n  Index '{PINECONE_INDEX}' is ready to use.")
    print(f"\n  To activate: set PINECONE_INDEX_NAME=ray-peat-corpus-v3 in .env")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Embed corpus V3 and upload to Pinecone")
    parser.add_argument("--jsonl", default=None, help="Path to corpus JSONL (default: latest in data/corpus_v3/)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without making API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (default behavior)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
