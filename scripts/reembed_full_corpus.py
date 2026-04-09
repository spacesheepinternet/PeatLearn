#!/usr/bin/env python3
"""
Full corpus re-embed: replace all zero-padded 768->3072 vectors with native 3072-dim embeddings.

Resumable — saves progress to data/artifacts/reembed_progress.json so it can
pick up where it left off if the daily quota runs out.

Usage:
    python scripts/reembed_full_corpus.py              # embed + upload (resumable)
    python scripts/reembed_full_corpus.py --dry-run     # parse pairs, show stats, don't embed
    python scripts/reembed_full_corpus.py --reset       # clear progress and start over
"""
from __future__ import annotations

import argparse
import json
import os
import re
import requests
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROCESSED_DIR = Path("data/processed/ai_cleaned")
PROGRESS_FILE = Path("data/artifacts/reembed_progress.json")
BATCH_SIZE = 20  # texts per Gemini API call
UPLOAD_BATCH = 100  # vectors per Pinecone upsert
RATE_SLEEP = 1.5  # seconds between embed calls (~40/min, well under 100/min limit)


def parse_all_pairs() -> list[dict]:
    """Parse every QA pair from the processed corpus."""
    pairs = []
    for fpath in sorted(PROCESSED_DIR.rglob("*_processed.txt")):
        rel = str(fpath.relative_to(PROCESSED_DIR)).replace("\\", "/")
        try:
            text = fpath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  SKIP (read error): {rel} -> {e}")
            continue

        blocks = re.split(r"(?=\*\*RAY PEAT:\*\*)", text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            m = re.match(
                r"\*\*RAY PEAT:\*\*\s*(.*?)(?:\n\s*\n\s*\*\*CONTEXT:\*\*\s*(.*?))?$",
                block,
                re.DOTALL,
            )
            if m:
                quote = m.group(1).strip()
                context = (m.group(2) or "").strip()
                if quote:
                    pairs.append({
                        "source_file": rel,
                        "ray_peat_response": quote,
                        "context": context,
                    })
    return pairs


def load_progress() -> dict:
    """Load resume state from disk."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"embedded_batches": 0, "vectors": [], "phase": "embed"}


def save_progress(state: dict):
    """Save resume state to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(state, f)


def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index("ray-peat-corpus")


def _embed_one_rest(text: str, api_key: str) -> list[float] | None:
    """Embed a single text via Gemini REST API. Returns values or None on failure."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}}

    for attempt in range(1, 16):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
        except Exception as e:
            print(f"    Network error (attempt {attempt}): {e}", flush=True)
            time.sleep(5)
            continue

        if r.status_code == 200:
            vals = r.json().get("embedding", {}).get("values")
            return vals

        err = r.text
        # Daily quota -> caller handles this
        if r.status_code == 429 and ("free_tier" in err.lower() or "per day" in err.lower()):
            raise RuntimeError("RATE_LIMITED_DAILY")

        # Per-minute rate limit -> backoff
        if r.status_code == 429:
            wait = 30
            delay_match = re.search(r"retryDelay.*?(\d+)s", err)
            if delay_match:
                wait = int(delay_match.group(1)) + 3
            if attempt <= 2:
                # quiet on first couple retries
                pass
            else:
                print(f"    Rate limited (attempt {attempt}/15), waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue

        # Other error
        print(f"    API error {r.status_code} (attempt {attempt}): {err[:150]}", flush=True)
        time.sleep(5)

    return None


def embed_pairs(pairs: list[dict], state: dict) -> dict:
    """Embed QA pairs with Gemini REST API, one at a time. Resumable."""
    api_key = os.environ["GEMINI_API_KEY"]
    start_idx = state.get("embedded_count", 0)
    vectors = state.get("vectors", [])

    print(f"\nEmbedding {len(pairs)} pairs via REST API (1 call per pair)")
    if start_idx > 0:
        print(f"  Resuming from pair {start_idx}/{len(pairs)} ({len(vectors)} vectors done)")

    for idx in range(start_idx, len(pairs)):
        pair = pairs[idx]
        text = f"{pair['context']} {pair['ray_peat_response']}"

        try:
            values = _embed_one_rest(text, api_key)
        except RuntimeError as e:
            if "DAILY" in str(e):
                print(f"\n  Daily quota exhausted at pair {idx}/{len(pairs)}.")
                print(f"  Progress saved. Run again tomorrow to resume.")
                state["embedded_count"] = idx
                state["vectors"] = vectors
                state["phase"] = "embed"
                save_progress(state)
                sys.exit(0)
            raise

        if values is None:
            print(f"  SKIP pair {idx} after 15 retries")
            continue

        vectors.append({
            "id": f"vec_{idx}",
            "values": values,
            "metadata": {
                "source_file": pair["source_file"],
                "context": pair["context"][:5_000],
                "ray_peat_response": pair["ray_peat_response"][:20_000],
                "tokens": len(pair["ray_peat_response"].split()),
            },
        })

        done = idx + 1
        pct = done * 100 // len(pairs)
        if done % 100 == 0 or done == len(pairs):
            print(f"  [{pct:3d}%] Embedded {done}/{len(pairs)} pairs", flush=True)

        # Save progress every 200 pairs
        if done % 200 == 0:
            state["embedded_count"] = done
            state["vectors"] = vectors
            state["phase"] = "embed"
            save_progress(state)

        time.sleep(0.7)  # ~85 req/min, under 100/min limit

    # All done embedding
    state["embedded_count"] = len(pairs)
    state["vectors"] = vectors
    state["phase"] = "upload"
    save_progress(state)
    print(f"\nEmbedding complete: {len(vectors)} vectors ready for upload.")
    return state


def delete_and_upload(idx, state: dict):
    """Delete all old vectors and upload new ones."""
    vectors = state.get("vectors", [])
    if not vectors:
        print("No vectors to upload.")
        return

    # Step 1: Delete all existing vectors
    print(f"\nDeleting all existing vectors from index...")
    stats = idx.describe_index_stats()
    old_count = stats.total_vector_count
    print(f"  Current vector count: {old_count}")

    if old_count > 0:
        # Delete by namespace (default namespace)
        idx.delete(delete_all=True)
        time.sleep(3)
        stats = idx.describe_index_stats()
        print(f"  After delete: {stats.total_vector_count} vectors")

    # Step 2: Upload new vectors in batches
    print(f"\nUploading {len(vectors)} new vectors in batches of {UPLOAD_BATCH}...")
    uploaded = 0
    for i in range(0, len(vectors), UPLOAD_BATCH):
        batch = vectors[i : i + UPLOAD_BATCH]
        upsert_data = [
            (
                v["id"],
                v["values"],
                {
                    **v["metadata"],
                    "context": v["metadata"].get("context", "")[:5_000],
                    "ray_peat_response": v["metadata"].get("ray_peat_response", "")[:20_000],
                },
            )
            for v in batch
        ]

        retries = 0
        while retries < 5:
            try:
                idx.upsert(vectors=upsert_data)
                break
            except Exception as e:
                retries += 1
                print(f"  Upload error (retry {retries}): {e}")
                time.sleep(2 ** retries)

        uploaded += len(batch)
        if uploaded % 500 == 0 or uploaded == len(vectors):
            print(f"  Uploaded {uploaded}/{len(vectors)}", flush=True)

    time.sleep(3)
    stats = idx.describe_index_stats()
    print(f"\nUpload complete!")
    print(f"  Index: {stats.total_vector_count} vectors, dimension={stats.dimension}")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    print("  Progress file cleaned up.")


def main():
    parser = argparse.ArgumentParser(description="Re-embed full corpus with native 3072-dim Gemini embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Parse and count pairs only")
    parser.add_argument("--reset", action="store_true", help="Clear saved progress and start fresh")
    args = parser.parse_args()

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            print("Progress cleared.")
        else:
            print("No progress file found.")
        return

    # Step 1: Parse corpus
    print("Parsing corpus...")
    pairs = parse_all_pairs()
    print(f"  Found {len(pairs)} QA pairs from {PROCESSED_DIR}")

    if args.dry_run:
        batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n  API calls needed: {batches}")
        print(f"  Est. runtime: {batches * RATE_SLEEP / 60:.1f} min (at {RATE_SLEEP}s/batch)")
        print(f"  Daily quota: ~1000 calls -> {(batches + 999) // 1000} day(s)")
        return

    # Step 2: Embed (resumable)
    state = load_progress()
    if state.get("phase") == "upload" and state.get("vectors"):
        print(f"\nPrevious run completed embedding ({len(state['vectors'])} vectors). Skipping to upload.")
    else:
        state = embed_pairs(pairs, state)

    # Step 3: Delete old + upload new
    idx = get_pinecone_index()
    delete_and_upload(idx, state)

    print("\nDone! All vectors are now native 3072-dim Gemini embeddings.")


if __name__ == "__main__":
    main()
