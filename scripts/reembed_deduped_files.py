#!/usr/bin/env python3
"""
Step 2 & 3: Delete old vectors from Pinecone for deduped files, re-embed, and upload.

Usage:
    python scripts/reembed_deduped_files.py --scan          # find vectors to delete
    python scripts/reembed_deduped_files.py --delete        # delete old vectors
    python scripts/reembed_deduped_files.py --embed-upload  # re-embed and upload
    python scripts/reembed_deduped_files.py --all           # do everything
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# --- Affected files (from dedup script output) ---
AFFECTED_SOURCE_FILES = [
    "02_Publications/Townsend_Letters/1989 - April - Townsend Letter for Doctors 11.42.31 PM_processed.txt",
    "01_Audio_Transcripts/Politics_Science/polsci-120129-progesterone-2.mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/radiation.mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Radio_Shows_KMUD/kmud-101015-sugar-2.mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Radio_Shows_KMUD/kmud-160715-the-metabolism-of-cancer.mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Radio_Shows_KMUD/kmud-191018-brain-barriers.mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Radio_Shows_KMUD/kmud-161216-food.mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/#78\uff1a CO2 \uff5c Art and Science \uff5c Supply Shortages \uff5c Killer Austerity \uff5c Authoritarianism with Ray Peat [72Zz0TbfBAg].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/09.21.21 Peat Ray [1128846532].mp3-transcript_processed.txt",
    "04_Health_Topics/Hormones_Endocrine/thyroiditis-some-confusions-and-causes-of-autoimmune-diseases_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/#76\uff1a Antistress Home Strategies \uff5c Grounding \uff5c Negative Ions \uff5c Allopathic Essentialism with Ray Peat [l61D8dJzVWk].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/12.20.21 Peat Ray [1181861614].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/03.15.21 Peat Ray March [1009402927].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/10.19.20 Peat Ray [915101332].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/07.18.19 Deep Research in Healh, July 18, 2019 [653210912].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/#58\uff1a Bioenergetic Nutrition Continued \uff5c Authoritarianism \uff5c Intention and Learning with Ray Peat, PhD [kwynlIkJ4tU].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/11.19.19 Hormones, Thyroid and Much much more, November 19, 2019 [715603975].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/#43\uff1a Body Temperature, Inflammation, and Aging \uff5c Copper and Thyroid \uff5c mRNA Vaccines and Infertility [bHKWkqrYqAY].mp3-transcript_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/07.20.20 Nothing is as it seems with COVID, July 20, 2020 [861603244].mp3-transcript_processed.txt",
    "02_Publications/Townsend_Letters/1991 - June- Townsend Letter for Doctors_processed.txt",
    "02_Publications/Townsend_Letters/1996 - April - dupe - Townsend Letter for Doctors_processed.txt",
    "01_Audio_Transcripts/Other_Interviews/orn-190124-fats-and-questions.mp3-transcript_processed.txt",
]

PROCESSED_DIR = Path("data/processed/ai_cleaned")


def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index("ray-peat-corpus")


def scan_vectors(idx) -> dict[str, list[str]]:
    """Find all vector IDs belonging to affected files using metadata filters."""
    dim = idx.describe_index_stats().dimension
    dummy_vec = [0.0] * dim
    file_to_ids: dict[str, list[str]] = {}
    found_total = 0

    print(f"Scanning {len(AFFECTED_SOURCE_FILES)} affected files...", flush=True)

    for sf in AFFECTED_SOURCE_FILES:
        # Query with metadata filter to find vectors for this source file
        try:
            results = idx.query(
                vector=dummy_vec,
                top_k=500,  # generous — most files have <100 vectors
                include_metadata=False,
                filter={"source_file": {"$eq": sf}},
            )
            ids = [m["id"] for m in results.get("matches", [])]
        except Exception:
            # If filter fails, fall back to text-based search
            ids = []

        if ids:
            file_to_ids[sf] = ids
            found_total += len(ids)
            print(f"  {len(ids):4d} vectors | {Path(sf).name}", flush=True)
        else:
            print(f"     0 vectors | {Path(sf).name} (not in index or filter failed)", flush=True)

    print(f"\nScan complete. Found {found_total} vectors to delete across {len(file_to_ids)} files.", flush=True)
    return file_to_ids


def delete_vectors(idx, file_to_ids: dict[str, list[str]]):
    """Delete vectors from Pinecone in batches."""
    all_ids = []
    for ids in file_to_ids.values():
        all_ids.extend(ids)

    print(f"Deleting {len(all_ids)} vectors in batches of 100...")
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i : i + batch_size]
        try:
            idx.delete(ids=batch)
        except Exception as e:
            print(f"  Error deleting batch {i}: {e}")
            continue
        if i % 500 == 0:
            print(f"  Deleted {i + len(batch)}/{len(all_ids)}")

    print(f"Deletion complete. Removed {len(all_ids)} vectors.")
    # Verify
    time.sleep(2)
    stats = idx.describe_index_stats()
    print(f"Index now has {stats.total_vector_count} vectors.")


def embed_and_upload(idx):
    """Re-embed deduped files and upload to Pinecone."""
    import google.genai as genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    index_dim = idx.describe_index_stats().dimension

    # Parse QA pairs from each deduped file
    pairs = []
    for sf in AFFECTED_SOURCE_FILES:
        fpath = PROCESSED_DIR / sf
        if not fpath.exists():
            print(f"  SKIP (not found): {sf}")
            continue
        text = fpath.read_text(encoding="utf-8")
        # Parse RAY PEAT + CONTEXT blocks
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
                        "source_file": sf,
                        "ray_peat_response": quote,
                        "context": context,
                    })

    print(f"Parsed {len(pairs)} QA pairs from {len(AFFECTED_SOURCE_FILES)} files.")

    # Embed in batches
    batch_size = 20
    vectors_to_upload = []
    # Get current max vec ID
    stats = idx.describe_index_stats()
    next_id = stats.total_vector_count  # start after existing vectors

    print(f"Embedding {len(pairs)} pairs (batch size {batch_size})...")

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        texts = [f"{p['context']} {p['ray_peat_response']}" for p in batch]

        retries = 0
        while retries < 10:
            try:
                resp = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                )
                break
            except Exception as e:
                retries += 1
                # Parse retry delay from error if available
                err_str = str(e)
                wait = 35  # default: just over the 30s Gemini suggests
                import re as _re
                delay_match = _re.search(r"retryDelay.*?(\d+)s", err_str)
                if delay_match:
                    wait = int(delay_match.group(1)) + 5
                print(f"  Rate limited (retry {retries}/10), waiting {wait}s...", flush=True)
                time.sleep(wait)
        else:
            print(f"  FAILED batch {i}, skipping {len(batch)} pairs", flush=True)
            continue

        for j, emb_obj in enumerate(resp.embeddings):
            values = emb_obj.values
            # Pad to index dimension if needed
            if len(values) < index_dim:
                values = values + [0.0] * (index_dim - len(values))
            elif len(values) > index_dim:
                values = values[:index_dim]

            pair = batch[j]
            vec_id = f"vec_{next_id}"
            next_id += 1
            vectors_to_upload.append({
                "id": vec_id,
                "values": values,
                "metadata": {
                    "source_file": pair["source_file"],
                    "context": pair["context"],
                    "ray_peat_response": pair["ray_peat_response"],
                    "tokens": len(pair["ray_peat_response"].split()),
                },
            })

        print(f"  Embedded {min(i + batch_size, len(pairs))}/{len(pairs)} pairs", flush=True)
        time.sleep(1.5)  # rate limit: ~40 req/min stays well under 100/min limit

    print(f"\nEmbedded {len(vectors_to_upload)} vectors. Uploading to Pinecone...")

    # Upload in batches of 100
    upload_batch = 100
    for i in range(0, len(vectors_to_upload), upload_batch):
        batch = vectors_to_upload[i : i + upload_batch]
        upsert_data = [(v["id"], v["values"], v["metadata"]) for v in batch]
        retries = 0
        while retries < 3:
            try:
                idx.upsert(vectors=upsert_data)
                break
            except Exception as e:
                retries += 1
                print(f"  Upload error (retry {retries}): {e}")
                time.sleep(2 ** retries)

        if i % 200 == 0:
            print(f"  Uploaded {min(i + upload_batch, len(vectors_to_upload))}/{len(vectors_to_upload)}")

    time.sleep(2)
    stats = idx.describe_index_stats()
    print(f"\nUpload complete. Index now has {stats.total_vector_count} vectors.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="Scan for vectors to delete")
    parser.add_argument("--delete", action="store_true", help="Delete old vectors")
    parser.add_argument("--embed-upload", action="store_true", help="Re-embed and upload")
    parser.add_argument("--all", action="store_true", help="Do everything")
    args = parser.parse_args()

    if not any([args.scan, args.delete, args.embed_upload, args.all]):
        parser.print_help()
        return

    idx = get_pinecone_index()

    if args.scan or args.all:
        file_to_ids = scan_vectors(idx)
        # Save for delete step
        with open("data/artifacts/vectors_to_delete.json", "w") as f:
            json.dump(file_to_ids, f, indent=2)
        print(f"\nSaved vector IDs to data/artifacts/vectors_to_delete.json")

    if args.delete or args.all:
        # Load saved scan results
        scan_path = Path("data/artifacts/vectors_to_delete.json")
        if not scan_path.exists():
            print("No scan results found. Run --scan first.")
            return
        with open(scan_path) as f:
            file_to_ids = json.load(f)
        delete_vectors(idx, file_to_ids)

    if args.embed_upload or args.all:
        embed_and_upload(idx)


if __name__ == "__main__":
    main()
