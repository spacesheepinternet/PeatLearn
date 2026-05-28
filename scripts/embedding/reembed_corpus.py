#!/usr/bin/env python3
"""
Re-embed the Ray Peat corpus using the fine-tuned EmbeddingGemma model.

Reads metadata from the existing Gemini embedding run, re-embeds each
Q&A pair using the local fine-tuned model, and saves new .npy + .json files.

Input:  data/embeddings/vectors/metadata_20250728_221826.json
Model:  data/models/embeddings/peat-embeddinggemma-ft/
Output: data/embeddings/vectors/embeddings_gemma_ft_<timestamp>.npy
        data/embeddings/vectors/metadata_gemma_ft_<timestamp>.json

Usage:
    python scripts/embedding/reembed_corpus.py
    python scripts/embedding/reembed_corpus.py --batch-size 128
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Config ---
METADATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "vectors" / "metadata_20250728_221826.json"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "embeddings" / "peat-embeddinggemma-ft"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "vectors"


def load_metadata() -> list[dict]:
    """Load existing corpus metadata."""
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} entries from {METADATA_FILE.name}")
    return metadata


def build_texts(metadata: list[dict]) -> list[str]:
    """Build embedding text for each Q&A pair (matches training data format)."""
    texts = []
    for entry in metadata:
        context = entry.get("context", "")
        response = entry.get("ray_peat_response", "")
        text = f"Context: {context}\nRay Peat: {response}"
        texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load fine-tuned model
    print(f"\nLoading fine-tuned model from {MODEL_DIR}...")
    model = SentenceTransformer(str(MODEL_DIR), trust_remote_code=True, device=device)
    print(f"Model loaded. Output dim: {model.get_embedding_dimension()}")

    # Load corpus metadata
    metadata = load_metadata()
    texts = build_texts(metadata)
    print(f"Built {len(texts)} texts for embedding")

    # Embed in batches
    print(f"\nEmbedding {len(texts)} texts (batch_size={args.batch_size})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start
    rate = len(texts) / elapsed
    print(f"Done in {elapsed:.1f}s ({rate:.0f} texts/sec)")
    print(f"Embeddings shape: {embeddings.shape}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npy_path = OUTPUT_DIR / f"embeddings_gemma_ft_{timestamp}.npy"
    meta_path = OUTPUT_DIR / f"metadata_gemma_ft_{timestamp}.json"

    np.save(str(npy_path), embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"\nSaved embeddings: {npy_path.name} ({embeddings.shape})")
    print(f"Saved metadata:   {meta_path.name} ({len(metadata)} entries)")

    # Quick sanity check — embed a colloquial query and find nearest neighbors
    print("\n--- Sanity Check ---")
    test_queries = [
        "are seed oils bad for you?",
        "how to fix low energy and brain fog",
        "is keto good or bad?",
    ]
    query_embs = model.encode(test_queries, normalize_embeddings=True)
    for i, query in enumerate(test_queries):
        sims = np.dot(embeddings, query_embs[i])
        top_idx = np.argsort(sims)[-3:][::-1]
        print(f"\nQuery: '{query}'")
        for rank, idx in enumerate(top_idx, 1):
            entry = metadata[idx]
            print(f"  #{rank} (sim={sims[idx]:.4f}): [{entry['source_file'][:40]}] {entry['context'][:80]}")


if __name__ == "__main__":
    main()
