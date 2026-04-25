#!/usr/bin/env python3
"""
Fine-tune cross-encoder/ms-marco-MiniLM-L-6-v2 on Ray Peat domain data.

Converts (anchor, positive, negative) triples into cross-encoder training
samples: (query, positive_passage) → 1.0, (query, negative_passage) → 0.0.

This teaches the reranker which Peat passages are actually relevant to a
query — fixing cases where generic ms-marco scores give low relevance to
domain-relevant passages (e.g., "bioidentical" matching hormone passages).

Input:  data/training/embedding_pairs_v2.jsonl
Output: data/models/reranker/peat-reranker-ft/

Usage:
    python scripts/finetune_cross_encoder.py
    python scripts/finetune_cross_encoder.py --epochs 3 --batch-size 16
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Config ---
TRAINING_DATA = PROJECT_ROOT / "data" / "training" / "embedding_pairs_v2.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "reranker" / "peat-reranker-ft"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_training_samples() -> list[dict]:
    """Load triples and convert to cross-encoder format.

    Each triple (anchor, positive, negative) becomes two samples:
      - (anchor, positive) → label 1.0
      - (anchor, negative) → label 0.0
    """
    samples = []
    with open(TRAINING_DATA, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            anchor = obj["anchor"]
            positive = obj["positive"]

            # Positive pair
            samples.append({
                "query": anchor,
                "passage": positive[:600],  # truncate for 512-token limit
                "label": 1.0,
            })

            # Negative pair (if available)
            negative = obj.get("negative")
            if negative:
                samples.append({
                    "query": anchor,
                    "passage": negative[:600],
                    "label": 0.0,
                })

    print(f"Loaded {len(samples)} cross-encoder training samples")
    pos = sum(1 for s in samples if s["label"] > 0.5)
    neg = len(samples) - pos
    print(f"  Positive: {pos}, Negative: {neg}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    import torch
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {BASE_MODEL}...")
    model = CrossEncoder(BASE_MODEL, max_length=512, device=device)
    print("Model loaded.")

    # Load data
    print(f"\nLoading training data from {TRAINING_DATA}...")
    samples = load_training_samples()

    # Split 90/10
    import random
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    print(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # Convert to InputExample format for cross-encoder trainer
    from sentence_transformers.readers import InputExample
    train_examples = [
        InputExample(texts=[s["query"], s["passage"]], label=s["label"])
        for s in train_samples
    ]

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size
    )

    # Evaluator — group eval samples by query for ranking eval
    eval_by_query = {}
    for s in eval_samples:
        q = s["query"]
        if q not in eval_by_query:
            eval_by_query[q] = {"query": q, "positive": [], "negative": []}
        if s["label"] > 0.5:
            eval_by_query[q]["positive"].append(s["passage"])
        else:
            eval_by_query[q]["negative"].append(s["passage"])

    # Only keep queries with both pos and neg for evaluation
    eval_data = [v for v in eval_by_query.values() if v["positive"] and v["negative"]]
    evaluator = None
    if eval_data:
        evaluator = CERerankingEvaluator(eval_data, name="peat-rerank-eval")
        print(f"Evaluator: {len(eval_data)} queries with pos+neg pairs")

    # Train
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    print(f"\nStarting fine-tuning: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"Warmup steps: {warmup_steps}")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(OUTPUT_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\nModel saved to: {OUTPUT_DIR}")

    # Sanity check
    print("\n--- Sanity Check ---")
    test_pairs = [
        ("are seed oils bad for you?", "Ray Peat: polyunsaturated fats are immunosuppressive and promote inflammation"),
        ("are seed oils bad for you?", "Ray Peat: I recommend eating liver once a week for vitamin A"),
        ("how to fix low energy", "Ray Peat: thyroid hormone with adequate glucose usually cures low energy"),
        ("how to fix low energy", "Ray Peat: the structure of water in cells is important for biology"),
    ]
    scores = model.predict(test_pairs)
    for (q, p), s in zip(test_pairs, scores):
        print(f"  [{s:+.3f}] Q: {q[:40]}  P: {p[:60]}")


if __name__ == "__main__":
    main()
