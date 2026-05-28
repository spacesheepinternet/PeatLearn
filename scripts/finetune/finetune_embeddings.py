#!/usr/bin/env python3
"""
Fine-tune EmbeddingGemma on Ray Peat domain-specific training pairs.

Uses MultipleNegativesRankingLoss (contrastive/InfoNCE) to push colloquial
query embeddings closer to their matching Peat corpus passages.

Input:  data/training/embedding_pairs.jsonl
Output: data/models/embeddings/peat-embeddinggemma-ft/

Usage:
    python scripts/finetune/finetune_embeddings.py
    python scripts/finetune/finetune_embeddings.py --epochs 3 --batch-size 16
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

# --- Config ---
TRAINING_DATA = PROJECT_ROOT / "data" / "training" / "embedding_pairs.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "embeddings" / "peat-embeddinggemma-ft"
BASE_MODEL = "google/embeddinggemma-300m"


def load_training_data() -> Dataset:
    """Load training pairs from JSONL into a HuggingFace Dataset."""
    records = []
    with open(TRAINING_DATA, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            record = {
                "anchor": obj["anchor"],
                "positive": obj["positive"],
            }
            if obj.get("negative"):
                record["negative"] = obj["negative"]
            records.append(record)

    print(f"Loaded {len(records)} training pairs")
    has_negatives = sum(1 for r in records if "negative" in r)
    print(f"  with hard negatives: {has_negatives}")

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model — cap sequence length to fit in 8GB VRAM
    print(f"\nLoading {BASE_MODEL}...")
    model = SentenceTransformer(BASE_MODEL, trust_remote_code=True, device=device)
    model.max_seq_length = 256
    print(f"Model loaded. Output dim: {model.get_embedding_dimension()}, max_seq_length: {model.max_seq_length}")

    # Load data
    print(f"\nLoading training data from {TRAINING_DATA}...")
    dataset = load_training_data()

    # Split 90/10 for train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Loss — MultipleNegativesRankingLoss works with (anchor, positive) pairs
    # and optionally (anchor, positive, negative) triples.
    # In-batch negatives are used automatically.
    train_loss = MultipleNegativesRankingLoss(model)

    # Training args
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=5,
        seed=42,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
    )

    print(f"\nStarting fine-tuning: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    trainer.train()

    # Save final model
    model.save(str(OUTPUT_DIR))
    print(f"\nModel saved to: {OUTPUT_DIR}")

    # Quick sanity check
    print("\nSanity check — embedding dimensions:")
    test_emb = model.encode(["What does Ray Peat say about carbs?"])
    print(f"  Output shape: {test_emb.shape}")


if __name__ == "__main__":
    main()
