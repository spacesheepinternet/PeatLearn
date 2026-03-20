#!/usr/bin/env python3
"""
Matrix Factorization (implicit-feedback) trainer for personalized recommendations.

- Reads interactions from SQLite: data/user_interactions/interactions.db
- Derives content_id from the first RAG source when explicit content_id is not present
- Trains with weighted pointwise targets: 1.0 for 👍, optionally small negatives if provided
- Saves: data/models/recs/mf_model.npz with user/item embeddings and mappings

Usage (CLI):
  ./venv/bin/python -m src.recommendation.mf_trainer
"""
from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DB_PATH = Path("data/user_interactions/interactions.db")
MODEL_DIR = Path("data/models/recs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "mf_model.npz"


def _derive_content_id_from_context(ctx: str | None) -> str | None:
    try:
        d = json.loads(ctx or "{}")
    except Exception:
        d = {}
    sources = d.get("sources") if isinstance(d, dict) else None
    if isinstance(sources, list) and sources:
        s0 = str(sources[0])
        try:
            left = s0.split("(")[0].strip()
            if "." in left:
                left = left.split(".", 1)[1].strip()
            return left
        except Exception:
            return s0
    return None


def load_interactions(db_path: Path = DB_PATH) -> List[Tuple[str, str, float]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id, user_feedback, context
        FROM interactions
        WHERE user_id IS NOT NULL
        ORDER BY id ASC
        """
    )
    rows = cur.fetchall()
    conn.close()
    interactions: List[Tuple[str, str, float]] = []
    for user_id, feedback, ctx in rows:
        content_id = _derive_content_id_from_context(ctx)
        if not content_id:
            continue
        # Map feedback (rating 1..10 or legacy +/-1) to implicit target in [0,1]
        w = None
        if feedback is None:
            continue
        try:
            fb = int(feedback)
            if 1 <= fb <= 10:
                w = (fb - 1) / 9.0
            elif fb in (-1, 1):
                w = 1.0 if fb > 0 else 0.0
        except Exception:
            w = None
        if w is None:
            continue
        interactions.append((str(user_id), content_id, float(w)))
    return interactions


def build_mappings(interactions: List[Tuple[str, str, float]]):
    users = sorted({u for u, _, _ in interactions})
    items = sorted({i for _, i, _ in interactions})
    user_to_idx = {u: idx for idx, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}
    return user_to_idx, item_to_idx


def train_mf(
    interactions: List[Tuple[int, int, float]],
    num_users: int,
    num_items: int,
    dim: int = 32,
    lr: float = 0.05,
    reg: float = 0.01,
    epochs: int = 15,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    U = 0.1 * rng.standard_normal((num_users, dim)).astype(np.float32)
    V = 0.1 * rng.standard_normal((num_items, dim)).astype(np.float32)

    for ep in range(epochs):
        rng.shuffle(interactions)
        for u, i, w in interactions:
            pu = U[u]
            qi = V[i]
            r_hat = np.dot(pu, qi)
            # Weighted pointwise target ~1.0
            err = (w - r_hat)
            # SGD updates
            U[u] += lr * (err * qi - reg * pu)
            V[i] += lr * (err * pu - reg * qi)
        lr *= 0.95
    return U, V


def save_model(U, V, user_to_idx: Dict[str, int], item_to_idx: Dict[str, int], path: Path = MODEL_PATH):
    np.savez(
        path,
        U=U,
        V=V,
        user_to_idx=np.array(list(user_to_idx.items()), dtype=object),
        item_to_idx=np.array(list(item_to_idx.items()), dtype=object),
    )


def main():
    interactions_raw = load_interactions()
    if not interactions_raw:
        print("No positive interactions found. Skipping training.")
        return
    user_to_idx, item_to_idx = build_mappings(interactions_raw)
    triples: List[Tuple[int, int, float]] = []
    for u, i, w in interactions_raw:
        if u in user_to_idx and i in item_to_idx:
            triples.append((user_to_idx[u], item_to_idx[i], w))
    if not triples:
        print("No valid triples. Skipping training.")
        return
    U, V = train_mf(triples, num_users=len(user_to_idx), num_items=len(item_to_idx))
    save_model(U, V, user_to_idx, item_to_idx)
    print(f"Saved model to {MODEL_PATH} with {len(user_to_idx)} users and {len(item_to_idx)} items.")


if __name__ == "__main__":
    main()
