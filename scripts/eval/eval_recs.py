#!/usr/bin/env python3
"""
Offline recommender evaluation for MF model on interactions.db

- Split: for each user, leave-one-out on their last positive interaction
- Candidates: the left-out positive item + N sampled negatives from items the user never interacted with
- Models compared:
  - MF-only: dot(U[user], V[item])
  - Popularity: rank by global item frequency (fallback baseline)
- Metrics: HitRate@K, NDCG@K, MAP@K (K=10 by default)

Usage:
  ./venv/bin/python scripts/eval/eval_recs.py
"""
from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np

DB_PATH = Path("data/user_interactions/interactions.db")
MODEL_PATH = Path("data/models/recs/mf_model.npz")
K = 10
NUM_NEG = 50
SEED = 42


def derive_content_id(ctx: str | None) -> str | None:
    try:
        d = json.loads(ctx or "{}")
    except Exception:
        return None
    sources = d.get("sources") if isinstance(d, dict) else None
    if isinstance(sources, list) and sources:
        s0 = str(sources[0])
        left = s0.split("(")[0].strip()
        if "." in left:
            left = left.split(".", 1)[1].strip()
        return left
    return None


def load_positives() -> Tuple[Dict[str, List[str]], Counter, Set[str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT user_id, user_feedback, context FROM interactions WHERE user_id IS NOT NULL ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    user_pos: Dict[str, List[str]] = defaultdict(list)
    item_counts: Counter = Counter()
    all_items: Set[str] = set()
    for user_id, feedback, ctx in rows:
        cid = derive_content_id(ctx)
        if not cid:
            continue
        all_items.add(cid)
        if feedback is not None and int(feedback) > 0:
            user_pos[str(user_id)].append(cid)
            item_counts[cid] += 1
    return user_pos, item_counts, all_items


def hitrate_at_k(ranked: List[str], pos_item: str, k: int) -> float:
    return 1.0 if pos_item in ranked[:k] else 0.0


def ndcg_at_k(ranked: List[str], pos_item: str, k: int) -> float:
    for idx, item in enumerate(ranked[:k]):
        if item == pos_item:
            return 1.0 / np.log2(idx + 2)
    return 0.0


def map_at_k(ranked: List[str], pos_item: str, k: int) -> float:
    # Single positive case: same as precision at rank of the positive
    for idx, item in enumerate(ranked[:k]):
        if item == pos_item:
            return 1.0 / (idx + 1)
    return 0.0


def main():
    rng = random.Random(SEED)
    user_pos, item_counts, all_items = load_positives()
    if not user_pos:
        print("No positive interactions found; cannot evaluate.")
        return

    # Load MF model if present
    mf_available = MODEL_PATH.exists()
    if mf_available:
        data = np.load(MODEL_PATH, allow_pickle=True)
        U = data['U']
        V = data['V']
        user_to_idx = dict(data['user_to_idx'])
        item_to_idx = dict(data['item_to_idx'])
    else:
        print("MF model not found. Evaluating only popularity baseline.")

    # Evaluate leave-one-out per user with at least 2 positives
    users = [u for u, pos in user_pos.items() if len(pos) >= 2]
    if not users:
        print("No users with >=2 positives; cannot run leave-one-out.")
        return

    hr_mf = []
    ndcg_mf = []
    map_mf = []
    hr_pop = []
    ndcg_pop = []
    map_pop = []

    for u in users:
        pos_list = user_pos[u]
        test_item = pos_list[-1]
        train_items = set(pos_list[:-1])
        # Sample negatives
        neg_pool = list(all_items - train_items - {test_item})
        if len(neg_pool) < NUM_NEG:
            negs = neg_pool
        else:
            negs = rng.sample(neg_pool, NUM_NEG)
        cand = [test_item] + negs

        # Popularity ranking
        cand_sorted_pop = sorted(cand, key=lambda i: item_counts[i], reverse=True)
        hr_pop.append(hitrate_at_k(cand_sorted_pop, test_item, K))
        ndcg_pop.append(ndcg_at_k(cand_sorted_pop, test_item, K))
        map_pop.append(map_at_k(cand_sorted_pop, test_item, K))

        # MF ranking
        if mf_available and u in user_to_idx:
            uidx = user_to_idx[u]
            def mf_score(item: str):
                idx = item_to_idx.get(item)
                if idx is None:
                    return 0.0
                return float(np.dot(U[uidx], V[idx]))
            cand_sorted_mf = sorted(cand, key=mf_score, reverse=True)
            hr_mf.append(hitrate_at_k(cand_sorted_mf, test_item, K))
            ndcg_mf.append(ndcg_at_k(cand_sorted_mf, test_item, K))
            map_mf.append(map_at_k(cand_sorted_mf, test_item, K))

    def summarize(name: str, hr, ndcg, m):
        if hr:
            print(f"{name}: HR@{K}={np.mean(hr):.3f}, NDCG@{K}={np.mean(ndcg):.3f}, MAP@{K}={np.mean(m):.3f} (n={len(hr)})")
        else:
            print(f"{name}: no eval users")

    summarize("Popularity", hr_pop, ndcg_pop, map_pop)
    summarize("MF", hr_mf, ndcg_mf, map_mf)

if __name__ == "__main__":
    main()
