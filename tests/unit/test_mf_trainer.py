#!/usr/bin/env python3
import os
import json
import sqlite3
from pathlib import Path

import numpy as np

from peatlearn.recommendation.mf_trainer import load_interactions, build_mappings, train_mf


def make_temp_db(tmp_path: Path):
    db_path = tmp_path / "interactions.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT,
            timestamp TEXT,
            user_query TEXT,
            llm_response TEXT,
            topic TEXT,
            user_feedback INTEGER,
            interaction_type TEXT,
            response_time REAL,
            context TEXT,
            jargon_score REAL,
            similarity_confidence REAL
        )
        """
    )
    # User U1 likes item A (positive), user U1 ignores item B
    ctx_A = json.dumps({"sources": ["1. path/to/item_A.txt (relevance: 0.8)"]})
    cur.execute("INSERT INTO interactions (user_id, user_feedback, context) VALUES (?, ?, ?)", ("U1", 1, ctx_A))
    # User U2 likes item B
    ctx_B = json.dumps({"sources": ["1. path/to/item_B.txt (relevance: 0.7)"]})
    cur.execute("INSERT INTO interactions (user_id, user_feedback, context) VALUES (?, ?, ?)", ("U2", 1, ctx_B))
    conn.commit()
    conn.close()
    return db_path


def test_mf_training_and_scoring(tmp_path):
    # Arrange: build a temporary interactions DB
    db_path = make_temp_db(tmp_path)

    # Act: load implicit positives
    inter = load_interactions(db_path)
    assert inter, "No interactions loaded from temp DB"

    user_to_idx, item_to_idx = build_mappings(inter)
    triples = [(user_to_idx[u], item_to_idx[i], w) for u, i, w in inter]
    U, V = train_mf(triples, num_users=len(user_to_idx), num_items=len(item_to_idx), epochs=8, dim=16)

    # Assert: each user ranks their positive item higher than the other item
    def score(u, i):
        return float(np.dot(U[user_to_idx[u]], V[item_to_idx[i]]))

    s_u1_pos = score("U1", "path/to/item_A.txt")
    s_u1_neg = score("U1", "path/to/item_B.txt")
    s_u2_pos = score("U2", "path/to/item_B.txt")
    s_u2_neg = score("U2", "path/to/item_A.txt")

    assert s_u1_pos > s_u1_neg, f"U1 should prefer A over B (got {s_u1_pos:.3f} vs {s_u1_neg:.3f})"
    assert s_u2_pos > s_u2_neg, f"U2 should prefer B over A (got {s_u2_pos:.3f} vs {s_u2_neg:.3f})"

    # Precision@1 across users (toy):
    p1 = 0
    for u, pos_item in [("U1", "path/to/item_A.txt"), ("U2", "path/to/item_B.txt")]:
        items = ["path/to/item_A.txt", "path/to/item_B.txt"]
        ranked = sorted(items, key=lambda i: score(u, i), reverse=True)
        if ranked[0] == pos_item:
            p1 += 1
    p1 /= 2.0
    assert p1 >= 0.5, f"Precision@1 expected >= 0.5, got {p1}"
