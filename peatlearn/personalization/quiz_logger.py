#!/usr/bin/env python3
"""
Quiz outcome logging to SQLite interactions DB for question-level analytics.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any

DB_PATH = Path("data/user_interactions/interactions.db")


def log_quiz_outcome(user_id: str, quiz_id: str, question_id: str, topic: str, correct: bool, time_taken: float, context: Dict[str, Any] | None = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            quiz_id TEXT,
            question_id TEXT,
            topic TEXT,
            correct INTEGER,
            time_taken REAL,
            context TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        INSERT INTO quiz_outcomes (user_id, quiz_id, question_id, topic, correct, time_taken, context)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, quiz_id, question_id, topic, int(bool(correct)), float(time_taken), json.dumps(context or {})
        )
    )
    conn.commit()
    conn.close()
