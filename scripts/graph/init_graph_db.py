"""
Initialise the knowledge graph SQLite database.

Creates data/knowledge_graph/triples.db with the full schema.
Safe to re-run — uses CREATE TABLE IF NOT EXISTS.

Usage:
  python scripts/graph/init_graph_db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/knowledge_graph/triples.db")


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS triples (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            subject         TEXT    NOT NULL,
            relationship    TEXT    NOT NULL,
            object          TEXT    NOT NULL,
            verbatim_quote  TEXT    NOT NULL,
            conditional     TEXT,
            claim_strength  TEXT    NOT NULL CHECK (claim_strength IN ('explicit', 'implied')),
            source_doc      TEXT    NOT NULL,
            source_file     TEXT    NOT NULL,
            doc_date        TEXT,          -- YYYY or YYYY-MM, from filename heuristic
            human_reviewed  INTEGER NOT NULL DEFAULT 0,  -- 0 = not reviewed, 1 = approved, -1 = rejected
            inserted_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Fast lookups by entity pair (the core graph query)
        CREATE INDEX IF NOT EXISTS idx_subject   ON triples (subject);
        CREATE INDEX IF NOT EXISTS idx_object    ON triples (object);
        CREATE INDEX IF NOT EXISTS idx_pair      ON triples (subject, object);
        CREATE INDEX IF NOT EXISTS idx_doc       ON triples (source_doc);
        CREATE INDEX IF NOT EXISTS idx_reviewed  ON triples (human_reviewed);

        -- Edge summary view: only show edges confirmed by 3+ distinct documents
        -- (filtered in queries, not as a view, because doc count needs GROUP BY)
    """)

    con.commit()
    con.close()
    print(f"Database ready: {db_path.resolve()}")


if __name__ == "__main__":
    init_db()
