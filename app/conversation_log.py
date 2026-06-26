"""
Conversation logger for the public web API.

Persists one row per answered question — timestamp, the question, the answer,
confidence tier, which sources were cited, latency, and a *hashed* visitor IP —
to a SQLite DB on the mounted /data volume so it survives container rebuilds /
deploys. Used to audit answer quality, spot abstentions/hallucinations, and see
what people actually ask.

Privacy: the raw IP is never stored — only a salted SHA-256 prefix, enough to
group a session or spot abuse without keeping PII. Set CONV_HASH_SALT in the
server .env to make the hash non-reversible across a rainbow table.

Best-effort: every function swallows its own errors so logging can never break
or slow a user's request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("peatlearn.conversation_log")

# Production sets CONV_DB_PATH=/data/conversations.db on the mounted volume.
_DB_PATH = os.getenv(
    "CONV_DB_PATH", os.path.join(tempfile.gettempdir(), "peatlearn_conversations.db")
)
_HASH_SALT = os.getenv("CONV_HASH_SALT", "")

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = _DB_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _conn = sqlite3.connect(path, check_same_thread=False)
        except (OSError, sqlite3.Error):
            fallback = os.path.join(tempfile.gettempdir(), "peatlearn_conversations.db")
            logger.warning("Conversation DB at %s unavailable; using %s", path, fallback)
            _conn = sqlite3.connect(fallback, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, "
            "ip_raw TEXT, "
            "ip_hash TEXT, "
            "question TEXT NOT NULL, "
            "answer TEXT, "
            "confidence TEXT, "
            "n_sources INTEGER, "
            "sources TEXT, "
            "followups TEXT, "
            "latency_s REAL)"
        )
        # Migrate DBs created before ip_raw existed (older rows stay NULL).
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "ip_raw" not in cols:
            _conn.execute("ALTER TABLE conversations ADD COLUMN ip_raw TEXT")
        _conn.commit()
    return _conn


def hash_ip(ip: str) -> str:
    """Salted, truncated SHA-256 of the IP — pseudonymous, never the raw value."""
    if not ip:
        return ""
    return hashlib.sha256((_HASH_SALT + ip).encode("utf-8")).hexdigest()[:16]


def log(
    question: str,
    answer: str,
    confidence: Optional[str],
    sources: Optional[List[Dict[str, Any]]],
    latency_s: float,
    ip: str = "",
    followups: Optional[List[str]] = None,
) -> None:
    """Persist one Q&A row. Never raises."""
    try:
        src_files = [
            str(s.get("source_file", ""))
            for s in (sources or [])
            if isinstance(s, dict)
        ][:12]
        row = (
            datetime.now(timezone.utc).isoformat(),
            (ip or "") or None,
            hash_ip(ip),
            (question or "")[:4000],
            (answer or "")[:20000],
            (confidence or "") or None,
            len(src_files),
            json.dumps(src_files, ensure_ascii=False),
            json.dumps(followups or [], ensure_ascii=False),
            round(float(latency_s), 3),
        )
        with _lock:
            conn = _db()
            conn.execute(
                "INSERT INTO conversations "
                "(ts, ip_raw, ip_hash, question, answer, confidence, n_sources, sources, followups, latency_s) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.commit()
    except Exception:  # noqa: BLE001 — logging must never break a request
        logger.warning("Conversation log write failed", exc_info=True)


def recent(limit: int = 100) -> List[Dict[str, Any]]:
    """Return the most recent rows (newest first). Never raises."""
    try:
        limit = max(1, min(int(limit), 1000))
        with _lock:
            conn = _db()
            cur = conn.execute(
                "SELECT id, ts, ip_raw, ip_hash, question, answer, confidence, n_sources, "
                "sources, followups, latency_s "
                "FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            out: List[Dict[str, Any]] = []
            for r in cur.fetchall():
                d = dict(zip(cols, r))
                for k in ("sources", "followups"):
                    try:
                        d[k] = json.loads(d[k]) if d[k] else []
                    except Exception:
                        d[k] = []
                out.append(d)
            return out
    except Exception:  # noqa: BLE001
        logger.warning("Conversation log read failed", exc_info=True)
        return []


def stats() -> Dict[str, Any]:
    """Aggregate counts — total, by confidence tier. Never raises."""
    try:
        with _lock:
            conn = _db()
            total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            by_tier = dict(
                conn.execute(
                    "SELECT COALESCE(confidence,'(none)'), COUNT(*) "
                    "FROM conversations GROUP BY confidence"
                ).fetchall()
            )
        return {"total": total, "by_confidence": by_tier}
    except Exception:  # noqa: BLE001
        return {"total": 0, "by_confidence": {}}
