#!/usr/bin/env python3
"""
PeatLearn Web API — production HTTP service for the public website.

Wraps the live, benchmarked RAG pipeline (`peatlearn/adaptive/rag_system.py`,
the 9.64/10 path the Streamlit app uses) behind a small, stateless JSON API so a
decoupled frontend (the Vite/React app in `web/`) can call it.

This is intentionally separate from `app/api.py`, which is the older
`peatlearn/rag` path and is NOT the pipeline we want to serve publicly.

Run locally:
    uvicorn app.web_api:app --port 8080 --reload

Run in a container: see Dockerfile / docker-compose.yml.
"""

from __future__ import annotations

import hmac
import logging
import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from peatlearn.adaptive.rag_system import RayPeatRAG

logger = logging.getLogger("peatlearn.web_api")

# --- Footer markers appended by RayPeatRAG.get_rag_response --------------------
_SOURCES_MARKER = "📚 Sources:"
_CONFIDENCE_MARKER = "🔒 Confidence:"


# --- Singleton ----------------------------------------------------------------
@lru_cache(maxsize=1)
def get_rag() -> RayPeatRAG:
    """Build the RAG pipeline once per process (loads Pinecone + warms reranker)."""
    logger.info("Initializing RayPeatRAG (adaptive pipeline)...")
    return RayPeatRAG()


# --- Rate limiting ------------------------------------------------------------
# Each /api/ask call costs money (Gemini + Cohere). Two SQLite-backed caps,
# reset daily at UTC midnight, protect the budget:
#   * per-IP   : DAILY_LIMIT_PER_IP questions per visitor per day (default 10)
#   * global   : DAILY_GLOBAL_CAP   questions total per day (0 = disabled)
# The DB lives on a mounted volume so counts survive container restarts.
DAILY_LIMIT_PER_IP = int(os.getenv("DAILY_LIMIT_PER_IP", "10"))
DAILY_GLOBAL_CAP = int(os.getenv("DAILY_GLOBAL_CAP", "0"))

# Admin bypass: a request carrying X-Admin-Token == ADMIN_TOKEN skips the rate
# limit entirely. Disabled unless ADMIN_TOKEN is set (non-empty) in the server
# .env. Keep this secret out of the repo — it lives only in the server's .env.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def is_admin(request: Request) -> bool:
    """True if the request presents the correct admin token (constant-time)."""
    if not ADMIN_TOKEN:
        return False
    presented = request.headers.get("x-admin-token", "")
    return bool(presented) and hmac.compare_digest(presented, ADMIN_TOKEN)
# Production (container) sets RATE_DB_PATH=/data/ratelimit.db on a mounted
# volume. The default is a temp path so local dev / tests just work.
_RATE_DB_PATH = os.getenv(
    "RATE_DB_PATH", os.path.join(tempfile.gettempdir(), "peatlearn_ratelimit.db")
)
_GLOBAL_KEY = "*GLOBAL*"

_rate_lock = threading.Lock()
_rate_conn: Optional[sqlite3.Connection] = None


def _rate_db() -> sqlite3.Connection:
    global _rate_conn
    if _rate_conn is None:
        path = _RATE_DB_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _rate_conn = sqlite3.connect(path, check_same_thread=False)
        except (OSError, sqlite3.Error):
            # e.g. the mounted volume isn't writable — fall back to temp so the
            # API still serves (limits just won't survive a restart).
            fallback = os.path.join(tempfile.gettempdir(), "peatlearn_ratelimit.db")
            logger.warning("Rate DB at %s unavailable; using %s", path, fallback)
            _rate_conn = sqlite3.connect(fallback, check_same_thread=False)
        _rate_conn.execute(
            "CREATE TABLE IF NOT EXISTS usage ("
            "day TEXT NOT NULL, ip TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (day, ip))"
        )
        _rate_conn.commit()
    return _rate_conn


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def client_ip(request: Request) -> str:
    """Real visitor IP. Caddy sets X-Forwarded-For; take the left-most entry."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(ip: str) -> int:
    """Check + increment both counters atomically. Returns remaining per-IP quota.

    Raises HTTPException(429) with a friendly message when a cap is hit; nothing
    is incremented in that case.
    """
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _rate_lock:
        conn = _rate_db()
        cur = conn.execute("SELECT count FROM usage WHERE day=? AND ip=?", (day, ip))
        row = cur.fetchone()
        ip_count = row[0] if row else 0

        if ip_count >= DAILY_LIMIT_PER_IP:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"You've reached today's limit of {DAILY_LIMIT_PER_IP} questions. "
                    "It resets at midnight UTC — see you tomorrow!"
                ),
                headers={"Retry-After": str(_seconds_until_utc_midnight())},
            )

        if DAILY_GLOBAL_CAP > 0:
            g = conn.execute(
                "SELECT count FROM usage WHERE day=? AND ip=?", (day, _GLOBAL_KEY)
            ).fetchone()
            if (g[0] if g else 0) >= DAILY_GLOBAL_CAP:
                raise HTTPException(
                    status_code=429,
                    detail="PeatLearn has hit its daily question limit. Please try again tomorrow.",
                    headers={"Retry-After": str(_seconds_until_utc_midnight())},
                )

        conn.execute(
            "INSERT INTO usage (day, ip, count) VALUES (?, ?, 1) "
            "ON CONFLICT(day, ip) DO UPDATE SET count = count + 1",
            (day, ip),
        )
        conn.execute(
            "INSERT INTO usage (day, ip, count) VALUES (?, ?, 1) "
            "ON CONFLICT(day, ip) DO UPDATE SET count = count + 1",
            (day, _GLOBAL_KEY),
        )
        conn.commit()
        return max(0, DAILY_LIMIT_PER_IP - (ip_count + 1))


# --- Request / response models ------------------------------------------------
class ChatTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    chat_history: Optional[List[ChatTurn]] = None
    max_sources: Optional[int] = Field(default=None, ge=1, le=15)


class Source(BaseModel):
    source_file: str
    score: float
    rerank_score: Optional[float] = None
    # The retrieved passage this source contributed — lets the UI show the
    # excerpt the answer was grounded in.
    context: str = ""
    ray_peat_response: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: Optional[str] = None
    # Suggested next questions (in-domain, standalone). Empty for ungrounded /
    # abstained answers. The frontend renders these as clickable buttons.
    followups: List[str] = []


# --- App ----------------------------------------------------------------------
app = FastAPI(
    title="PeatLearn Web API",
    description="Grounded Ray Peat Q&A over the curated corpus (9.64/10 RAG pipeline).",
    version="1.0.0",
)

# CORS — comma-separated list of allowed origins (the Next.js site).
# Example: ALLOWED_ORIGINS="https://peatlearn.com,http://localhost:3000"
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _split_answer(raw: str) -> tuple[str, Optional[str]]:
    """Strip the inline Sources/Confidence footers off the answer string.

    The structured sources come from `_last_sources`; the confidence tier is
    returned separately so the frontend can render it as a badge.
    """
    body = raw
    confidence: Optional[str] = None

    if _CONFIDENCE_MARKER in body:
        body, _, conf_part = body.partition(_CONFIDENCE_MARKER)
        # conf_part looks like " HIGH | reason; reason"
        confidence = conf_part.strip().split("|", 1)[0].strip() or None

    if _SOURCES_MARKER in body:
        body = body.split(_SOURCES_MARKER, 1)[0]

    return body.strip(), confidence


def _serialize_sources(raw_sources: List[Dict[str, Any]]) -> List[Source]:
    out: List[Source] = []
    for s in raw_sources:
        if not isinstance(s, dict):
            continue
        out.append(
            Source(
                source_file=str(s.get("source_file", "unknown")),
                score=float(s.get("score", 0.0) or 0.0),
                rerank_score=(float(s["rerank_score"]) if s.get("rerank_score") is not None else None),
                context=str(s.get("context", "") or ""),
                ray_peat_response=str(s.get("ray_peat_response", "") or ""),
            )
        )
    return out


# --- Full-document serving -----------------------------------------------------
# Base dir holding the cleaned source texts. In the container these are copied
# to /app/data/processed/ai_cleaned; locally it resolves under the repo root.
_DOC_BASE = Path(os.getenv("DOC_BASE_PATH", "data/processed/ai_cleaned")).resolve()


@app.get("/api/document")
async def get_document(file: str) -> Dict[str, str]:
    """Return the full text of a source file.

    `file` is the `source_file` from a search result (Windows-style backslashes
    are normalized). The resolved path is constrained to _DOC_BASE to prevent
    path-traversal.
    """
    rel = file.replace("\\", "/").lstrip("/")
    target = (_DOC_BASE / rel).resolve()
    try:
        target.relative_to(_DOC_BASE)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Full document not available for this source.")
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not read document: {e}") from e
    return {"file": file, "content": content}


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness + readiness. Reports whether the RAG engine is wired up."""
    rag = get_rag()
    ready = bool(getattr(rag, "search_engine", None)) and bool(getattr(rag, "api_key", None))
    return {"status": "ok", "rag_ready": ready}


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request, response: Response) -> AskResponse:
    # Admins (valid X-Admin-Token) skip the daily caps entirely. Everyone else
    # is metered BEFORE spending any money on the LLM call.
    if is_admin(request):
        response.headers["X-RateLimit-Limit"] = "unlimited"
        response.headers["X-RateLimit-Remaining"] = "unlimited"
    else:
        remaining = enforce_rate_limit(client_ip(request))
        response.headers["X-RateLimit-Limit"] = str(DAILY_LIMIT_PER_IP)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

    rag = get_rag()
    history = [t.model_dump() for t in req.chat_history] if req.chat_history else None

    try:
        # get_rag_response is synchronous (it manages its own event loop
        # internally), so run it off the main loop to stay non-blocking.
        raw = await run_in_threadpool(
            rag.get_rag_response,
            req.query,
            None,            # user_profile — not used by the public site
            history,
            req.max_sources,
        )
    except Exception as e:  # noqa: BLE001 — surface a clean 500 to the client
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail=f"RAG request failed: {e}") from e

    answer, confidence = _split_answer(raw)
    raw_sources = list(getattr(rag, "_last_sources", []) or [])
    sources = _serialize_sources(raw_sources)

    # Suggest follow-up questions — only for grounded answers (have sources and
    # not an abstain). Skipped otherwise: nothing useful to ask next, and we
    # avoid a wasted LLM call. Runs off the main loop; never fails the request.
    followups: List[str] = []
    if raw_sources and (confidence or "").upper() != "ABSTAIN":
        try:
            from peatlearn.rag.followups import suggest_followups
            followups = await run_in_threadpool(
                suggest_followups, req.query, answer, getattr(rag, "api_key", None)
            )
        except Exception:  # noqa: BLE001 — follow-ups are best-effort
            logger.warning("Follow-up suggestion failed", exc_info=True)

    return AskResponse(
        answer=answer, sources=sources, confidence=confidence, followups=followups
    )
