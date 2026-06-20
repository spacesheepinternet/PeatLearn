#!/usr/bin/env python3
"""
PeatLearn Web API — production HTTP service for the public website.

Wraps the live, benchmarked RAG pipeline (`peatlearn/adaptive/rag_system.py`,
the 9.64/10 path the Streamlit app uses) behind a small, stateless JSON API so a
decoupled frontend (Next.js) can call it.

This is intentionally separate from `app/api.py`, which is the older
`peatlearn/rag` path and is NOT the pipeline we want to serve publicly.

Run locally:
    uvicorn app.web_api:app --port 8080 --reload

Run in a container: see Dockerfile / docker-compose.yml.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
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


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: Optional[str] = None


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
            )
        )
    return out


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness + readiness. Reports whether the RAG engine is wired up."""
    rag = get_rag()
    ready = bool(getattr(rag, "search_engine", None)) and bool(getattr(rag, "api_key", None))
    return {"status": "ok", "rag_ready": ready}


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
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
    sources = _serialize_sources(list(getattr(rag, "_last_sources", []) or []))
    return AskResponse(answer=answer, sources=sources, confidence=confidence)
