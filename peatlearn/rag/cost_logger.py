"""Per-request, per-stage LLM cost/token logging for the RAG pipeline.

Writes one JSON line per query to ``data/cost_logs/cost_<YYYYMMDD>.jsonl`` so we
can measure the REAL per-stage cost split (embed / rerank / generate / verify)
from live traffic — instead of estimating it. This exists to settle one
question: is retrieval actually a meaningful fraction of per-query cost, or is
it dominated by generation + verification?

Design: fully non-invasive. Every public function swallows its own exceptions,
so a logging failure can never break or slow a user-facing response. State is
held in a ``contextvars.ContextVar`` so it is correct under both threads and
asyncio without threading state through call signatures.

Usage (already wired into rag_system.py / verifier.py):
    cost_logger.start(query)
    try:
        ... run the pipeline; stages call record_* ...
    finally:
        cost_logger.flush()
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cost_logs"

# --- Pricing, USD per 1M tokens (input, output). Thinking tokens bill as output.
# These are editable constants — update if Google/Cohere change pricing.
_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash":      {"in": 0.30,  "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10,  "out": 0.40},
    "gemini-2.0-flash":      {"in": 0.10,  "out": 0.40},
    "gemini-2.0-flash-lite": {"in": 0.075, "out": 0.30},
    "gemini-1.5-flash":      {"in": 0.075, "out": 0.30},
    "gemini-embedding-001":  {"in": 0.15,  "out": 0.0},
}

# Cohere rerank-4-pro via OpenRouter: ~$2.00 per 1,000 searches, where one
# "search" = one query scored against up to 100 documents.
_RERANK_COST_PER_SEARCH = 0.002

_ctx: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "rag_cost_ctx", default=None
)


def _price(model: str) -> Dict[str, float]:
    return _PRICING.get(model, {"in": 0.0, "out": 0.0})


def _bucket(ctx: Dict[str, Any], stage: str) -> Dict[str, Any]:
    stages = ctx.setdefault("stages", {})
    return stages.setdefault(
        stage,
        {"model": None, "tokens_in": 0, "tokens_out": 0, "tokens_think": 0,
         "cost_usd": 0.0, "calls": 0},
    )


def start(query: str = "") -> None:
    """Begin a fresh per-request cost accumulator. Safe to call repeatedly."""
    try:
        _ctx.set({"query": (query or "")[:200], "t0": time.time(), "stages": {}})
    except Exception:
        pass


def record_gemini(stage: str, model: str, usage: Optional[Dict[str, Any]]) -> None:
    """Record one Gemini call from its REST ``usageMetadata`` block.

    ``usage`` keys (camelCase from the REST API): promptTokenCount,
    candidatesTokenCount, thoughtsTokenCount. Missing/None usage is tolerated
    (recorded as a zero-token call so the call still counts).
    """
    try:
        ctx = _ctx.get()
        if ctx is None:
            return
        usage = usage or {}
        t_in = int(usage.get("promptTokenCount", 0) or 0)
        t_out = int(usage.get("candidatesTokenCount", 0) or 0)
        t_think = int(usage.get("thoughtsTokenCount", 0) or 0)
        p = _price(model)
        # Thinking tokens are billed at the output rate on Gemini.
        cost = (t_in * p["in"] + (t_out + t_think) * p["out"]) / 1_000_000
        b = _bucket(ctx, stage)
        b["model"] = model
        b["tokens_in"] += t_in
        b["tokens_out"] += t_out
        b["tokens_think"] += t_think
        b["cost_usd"] += cost
        b["calls"] += 1
    except Exception:
        pass


def record_embedding(stage: str, model: str, text: str) -> None:
    """Record an embedding call. Tokens are estimated from text length
    (~4 chars/token) since the embed endpoint does not return usage."""
    try:
        ctx = _ctx.get()
        if ctx is None:
            return
        est_tokens = max(1, len(text or "") // 4)
        p = _price(model)
        cost = est_tokens * p["in"] / 1_000_000
        b = _bucket(ctx, stage)
        b["model"] = model
        b["tokens_in"] += est_tokens
        b["cost_usd"] += cost
        b["calls"] += 1
    except Exception:
        pass


def record_rerank(stage: str, model: str, n_docs: int) -> None:
    """Record a rerank pass. Only the Cohere/OpenRouter path costs money;
    local cross-encoder and keyword fallback are free (cost 0)."""
    try:
        ctx = _ctx.get()
        if ctx is None:
            return
        cost = _RERANK_COST_PER_SEARCH if model == "cohere" else 0.0
        b = _bucket(ctx, stage)
        b["model"] = model
        b["tokens_in"] += int(n_docs or 0)  # store n_docs in tokens_in for reference
        b["cost_usd"] += cost
        b["calls"] += 1
    except Exception:
        pass


def flush() -> None:
    """Write the accumulated per-query record as one JSON line, then reset."""
    try:
        ctx = _ctx.get()
        if ctx is None:
            return
        _ctx.set(None)
        stages = ctx.get("stages", {})
        total = round(sum(s["cost_usd"] for s in stages.values()), 6)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": ctx.get("query", ""),
            "elapsed_s": round(time.time() - ctx.get("t0", time.time()), 3),
            "total_usd": total,
            "stages": {
                name: {
                    "model": s["model"],
                    "tokens_in": s["tokens_in"],
                    "tokens_out": s["tokens_out"],
                    "tokens_think": s["tokens_think"],
                    "cost_usd": round(s["cost_usd"], 6),
                    "calls": s["calls"],
                }
                for name, s in stages.items()
            },
        }
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f"cost_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "RAG cost $%.5f | %s",
            total,
            " ".join(f"{n}=${s['cost_usd']:.5f}" for n, s in record["stages"].items()),
        )
    except Exception:
        # Never let logging failures surface to the request path.
        try:
            _ctx.set(None)
        except Exception:
            pass
