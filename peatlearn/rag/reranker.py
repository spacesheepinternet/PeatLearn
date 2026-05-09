"""
Reranker for RAG candidate passages.

Priority chain:
  1. Cohere rerank-v4-pro via OpenRouter (best domain quality, free, ~300ms)
  2. Local cross-encoder: peat-reranker-ft if present, else ms-marco-MiniLM-L-6-v2
  3. Keyword-overlap fallback (no models, no API)

Score scaling:
  Cohere returns 0–1 relevance probabilities. These are scaled to the same
  logit range as MiniLM via (score - 0.5) * 10 so confidence.py thresholds
  remain valid across both paths:
    Cohere 0.65 → +1.5  (MEDIUM threshold)
    Cohere 0.40 → -1.0  (LOW threshold)
    Cohere 0.20 → -3.0  (ABSTAIN threshold)
"""

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)

# --- Cohere / OpenRouter config ---
_OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
_COHERE_MODEL = "cohere/rerank-4-pro"
_COHERE_TIMEOUT = 30  # seconds — 80-doc batches need ~20s on OpenRouter

# --- Local cross-encoder singleton ---
_model = None
_model_load_attempted = False
_load_done = threading.Event()

_STOP = {
    "the","a","an","and","or","of","to","in","is","it","on","for","with","as","by",
    "that","this","are","be","at","from","about","into","over","under","than","then",
    "but","if","so","not",
}

_FT_MODEL_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "data" / "models" / "reranker" / "peat-reranker-ft"
)


# ---------------------------------------------------------------------------
# Cohere via OpenRouter
# ---------------------------------------------------------------------------

def _cohere_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Call Cohere rerank-v4-pro via OpenRouter. Returns scored+sorted list or None on failure."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    documents = [
        f"{c.get('context', '')[:300]} {c.get('ray_peat_response', '')[:600]}".strip()
        for c in candidates
    ]

    try:
        resp = requests.post(
            _OPENROUTER_RERANK_URL,
            json={"model": _COHERE_MODEL, "query": query, "documents": documents},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=_COHERE_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"Cohere rerank returned {resp.status_code}: {resp.text[:200]}")
            return None

        results = resp.json().get("results", [])
        if not results:
            return None

        # Map results back to candidates using the returned index
        scored = []
        for r in results:
            idx = r.get("index", -1)
            raw_score = float(r.get("relevance_score", 0.0))
            # Scale 0–1 → MiniLM logit range so confidence.py thresholds stay valid
            logit_score = (raw_score - 0.5) * 10
            entry = dict(candidates[idx])
            entry["rerank_score"] = logit_score
            entry["_cohere_raw"] = raw_score
            entry["_reranker_model"] = "cohere"
            scored.append(entry)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info(
            f"Cohere reranked {len(scored)} candidates. "
            f"Top raw={scored[0]['_cohere_raw']:.3f} logit={scored[0]['rerank_score']:.2f}"
        )
        return scored

    except requests.Timeout:
        logger.warning("Cohere rerank timed out — falling back to local cross-encoder")
        return None
    except Exception as e:
        logger.warning(f"Cohere rerank failed: {e} — falling back to local cross-encoder")
        return None


# ---------------------------------------------------------------------------
# Local cross-encoder (fallback)
# ---------------------------------------------------------------------------

def _get_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        if Path(_FT_MODEL_DIR).exists():
            _model = CrossEncoder(_FT_MODEL_DIR, max_length=512)
            logger.info("Cross-encoder loaded: peat-reranker-ft (fine-tuned)")
        else:
            _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            logger.info("Cross-encoder loaded: ms-marco-MiniLM-L-6-v2 (fallback)")
    except Exception as e:
        logger.warning(f"Cross-encoder unavailable: {e}")
        _model = None
    finally:
        _load_done.set()
    return _model


def _warmup():
    t = threading.Thread(target=_get_model, daemon=True, name="reranker-warmup")
    t.start()


_warmup()


# ---------------------------------------------------------------------------
# Keyword overlap (last-resort fallback)
# ---------------------------------------------------------------------------

def _keyword_score(query: str, candidate: Dict[str, Any]) -> float:
    """0.7 × vector_sim + 0.3 × keyword overlap."""
    def tok(t: str) -> List[str]:
        return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", t.lower()) if w not in _STOP]

    q_vocab = set(tok(query)) or set(re.findall(r"[a-zA-Z]+", query.lower()))
    text = f"{candidate.get('context', '')} {candidate.get('ray_peat_response', '')}"
    words = tok(text)
    overlap = len(q_vocab.intersection(words)) / max(1, len(q_vocab)) if words else 0.0
    return 0.7 * float(candidate.get("score", 0.0)) + 0.3 * overlap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score and sort candidates. Tries Cohere → local cross-encoder → keyword overlap.

    Adds 'rerank_score' to every candidate dict. Does not mutate originals.
    """
    if not candidates:
        return candidates

    # 1. Cohere via OpenRouter (primary)
    result = _cohere_rerank(query, candidates)
    if result is not None:
        return result

    # 2. Local cross-encoder (fallback)
    _load_done.wait()
    model = _get_model()
    if model is not None:
        passages = [
            f"{c.get('context', '')[:200]} {c.get('ray_peat_response', '')[:400]}".strip()
            for c in candidates
        ]
        pairs = [[query, p] for p in passages]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
            result = []
            for c, score in zip(candidates, scores):
                entry = dict(c)
                entry["rerank_score"] = float(score)
                entry["_reranker_model"] = "cross-encoder"
                result.append(entry)
            result.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info(
                f"Local cross-encoder reranked {len(result)} candidates. "
                f"Top score: {result[0]['rerank_score']:.3f}"
            )
            return result
        except Exception as e:
            logger.warning(f"Local cross-encoder scoring failed: {e} — using keyword fallback")

    # 3. Keyword overlap (last resort)
    logger.warning("All rerankers unavailable — using keyword-overlap fallback.")
    result = []
    for c in candidates:
        entry = dict(c)
        entry["rerank_score"] = _keyword_score(query, c)
        entry["_keyword_fallback"] = True
        entry["_reranker_model"] = "keyword"
        result.append(entry)
    result.sort(key=lambda x: x["rerank_score"], reverse=True)
    return result


def is_cross_encoder_available() -> bool:
    """Return True if the local cross-encoder loaded successfully."""
    return _get_model() is not None
