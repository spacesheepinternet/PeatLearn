"""
Cross-encoder reranker for RAG candidate passages.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to score (query, passage) pairs
jointly — far more accurate than keyword overlap because the model reads
both the query and the passage together before scoring.

Falls back to keyword-overlap scoring automatically if the model fails to load
(no API keys needed, no network required once downloaded).
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused across all requests
_model = None
_model_load_attempted = False


_STOP = {
    "the","a","an","and","or","of","to","in","is","it","on","for","with","as","by",
    "that","this","are","be","at","from","about","into","over","under","than","then",
    "but","if","so","not",
}


def _get_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
        logger.info("Cross-encoder reranker loaded: ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        logger.warning(f"Cross-encoder unavailable — falling back to keyword overlap: {e}")
        _model = None
    return _model


def _warmup():
    """Load the cross-encoder in a background thread at import time.

    This ensures the model is ready before the first user query instead of
    causing a ~25s cold-start delay on the first request.
    """
    import threading
    t = threading.Thread(target=_get_model, daemon=True, name="reranker-warmup")
    t.start()


# Kick off warmup immediately when the module is imported
_warmup()


def _keyword_score(query: str, candidate: Dict[str, Any]) -> float:
    """Keyword overlap fallback: 0.7 × vector_sim + 0.3 × overlap."""
    def tok(t: str) -> List[str]:
        return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", t.lower()) if w not in _STOP]

    q_vocab = set(tok(query)) or set(re.findall(r"[a-zA-Z]+", query.lower()))
    text = f"{candidate.get('context', '')} {candidate.get('ray_peat_response', '')}"
    words = tok(text)
    overlap = len(q_vocab.intersection(words)) / max(1, len(q_vocab)) if words else 0.0
    return 0.7 * float(candidate.get("score", 0.0)) + 0.3 * overlap


def rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score and sort candidates using cross-encoder, falling back to keyword overlap.

    Adds a 'rerank_score' key to every candidate dict and returns the list
    sorted descending by that score. Does NOT mutate the originals — works on
    a shallow copy so callers can inspect both raw and reranked scores.

    Args:
        query: The user's query (or resolved/HyDE query).
        candidates: List of dicts with at least 'context', 'ray_peat_response',
                    and 'score' (Pinecone cosine similarity) keys.

    Returns:
        Same list, sorted by rerank_score descending.
    """
    if not candidates:
        return candidates

    model = _get_model()

    if model is not None:
        # Build (query, passage) pairs — truncate to keep within 512-token limit
        passages = [
            (
                f"{c.get('context', '')[:200]} "
                f"{c.get('ray_peat_response', '')[:400]}"
            ).strip()
            for c in candidates
        ]
        pairs = [[query, p] for p in passages]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
            result = []
            for c, score in zip(candidates, scores):
                entry = dict(c)
                entry["rerank_score"] = float(score)
                result.append(entry)
            result.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.debug(
                f"Cross-encoder reranked {len(result)} candidates. "
                f"Top score: {result[0]['rerank_score']:.3f}"
            )
            return result
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e} — using keyword fallback")

    # Keyword overlap fallback
    result = []
    for c in candidates:
        entry = dict(c)
        entry["rerank_score"] = _keyword_score(query, c)
        result.append(entry)
    result.sort(key=lambda x: x["rerank_score"], reverse=True)
    return result


def is_cross_encoder_available() -> bool:
    """Return True if the cross-encoder model loaded successfully."""
    return _get_model() is not None
