"""
History-aware query rewriting (conversational memory for retrieval).

RAG retrieval matches the *current* query against the corpus. A follow-up like
"what about the dosage?" or "how much?" or "and for women?" has no standalone
meaning, so it retrieves the wrong passages and the chatbot appears to have no
memory of the conversation.

This module rewrites such follow-ups into a self-contained query using the
recent turns, so retrieval fetches the right sources. It only rewrites the
*search query* — the answer is still generated from freshly retrieved sources,
so a hallucination in an earlier turn is NOT injected as fact into the next
answer (the reason prior assistant text is kept out of the generation prompt).

Cheap and conservative:
  - skips the LLM call entirely for self-contained queries (no pronouns /
    follow-up markers and not ultra-short);
  - one gemini-2.5-flash-lite call otherwise;
  - fails OPEN — any error or missing key returns the original query unchanged.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

REWRITE_MODEL = "gemini-2.5-flash-lite"
REWRITE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{REWRITE_MODEL}:generateContent"
)

# Deictic / reference tokens that usually point back at earlier turns.
_PRONOUNS = {
    "it", "its", "it's", "that", "this", "they", "them", "those", "these",
    "he", "him", "his", "she", "her", "one", "ones", "there",
}
# Multi-word follow-up phrasings.
_MARKERS = (
    "what about", "how about", "why not", "tell me more", "more about",
    "any more", "anything else", "what else", "and what", "go on",
    "elaborate", "expand", "the same", "that one", "this one", "instead",
    "compared to", "versus", " vs ", "difference between", "as well",
    "on that", "about it", "about that", "of it", "of that",
)


def _needs_context(query: str) -> bool:
    """True if the query looks like it depends on earlier turns."""
    q = (query or "").lower().strip()
    if not q:
        return False
    words = q.split()
    if len(words) <= 3:
        return True
    if any(m in q for m in _MARKERS):
        return True
    if any(w.strip(".,!?;:") in _PRONOUNS for w in words):
        return True
    # Opens with a conjunction ("and thyroid?", "but why?")
    if words[0] in {"and", "but", "so", "or", "also", "then", "plus"}:
        return True
    return False


def _build_transcript(chat_history: List[Dict[str, Any]]) -> str:
    """Last few turns as a compact transcript (assistant text truncated)."""
    turns = chat_history[-6:]
    lines = []
    for m in turns:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content[:300]}")
        elif role == "assistant":
            lines.append(f"Assistant: {content[:300]}")
    return "\n".join(lines)


_PROMPT = """You rewrite a user's latest message into a single, self-contained search query for a Ray Peat health Q&A system.

Using the conversation, resolve any references ("it", "that", "he", "the dosage", "for women", "how much", etc.) so the query names its subject explicitly. Pull the topic from the conversation into the query even when the latest message is grammatically complete but relies on context (e.g. "how much should I take?" -> "how much <subject> should I take?"). Do NOT add new facts, change the topic, or answer the question.

Example:
  Conversation: User: What did Peat think about coffee? / Assistant: He viewed it as protective...
  Latest message: how much per day?
  Rewrite: how much coffee per day did Ray Peat recommend?

Conversation:
{transcript}

Latest message: {query}

Rewritten standalone query (one line, no quotes):"""


def contextualize(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Return a standalone version of ``query`` given recent turns.

    Returns the original query unchanged when there's no history, when the query
    is already self-contained, or on any failure.
    """
    if not query or not chat_history:
        return query
    if not _needs_context(query):
        return query
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return query

    transcript = _build_transcript(chat_history)
    if not transcript:
        return query

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": _PROMPT.format(
                transcript=transcript, query=query
            )}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 80,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        resp = requests.post(REWRITE_URL, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Query contextualizer API error {resp.status_code}")
            return query
        _j = resp.json()
        text = _j["candidates"][0]["content"]["parts"][0]["text"].strip()
        try:
            from peatlearn.rag import cost_logger as _cl
            _cl.record_gemini("contextualize", REWRITE_MODEL, _j.get("usageMetadata"))
        except Exception:
            pass
        # Clean up: single line, strip wrapping quotes.
        text = text.splitlines()[0].strip().strip('"').strip("'").strip()
        text = re.sub(r"\s+", " ", text)
        # Sanity: keep the rewrite only if it's plausible; else fall back.
        if not text or len(text) > 300:
            return query
        logger.info(f"Contextualized query: {query!r} -> {text!r}")
        return text
    except Exception as e:
        logger.warning(f"Query contextualizer failed: {e}")
        return query
