"""
Follow-up question suggester.

After a grounded answer, propose a few short follow-up questions the user might
naturally ask next. Rendered in the UI as clickable buttons that submit as the
next query.

Constraints baked into the prompt:
  - stay inside Ray Peat's health / bioenergetic domain (so a click can't land
    on an out-of-domain query the domain guard would just refuse);
  - standalone phrasing (no "it"/"that") so the suggestion works as a fresh query;
  - short, so it fits on a button.

Cost: one cheap gemini-2.5-flash-lite call per grounded answer. Fails CLOSED —
any error returns an empty list, so the buttons simply don't render.
"""

import json
import logging
import os
import re
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

FOLLOWUP_MODEL = "gemini-2.5-flash-lite"
FOLLOWUP_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{FOLLOWUP_MODEL}:generateContent"
)

_PROMPT = """You suggest follow-up questions for a Q&A assistant grounded in Dr. Ray Peat's work on bioenergetic health, nutrition, hormones, and metabolism.

Given the user's question and the assistant's answer, write {n} short follow-up questions the user might naturally ask next.

Rules:
- Each must stay within Ray Peat's health/bioenergetic domain.
- Each must be standalone — no "it", "that", "this"; name the topic explicitly.
- Keep each under 12 words, phrased as a natural question.
- Make them genuinely useful next steps, not restatements of the original question.

User question:
{query}

Assistant answer:
{answer}

Return ONLY a JSON array of {n} strings — no markdown, no code fences. Example:
["What foods raise progesterone?", "How does light affect thyroid?"]"""


def _parse_questions(text: str, n: int) -> List[str]:
    """Extract a list of question strings from the model output."""
    text = text.strip()
    # Strip ```json fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except Exception:
        # Last-ditch: grab the first JSON array in the text.
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for item in data:
        if isinstance(item, str):
            q = item.strip()
            if q and len(q) <= 120:
                out.append(q)
    return out[:n]


def suggest_followups(
    query: str,
    answer: str,
    api_key: Optional[str] = None,
    n: int = 3,
) -> List[str]:
    """Return up to ``n`` short follow-up questions, or [] on any failure.

    Args:
        query: The user's original question.
        answer: The assistant's grounded answer.
        api_key: Gemini API key; falls back to GEMINI_API_KEY env var.
        n: How many suggestions to request.

    Returns:
        A list of standalone, in-domain follow-up question strings (possibly
        empty — callers should render nothing when the list is empty).
    """
    if not (query and answer):
        return []
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return []

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": _PROMPT.format(
                n=n, query=query, answer=answer[:2000]
            )}]}
        ],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        resp = requests.post(FOLLOWUP_URL, json=payload, headers=headers, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"Follow-up API error {resp.status_code}")
            return []
        _j = resp.json()
        text = _j["candidates"][0]["content"]["parts"][0]["text"]
        try:
            from peatlearn.rag import cost_logger as _cl
            _cl.record_gemini("followups", FOLLOWUP_MODEL, _j.get("usageMetadata"))
        except Exception:
            pass
        return _parse_questions(text, n)
    except Exception as e:
        logger.warning(f"Follow-up suggestion failed: {e}")
        return []
