"""
Grounding verifier for RAG answers.

Makes a second LLM pass (Gemini 2.5-flash-lite) that reads (answer, sources)
with NO query context and checks whether each claim is grounded verbatim in
the cited source. Claims that cannot be backed by a quoted span are marked
UNSUPPORTED and stripped from the answer.

This catches Gemini's tendency to extrapolate between sources — e.g., Source A
says "thyroid needs iodine" and Source B says "Peat liked coconut oil", and
Gemini writes "Peat recommended coconut oil for iodine-dependent thyroid
function" — a plausible synthesis that Peat never actually made.

Also enforces two cheap post-processing checks (no LLM call):
    1. Quote-fabrication: any substring in double-quotes must appear verbatim
       in at least one cited source.
    2. Citation-ID drift: any [Sn] where n > len(sources) triggers a flag.

Cost: +1 LLM call per query (~1-2s, ~600 input tokens on flash-lite).
Skipped on ABSTAIN answers.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

VERIFY_MODEL = "gemini-2.5-flash-lite"
VERIFY_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{VERIFY_MODEL}:generateContent"
)


@dataclass
class VerificationResult:
    """Result of grounding verification."""
    claims: List[Dict]            # all parsed claims with support status
    unsupported: List[Dict]       # claims that could not be grounded
    citation_drift: List[str]     # [Sn] references where n > len(sources)
    fabricated_quotes: List[str]  # quoted substrings not in any source
    revised_answer: str           # answer with unsupported claims stripped
    verified: bool                # True if no unsupported claims found


def _check_citation_drift(answer: str, n_sources: int) -> List[str]:
    """Find [Sn] references where n exceeds the number of sources."""
    refs = re.findall(r"\[S(\d+)\]", answer)
    return [f"[S{n}]" for n in refs if int(n) > n_sources]


def _check_fabricated_quotes(answer: str, sources_text: str) -> List[str]:
    """Find double-quoted substrings in the answer not present in sources."""
    # Match text inside double quotes (at least 5 words to avoid short phrases)
    quotes = re.findall(r'"([^"]{20,})"', answer)
    fabricated = []
    sources_lower = sources_text.lower()
    for q in quotes:
        if q.lower().strip() not in sources_lower:
            fabricated.append(q)
    return fabricated


def _strip_unsupported_claims(answer: str, unsupported_texts: List[str]) -> str:
    """Remove unsupported claim sentences from the answer."""
    if not unsupported_texts:
        return answer
    # Split into sentences and filter out unsupported ones
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    kept = []
    for sent in sentences:
        sent_lower = sent.lower().strip()
        is_unsupported = False
        for claim_text in unsupported_texts:
            # Check if this sentence contains the unsupported claim
            if claim_text.lower().strip() in sent_lower or sent_lower in claim_text.lower().strip():
                is_unsupported = True
                break
        if not is_unsupported:
            kept.append(sent)
    return " ".join(kept).strip()


def verify_claims(
    answer: str,
    sources: List[Dict],
    api_key: Optional[str] = None,
) -> VerificationResult:
    """Verify that every claim in the answer is grounded in the cited sources.

    Args:
        answer: The RAG-generated answer text (with [Sn] citations).
        sources: List of source dicts with at least 'source_file',
                 'context', and 'ray_peat_response' keys.
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.

    Returns:
        VerificationResult with claims, unsupported list, and revised answer.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    n_sources = len(sources)

    # --- Cheap checks first (no LLM call) ---
    citation_drift = _check_citation_drift(answer, n_sources)

    # Build a concatenated sources text for quote-fabrication check
    sources_text = "\n".join(
        f"{s.get('context', '')} {s.get('ray_peat_response', '')}"
        for s in sources
    )
    fabricated_quotes = _check_fabricated_quotes(answer, sources_text)

    # --- LLM verification pass ---
    # Build numbered source blocks matching the [Sn] format in the answer
    source_blocks = []
    for i, s in enumerate(sources, 1):
        source_blocks.append(
            f"[S{i}] {s.get('source_file', 'unknown')}\n"
            f"Context: {s.get('context', '')[:300]}\n"
            f"Peat's words: {s.get('ray_peat_response', '')[:500]}"
        )
    sources_for_prompt = "\n---\n".join(source_blocks)

    prompt = f"""You are a strict fact-checker. You will receive an ANSWER and the SOURCES it cites.
Your job: for each factual claim in the answer, determine if it is SUPPORTED by a verbatim
or near-verbatim span in the cited source.

IMPORTANT: You have NO access to the original question. Do NOT evaluate whether the answer
is helpful — only evaluate whether each claim is grounded in the sources.

ANSWER:
{answer}

SOURCES:
{sources_for_prompt}

Instructions:
1. Parse the answer into individual factual claims (one claim per sentence or clause).
2. For each claim, find the [Sn] citation and look up that source.
3. If you can quote a supporting span (5+ words) from the cited source, mark it SUPPORTED.
4. If the claim has no citation, or the cited source does not contain a supporting span,
   mark it UNSUPPORTED and explain why.
5. Ignore meta-statements ("Peat argued...", "In his view...") — these are attribution
   frames, not factual claims. Only check the substance of what is attributed.

Return ONLY a JSON object — no markdown, no code fences:
{{
  "claims": [
    {{
      "text": "<the claim text>",
      "citation": "<[Sn] or 'none'>",
      "status": "<SUPPORTED or UNSUPPORTED>",
      "supporting_quote": "<verbatim quote from source, or null>",
      "reason": "<why unsupported, or null>"
    }}
  ]
}}"""

    claims = []
    unsupported = []

    if not api_key:
        logger.warning("No API key for verifier — skipping LLM verification")
        return VerificationResult(
            claims=[], unsupported=[], citation_drift=citation_drift,
            fabricated_quotes=fabricated_quotes, revised_answer=answer,
            verified=False,
        )

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    text = ""
    for attempt in range(3):
        try:
            resp = requests.post(VERIFY_URL, json=payload, headers=headers, timeout=45)
            if resp.status_code == 429:
                time.sleep(10)
                continue
            if resp.status_code != 200:
                logger.warning(f"Verifier API error {resp.status_code}")
                break
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            break
        except Exception as e:
            logger.warning(f"Verifier call failed: {e}")
            time.sleep(5)

    if text:
        # Parse JSON from response
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

        claims = parsed.get("claims", [])
        unsupported = [c for c in claims if c.get("status") == "UNSUPPORTED"]

    # Build revised answer by stripping unsupported claims
    unsupported_texts = [c.get("text", "") for c in unsupported if c.get("text")]
    revised = _strip_unsupported_claims(answer, unsupported_texts)

    # Also add citation-drift and fabricated-quote entries to unsupported
    for ref in citation_drift:
        unsupported.append({
            "text": f"Citation {ref} references a non-existent source",
            "citation": ref, "status": "UNSUPPORTED",
            "reason": f"Only {n_sources} sources provided",
        })
    for q in fabricated_quotes:
        unsupported.append({
            "text": f'Fabricated quote: "{q[:80]}..."',
            "citation": "none", "status": "UNSUPPORTED",
            "reason": "Quoted text not found verbatim in any cited source",
        })

    logger.info(
        f"Verifier: {len(claims)} claims, {len(unsupported)} unsupported, "
        f"{len(citation_drift)} citation-drift, {len(fabricated_quotes)} fabricated-quotes"
    )

    return VerificationResult(
        claims=claims,
        unsupported=unsupported,
        citation_drift=citation_drift,
        fabricated_quotes=fabricated_quotes,
        revised_answer=revised if unsupported_texts else answer,
        verified=len(unsupported) == 0,
    )
