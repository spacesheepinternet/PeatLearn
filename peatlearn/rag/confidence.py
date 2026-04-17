"""
Multi-signal confidence scorer for RAG retrieval quality.

Scores every retrieval using three orthogonal signals and assigns a
confidence tier: HIGH / MEDIUM / LOW / ABSTAIN. On ABSTAIN the RAG
pipeline short-circuits to a templated refusal — no LLM call, no
generated answer.

Signals:
    1. Cross-encoder top-1 score (ms-marco-MiniLM logits, range ~-10 to +10)
    2. Top-k agreement: count of candidates with rerank_score > +1
    3. HyDE-variant agreement: cosine between academic-HyDE and email-HyDE
       embeddings — low agreement means the query is out-of-distribution

Cost: ~5 ms. No additional LLM or API calls — all signals already exist
in the pipeline by the time this runs.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceReport:
    """Result of retrieval confidence scoring."""
    tier: Literal["HIGH", "MEDIUM", "LOW", "ABSTAIN"]
    reasons: List[str] = field(default_factory=list)
    top_rerank: float = 0.0
    strong_candidate_count: int = 0
    hyde_agreement: Optional[float] = None
    avg_cosine: float = 0.0


# --- Thresholds -------------------------------------------------------
# Tuned on: Category F (edge-nuanced) must stay HIGH/MEDIUM;
#           Category H (adversarial) must become ABSTAIN/LOW.
# These are starting points — recalibrate after each defence phase.

RERANK_ABSTAIN = -1.5     # top-1 below this → definitely no match
RERANK_LOW = 0.0          # top-1 below this → weak match
RERANK_MEDIUM = 2.0       # top-1 below this → moderate match
STRONG_CANDIDATE_THR = 1.0  # rerank score above this = "strong"
STRONG_MIN_LOW = 2        # fewer strong → LOW
STRONG_MIN_MEDIUM = 4     # fewer strong → MEDIUM
HYDE_DIVERGENCE_THR = 0.55  # HyDE cosine below this → out-of-distribution

# Cosine-rerank divergence: when the cross-encoder says "no match" but
# Pinecone cosine says "topic vocabulary is present", the question likely
# uses an inverted framing ("Peat endorsed X" when he actually opposed X).
# In that case, route to LOW instead of ABSTAIN so the LLM can read the
# sources and correct the premise.
COSINE_PRESENT_THR = 0.45   # avg Pinecone cosine of top-5 above this → topic is present
COSINE_DIVERGENCE_N = 5     # number of top candidates to average


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_retrieval(
    candidates: List[Dict],
    academic_hyde_embedding: Optional[List[float]] = None,
    email_hyde_embedding: Optional[List[float]] = None,
) -> ConfidenceReport:
    """Score retrieval confidence from reranked candidates and HyDE embeddings.

    Call this immediately after the cross-encoder reranker and before the
    MMR diversity loop. The ``candidates`` list must already have a
    ``rerank_score`` key on every entry (set by ``peatlearn.rag.reranker``).

    Args:
        candidates: Reranked candidate list (must have ``rerank_score``).
        academic_hyde_embedding: Embedding of the academic-style HyDE answer,
            or None if academic HyDE failed/was skipped.
        email_hyde_embedding: Embedding of the email-style HyDE answer,
            or None if email HyDE failed/was skipped.

    Returns:
        ConfidenceReport with tier, reasons, and raw signal values.
    """
    if not candidates:
        return ConfidenceReport(
            tier="ABSTAIN",
            reasons=["No candidates returned from retrieval"],
            top_rerank=-10.0,
        )

    # --- Signal 1: Cross-encoder top-1 score ---
    top_rerank = candidates[0].get("rerank_score", -10.0)

    # --- Signal 2: Top-k agreement ---
    strong_count = sum(
        1 for c in candidates
        if c.get("rerank_score", 0.0) > STRONG_CANDIDATE_THR
    )

    # --- Signal 3: HyDE agreement (only when both embeddings exist) ---
    hyde_agreement: Optional[float] = None
    if academic_hyde_embedding and email_hyde_embedding:
        hyde_agreement = _cosine_similarity(
            academic_hyde_embedding, email_hyde_embedding
        )

    # --- Signal 4: Cosine-rerank divergence ---
    # Average Pinecone cosine of top-N candidates. High cosine + low rerank
    # signals "topic present but framing is inverted" (e.g., "Peat endorsed X"
    # when corpus says "Peat opposed X"). The cross-encoder reads semantic
    # direction, so inversions score negative — but the topic IS in the corpus.
    top_n = candidates[:COSINE_DIVERGENCE_N]
    avg_cosine = (
        sum(c.get("score", 0.0) for c in top_n) / len(top_n)
        if top_n else 0.0
    )

    # --- Decision logic ---
    reasons: List[str] = []

    # ABSTAIN: clear absence of relevant material
    if top_rerank < RERANK_ABSTAIN:
        # Before committing to ABSTAIN, check for cosine-rerank divergence.
        # High cosine means topic vocabulary is present — the query likely
        # uses an inverted framing. Route to LOW so the LLM can read the
        # sources and correct the false premise.
        if avg_cosine > COSINE_PRESENT_THR:
            tier = "LOW"
            reasons.append(
                f"Cross-encoder score ({top_rerank:.2f}) suggests no match, "
                f"but Pinecone cosine ({avg_cosine:.2f}) indicates the topic "
                f"is present — likely inverted framing. Routing to LOW so "
                f"the LLM can correct the premise."
            )
        else:
            tier = "ABSTAIN"
            reasons.append(
                f"Top rerank score ({top_rerank:.2f}) far below threshold "
                f"({RERANK_ABSTAIN}) — no relevant passages found"
            )
    elif (
        top_rerank < RERANK_LOW
        and hyde_agreement is not None
        and hyde_agreement < HYDE_DIVERGENCE_THR
    ):
        # Same divergence check for the HyDE-triggered ABSTAIN path
        if avg_cosine > COSINE_PRESENT_THR:
            tier = "LOW"
            reasons.append(
                f"Weak rerank ({top_rerank:.2f}) + HyDE divergence "
                f"({hyde_agreement:.2f}), but cosine ({avg_cosine:.2f}) "
                f"shows topic is present — possible inverted framing"
            )
        else:
            tier = "ABSTAIN"
            reasons.append(
                f"Weak retrieval (top rerank {top_rerank:.2f}) combined with "
                f"HyDE divergence ({hyde_agreement:.2f}) — query likely "
                f"out-of-distribution for the corpus"
            )
    # LOW: marginal retrieval
    elif top_rerank < RERANK_LOW or strong_count < STRONG_MIN_LOW:
        tier = "LOW"
        if top_rerank < RERANK_LOW:
            reasons.append(
                f"Top rerank score ({top_rerank:.2f}) below zero — weak relevance"
            )
        if strong_count < STRONG_MIN_LOW:
            reasons.append(
                f"Only {strong_count} strong candidate(s) (rerank > "
                f"{STRONG_CANDIDATE_THR})"
            )
    # MEDIUM: decent but not rock-solid
    elif top_rerank < RERANK_MEDIUM or strong_count < STRONG_MIN_MEDIUM:
        tier = "MEDIUM"
        if top_rerank < RERANK_MEDIUM:
            reasons.append(f"Top rerank score ({top_rerank:.2f}) moderate")
        if strong_count < STRONG_MIN_MEDIUM:
            reasons.append(
                f"{strong_count} strong candidate(s) — below the "
                f"{STRONG_MIN_MEDIUM} threshold for HIGH"
            )
    # HIGH: strong retrieval across the board
    else:
        tier = "HIGH"
        reasons.append(
            f"Strong retrieval: top rerank {top_rerank:.2f}, "
            f"{strong_count} strong candidates"
        )

    # Supplemental warning: HyDE divergence on a non-ABSTAIN answer
    if (
        hyde_agreement is not None
        and hyde_agreement < HYDE_DIVERGENCE_THR
        and tier != "ABSTAIN"
    ):
        reasons.append(
            f"HyDE embeddings diverge ({hyde_agreement:.2f}) — possible "
            f"out-of-distribution query"
        )

    # Keyword-fallback downgrade: if the cross-encoder wasn't available and
    # reranking used keyword overlap, the scores aren't calibrated for our
    # thresholds. Downgrade by one tier as a safety margin.
    if candidates and candidates[0].get("_keyword_fallback"):
        _DOWNGRADE = {"HIGH": "MEDIUM", "MEDIUM": "LOW"}
        if tier in _DOWNGRADE:
            old_tier = tier
            tier = _DOWNGRADE[tier]
            reasons.append(
                f"Downgraded {old_tier} → {tier}: cross-encoder unavailable, "
                f"keyword-overlap scores are not calibrated for abstention"
            )

    report = ConfidenceReport(
        tier=tier,
        reasons=reasons,
        top_rerank=round(top_rerank, 3),
        strong_candidate_count=strong_count,
        hyde_agreement=round(hyde_agreement, 3) if hyde_agreement is not None else None,
        avg_cosine=round(avg_cosine, 3),
    )
    logger.info(
        f"Confidence: {report.tier} | top_rerank={report.top_rerank} "
        f"avg_cosine={report.avg_cosine} strong={report.strong_candidate_count} "
        f"hyde_agree={report.hyde_agreement}"
    )
    return report
