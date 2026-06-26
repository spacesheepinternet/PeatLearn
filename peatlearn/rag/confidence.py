"""
Multi-signal confidence scorer for RAG retrieval quality.

Scores every retrieval using three signals and assigns a confidence tier:
HIGH / MEDIUM / LOW / ABSTAIN. On ABSTAIN the RAG pipeline short-circuits
to a templated refusal — no LLM call, no generated answer.

Signals:
    1. Cross-encoder top-1 score (rerank_score logits)
    2. Top-k agreement: count of candidates with rerank_score > threshold
    3. Cosine-rerank divergence: avg Pinecone cosine of top-5 candidates
       — low cosine + weak rerank = topic not in corpus → ABSTAIN

Note: HyDE-variant agreement was removed when HyDE was disabled (commit
057580e). The cosine signal now serves as the out-of-distribution detector.

Cost: ~5 ms. No additional LLM or API calls.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

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

# ms-marco-MiniLM scores on health/dietary queries sit ~1.5–2 logits lower
# than on web queries (the model's training domain). The ladder is shifted
# down accordingly so that valid Ray Peat retrievals don't flood the LOW tier.
RERANK_ABSTAIN = -3.0     # top-1 below this → definitely no match
RERANK_LOW = -1.0         # top-1 below this → weak match
RERANK_MEDIUM = 1.5       # top-1 below this → moderate match
STRONG_CANDIDATE_THR = -0.5  # rerank score above this = "strong"
STRONG_MIN_LOW = 2        # fewer strong → LOW
STRONG_MIN_MEDIUM = 4     # fewer strong → MEDIUM

# Cosine-rerank divergence: when the cross-encoder says "no match" but
# Pinecone cosine says "topic vocabulary is present", the question likely
# uses an inverted framing ("Peat endorsed X" when he actually opposed X).
# In that case, route to LOW instead of ABSTAIN so the LLM can read the
# sources and correct the premise.
COSINE_PRESENT_THR = 0.45   # avg Pinecone cosine of top-5 above this → topic is present
COSINE_DIVERGENCE_N = 5     # number of top candidates to average
# When the cross-encoder scores LOW due to domain mismatch, but the Pinecone
# cosine (Gemini embedding-001 or fine-tuned EmbeddingGemma, depending on
# active index) is strong, the topic IS genuinely present. Promote LOW →
# MEDIUM — the corpus embedding is more domain-appropriate than ms-marco-MiniLM.
COSINE_PROMOTE_THR = 0.70   # avg cosine above this → promote LOW to MEDIUM


# --- Entity grounding check -----------------------------------------------
# Broad vocabulary that is NOT diagnostic for entity grounding. The goal
# is to strip everything except domain-specific substance/entity names
# (berberine, psilocybin, taurine, progesterone, etc.) whose absence
# from the corpus means Peat never discussed them.
_ENTITY_STOP = {
    # --- English function words ---
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "on",
    "for", "with", "as", "by", "that", "this", "are", "be", "at", "from",
    "about", "into", "over", "under", "than", "then", "but", "if", "so",
    "not", "no", "can", "do", "does", "did", "will", "would", "could",
    "should", "has", "have", "had", "was", "were", "been", "being",
    "its", "his", "her", "their", "my", "your", "our", "what", "which",
    "who", "how", "why", "when", "where", "there", "here", "more", "most",
    "some", "any", "all", "each", "every", "both", "few", "many", "much",
    "such", "very", "just", "also", "too", "only", "own", "same", "other",
    "another", "like", "these", "those", "them", "they", "between", "through",
    # --- Query framing ---
    "say", "says", "said", "think", "thinks", "thought",
    "recommend", "recommends", "recommended", "recommendation", "recommendations",
    "view", "views", "opinion", "opinions", "believe", "believes",
    "advocate", "advocates", "advocated",
    "favor", "favors", "favored", "prefer", "prefers", "preference", "preferences",
    "approach", "stance", "position", "perspective", "perspectives",
    "limit", "limits", "limitation", "limitations",
    "uncertainty", "uncertainties",
    "criticism", "criticisms", "critique", "critiques",
    "concern", "concerns", "caveat", "caveats",
    "drawback", "drawbacks", "downside", "downsides",
    "advantage", "advantages", "disadvantage", "disadvantages",
    "controversy", "controversies",
    "suggest", "suggests", "argue", "argues", "argued",
    "explain", "explains", "describe", "describes",
    "design", "designed", "warn", "warns", "warned",
    "true", "accurate", "right", "correct", "according", "regarding",
    # --- Peat-specific ---
    "peat", "peats", "ray",
    # --- Common verbs ---
    "take", "taking", "taken", "give", "giving", "given",
    "make", "makes", "making", "made",
    "cause", "causes", "causing", "caused",
    "help", "helps", "helping",
    "improve", "improves", "improving",
    "reduce", "reduces", "reducing",
    "increase", "increases", "increasing",
    "prevent", "prevents", "preventing",
    "manage", "manages", "managing",
    "treat", "treats", "treating", "treated",
    "consume", "consumes", "consuming", "consumption",
    "deplete", "depletes", "depleting",
    "damage", "damages", "damaging",
    "affect", "affects", "affecting",
    "promote", "promotes", "promoting",
    "support", "supports", "supporting",
    "protect", "protects", "protecting",
    "include", "includes", "including",
    "need", "needs", "needed",
    "lead", "leads", "leading",
    "link", "links", "linked",
    "play", "plays", "playing",
    "work", "works", "working",
    "produce", "produces", "producing",
    "maintain", "maintains",
    "eat", "eats", "eating",
    "drink", "drinks", "drinking",
    "use", "uses", "using", "used",
    "cite", "cited", "citing",
    "optimize", "optimizing",
    # --- Common adjectives/modifiers ---
    "good", "bad", "best", "better", "worse", "worst",
    "high", "higher", "highest", "low", "lower", "lowest",
    "strict", "optimal", "daily", "primary",
    "safe", "safer", "safest", "dangerous", "harmful",
    "important", "essential", "necessary",
    "specific", "particular", "general", "overall",
    "effective", "natural", "normal",
    "different", "similar", "certain", "various",
    "major", "minor", "main",
    "long", "short", "full",
    "real", "really", "actually",
    "ever", "never", "always", "often", "sometimes",
    # --- Generic health/science vocabulary ---
    "health", "healthy", "body", "bodies",
    "diet", "diets", "dietary", "nutrition", "nutritional",
    "food", "foods", "meal", "meals",
    "supplement", "supplements", "supplementation",
    "drug", "drugs", "medication", "medications",
    "treatment", "therapy", "therapeutic",
    "disease", "diseases", "condition", "conditions", "disorder",
    "symptom", "symptoms",
    "level", "levels", "rate", "rates",
    "dose", "doses", "dosage", "amount", "amounts",
    "effect", "effects", "benefit", "benefits", "risk", "risks",
    "factor", "factors", "role", "roles",
    "function", "functions", "system", "process",
    "source", "sources", "study", "studies", "research",
    "people", "person", "women", "men", "patient", "patients",
    "weight", "loss", "gain",
    "blood", "pressure",
    "sugar", "salt", "water",
    "energy", "cell", "cells",
    "glass", "glasses", "cup", "cups",
    "gland", "glands", "organ", "organs", "tissue", "tissues",
    "nutrient", "nutrients",
    "protocol", "program", "method",
    "digestive", "cardiovascular", "respiratory",
    "management", "regulation", "production", "absorption",
    "instead", "rather", "whether",
    "longevity", "aging",
    "fitness", "training", "exercise", "exercises",
    "cardio", "aerobic",
    "hydration", "counting",
    "evidence", "claim", "claims",
}
_MIN_ENTITY_LEN = 4


def _term_found(term: str, text: str) -> bool:
    """Check if a term or its stem appears in text (case-insensitive)."""
    if term in text:
        return True
    if len(term) > 6:
        stem = term[:len(term) - 3]
        if stem in text:
            return True
    # Hyphenated terms: check if any component appears
    if "-" in term:
        parts = [p for p in term.split("-") if len(p) >= _MIN_ENTITY_LEN]
        if any(p in text or (len(p) > 6 and p[:len(p) - 3] in text) for p in parts):
            return True
    return False


def check_entity_grounding(
    query: str,
    candidates: List[Dict],
) -> Tuple[List[str], bool]:
    """Check if the query's key entities appear in retrieved source texts.

    Returns:
        (missing_entities, should_downgrade, all_missing) — should_downgrade is
        True when more than half of the query's key entities are absent from all
        sources; all_missing is True when EVERY key entity is absent (the topic
        is entirely outside the retrieved corpus, so any answer is fabricated).
    """
    # Split hyphenated/possessive forms into components
    raw_words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", query.lower())
    words = []
    for w in raw_words:
        parts = re.split(r"[-']", w)
        words.extend(p for p in parts if p)

    key_terms = [
        w for w in words
        if w not in _ENTITY_STOP and len(w) >= _MIN_ENTITY_LEN
    ]

    if not key_terms:
        return [], False, False

    corpus_text = " ".join(
        f"{c.get('context', '')} {c.get('ray_peat_response', '')}"
        for c in candidates
    ).lower()

    missing = [t for t in key_terms if not _term_found(t, corpus_text)]

    # Only flag when the MAJORITY of key terms are absent — prevents
    # false positives when a few vocabulary mismatches exist but the
    # core topic IS in the corpus.
    should_downgrade = len(missing) > len(key_terms) * 0.5
    # EVERY key term absent: the question's subject is entirely outside the
    # retrieved sources, so any answer is fabricated by analogy.
    all_missing = len(missing) == len(key_terms)

    if missing:
        logger.info(
            f"Entity grounding: {len(missing)}/{len(key_terms)} key terms "
            f"missing from sources: {missing}. Downgrade: {should_downgrade}, "
            f"all_missing: {all_missing}"
        )

    return missing, should_downgrade, all_missing


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
    # LOW / ABSTAIN: marginal retrieval zone (RERANK_ABSTAIN ≤ score < RERANK_LOW
    # or too few strong candidates). Use cosine as the tie-breaker:
    #   weak rerank + weak cosine = topic not in corpus → ABSTAIN
    #   weak rerank + strong cosine = domain mismatch or inverted framing → promote
    elif top_rerank < RERANK_LOW or strong_count < STRONG_MIN_LOW:
        if top_rerank < RERANK_LOW and avg_cosine < COSINE_PRESENT_THR:
            # Both signals agree: nothing relevant here → ABSTAIN
            tier = "ABSTAIN"
            reasons.append(
                f"Weak rerank ({top_rerank:.2f}) and weak cosine ({avg_cosine:.2f}) "
                f"— query likely out-of-distribution for the corpus"
            )
        elif avg_cosine >= COSINE_PROMOTE_THR:
            # Cross-encoder undershoots on health/bio domain text; cosine says
            # the topic IS present — promote to MEDIUM
            tier = "MEDIUM"
            reasons.append(
                f"Cross-encoder score ({top_rerank:.2f}) below LOW threshold, "
                f"but Pinecone cosine ({avg_cosine:.2f}) is strong — "
                f"domain mismatch likely; promoted to MEDIUM"
            )
        else:
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
