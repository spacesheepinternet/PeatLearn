#!/usr/bin/env python3
"""
Lightweight, explainable personalization utilities that work without user logs:
- Difficulty scoring for content passages
- Content-based recommendations with diversification
- Retrieval-based quiz question generation from RAG results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import math
import re

import numpy as np


@dataclass
class Passage:
    id: str
    context: str
    source_file: str
    similarity_score: float


TECH_TERMS = set([
    # Domain-relevant terms; extend as needed
    "thyroid", "t3", "t4", "estrogen", "progesterone", "cortisol",
    "metabolism", "oxidation", "mitochondria", "glycolysis", "serotonin",
    "dopamine", "gaba", "insulin", "glycogen", "lactate", "co2",
    "aspirin", "vitamin", "saturated", "polyunsaturated", "pufa",
])


def estimate_difficulty_score(text: str) -> float:
    """Estimate difficulty in [0,1] from readability + technical density.

    Heuristics:
    - Longer sentences and higher type-token ratio → harder
    - Higher density of domain terms → harder
    """
    if not text:
        return 0.5

    # Basic sentence tokenization
    sentences = re.split(r"[.!?]+\s+", text)
    sentences = [s for s in sentences if s]
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    num_words = len(words) or 1

    # Sentence length features
    avg_sent_len = num_words / max(1, len(sentences))
    sent_len_penalty = min(1.0, avg_sent_len / 30.0)  # 30 words ~ upper comfortable

    # Lexical richness
    vocab = set(words)
    type_token_ratio = len(vocab) / num_words
    ttr_component = min(1.0, type_token_ratio * 3.0)  # scale

    # Technical term density
    tech_hits = sum(1 for w in words if w in TECH_TERMS)
    tech_density = tech_hits / num_words
    tech_component = min(1.0, tech_density * 10.0)

    # Combine
    difficulty = 0.4 * sent_len_penalty + 0.3 * ttr_component + 0.3 * tech_component
    return float(max(0.0, min(1.0, difficulty)))


def mmr_diversify(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    lambda_mult: float = 0.7,
    top_k: int = 10,
) -> List[int]:
    """Maximal Marginal Relevance for diversification.

    Args:
        query_vec: [D]
        candidate_vecs: [N, D]
    Returns: indices of selected items
    """
    if candidate_vecs.size == 0:
        return []

    # Normalize
    def normalize(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
        return x / n

    q = normalize(query_vec)
    X = normalize(candidate_vecs)

    # Similarities
    rel = X @ q

    selected: List[int] = []
    candidates = list(range(X.shape[0]))
    while len(selected) < min(top_k, X.shape[0]):
        best_idx = None
        best_score = -1e9
        for i in candidates:
            # Diversity penalty vs. already selected
            div = 0.0
            if selected:
                div = max((X[i] @ X[j] for j in selected), default=0.0)
            score = lambda_mult * rel[i] - (1 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)  # type: ignore[arg-type]
        candidates.remove(best_idx)  # type: ignore[arg-type]
    return selected


def generate_mcq_from_passage(
    topic: str,
    passage_text: str,
    num_options: int = 4
) -> Dict[str, Any]:
    """Heuristic MCQ grounded on the passage (no LLM).

    Strategy:
    - Extract a central sentence from the passage
    - Make the stem reference the excerpt directly (no meta-phrases)
    - Use the central claim as the correct option; create plausible distractors via negation and distortion
    """
    text = (passage_text or "").strip()
    # Basic sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s and len(s.strip()) > 20]
    if not sentences:
        # fallback simple question
        key_phrase = (topic or "the topic").strip()
        central = f"A key idea is that {key_phrase} modulates metabolic function."
    else:
        # Choose a mid-length, contentful sentence
        sentences.sort(key=len)
        central = sentences[min(len(sentences)//2, len(sentences)-1)]
        # Clip overly long
        if len(central) > 220:
            central = central[:220].rsplit(" ", 1)[0] + "..."

    # Build stem
    stem = f"According to the excerpt, which statement best reflects the passage?"

    # Correct option is a light paraphrase or the sentence itself
    correct = central
    # Simple distractors
    negation = re.sub(r"\b(is|are|was|were|supports|promotes|increases|reduces)\b",
                      "does not", central, flags=re.IGNORECASE)
    distortion = f"{central.split(',')[0]} in only rare cases, if at all."
    unrelated = f"The passage argues an unrelated point about {(topic or 'another topic')} instead."
    options = [correct, negation, distortion, unrelated][:num_options]

    difficulty = estimate_difficulty_score(text)

    return {
        "question_text": stem,
        "options": options,
        "correct_answer": 0,
        "difficulty": difficulty,
        "rationale": text[:400] + ("..." if len(text) > 400 else ""),
    }
