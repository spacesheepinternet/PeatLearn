#!/usr/bin/env python3
"""
Lightweight concept graph builder using co-occurrence statistics over the cleaned corpus.
- No heavy ML dependencies; pure-Python + NumPy
- Extracts lowercase terms via simple token regex and domain keyword boosting
- Computes Positive PMI (PPMI) to weight edges
- Provides query expansion using top connected terms
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

DATA_DIR = Path("data/processed/ai_cleaned")
CACHE_PATH = DATA_DIR / "concept_graph_ppmi.json"

# Very simple tokenization and stopwording; customize as needed
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
STOP = set(
    "the a an of and to in for with on by about from as is are was were be been it this that those these you your our their his her its at not no yes or but into over under between around more most less many few one two three can may might should would will could than then when where how what which who whom whose because since due during after before against despite toward within without across per each any all some such same other another".split()
)

# Domain boosters ensure these terms are retained even if frequent
DOMAIN_TERMS = set(
    [
        "thyroid", "t3", "t4", "progesterone", "estrogen", "cortisol", "serotonin", "dopamine", "gaba",
        "metabolism", "oxidation", "glycolysis", "mitochondria", "co2", "glycogen", "lactate",
        "aspirin", "vitamin", "saturated", "polyunsaturated", "pufa", "cholesterol",
        "glucose", "sucrose", "fructose", "sodium", "calcium", "phosphorus", "magnesium",
    ]
)


def iter_documents(limit: Optional[int] = None) -> Iterable[str]:
    count = 0
    if not DATA_DIR.exists():
        return
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if not f.endswith(".txt"):
                continue
            p = Path(root) / f
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            yield text
            count += 1
            if limit is not None and count >= limit:
                return


def extract_terms(text: str) -> List[str]:
    words = [w.lower() for w in TOKEN_RE.findall(text)]
    terms = [w for w in words if w not in STOP and len(w) > 2]
    return terms


def build_cooccurrence(window_size: int = 20, doc_limit: Optional[int] = None) -> Tuple[Counter, Counter]:
    term_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for doc in iter_documents(limit=doc_limit):
        terms = extract_terms(doc)
        term_counts.update(terms)
        # sliding window
        for i in range(len(terms)):
            w_i = terms[i]
            # pair within window to the right
            for j in range(i + 1, min(len(terms), i + window_size)):
                w_j = terms[j]
                if w_i == w_j:
                    continue
                a, b = (w_i, w_j) if w_i < w_j else (w_j, w_i)
                pair_counts[(a, b)] += 1

    return term_counts, pair_counts


def compute_ppmi(term_counts: Counter, pair_counts: Counter, k: float = 1.0) -> Dict[str, Dict[str, float]]:
    total_terms = sum(term_counts.values())
    total_pairs = sum(pair_counts.values())
    if total_terms == 0 or total_pairs == 0:
        return {}

    # Probabilities with add-k smoothing (light)
    def p_term(t: str) -> float:
        return (term_counts[t] + k) / (total_terms + k * len(term_counts))

    graph: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (a, b), c in pair_counts.items():
        p_a = p_term(a)
        p_b = p_term(b)
        p_ab = (c + k) / (total_pairs + k * len(pair_counts))
        denom = p_a * p_b
        if denom <= 0:
            continue
        pmi = np.log(max(p_ab / denom, 1e-12))
        ppmi = float(max(0.0, pmi))
        if ppmi == 0.0:
            continue
        graph[a][b] = ppmi
        graph[b][a] = ppmi
    return graph


def build_and_cache_graph(doc_limit: Optional[int] = 400) -> Dict[str, Dict[str, float]]:
    term_counts, pair_counts = build_cooccurrence(doc_limit=doc_limit)
    graph = compute_ppmi(term_counts, pair_counts)
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"graph": graph}, f)
    except Exception:
        pass
    return graph


def load_graph() -> Dict[str, Dict[str, float]]:
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            return data.get("graph", {})
        except Exception:
            return {}
    return {}


def ensure_graph(doc_limit: Optional[int] = 400) -> Dict[str, Dict[str, float]]:
    g = load_graph()
    if g:
        return g
    return build_and_cache_graph(doc_limit=doc_limit)


def expand_query_terms(query: str, max_expansions: int = 5, doc_limit: Optional[int] = 400) -> Dict[str, any]:
    graph = ensure_graph(doc_limit=doc_limit)
    q_terms = extract_terms(query)
    expansions: Dict[str, float] = {}
    for q in q_terms:
        neighbors = graph.get(q, {})
        for term, weight in neighbors.items():
            if term in q_terms:
                continue
            # boost domain terms
            score = weight * (1.2 if term in DOMAIN_TERMS else 1.0)
            if term not in expansions or score > expansions[term]:
                expansions[term] = score
    # Rank and take top-k
    ranked = sorted(expansions.items(), key=lambda x: x[1], reverse=True)[:max_expansions]
    expanded_terms = [t for t, _ in ranked]
    expanded_query = query + (" " + " ".join(expanded_terms) if expanded_terms else "")
    return {
        "original_query": query,
        "expanded_query": expanded_query,
        "expansion_terms": expanded_terms,
        "num_expansions": len(expanded_terms),
    }
