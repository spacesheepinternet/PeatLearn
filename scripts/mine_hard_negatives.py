#!/usr/bin/env python3
"""
Mine hard negatives from the fine-tuned embedding index using cross-encoder scoring.

A "hard negative" is a passage that:
  - High cosine similarity (looks similar in embedding space)
  - Low cross-encoder relevance (actually not relevant to the query)

These are the most informative negatives for contrastive learning — they
teach the embedding model where its current space is wrong.

Also generates additional colloquial anchors for broader topic coverage.

Input:  data/training/embedding_pairs.jsonl (existing pairs)
        Pinecone ray-peat-corpus-v2 (current index)
Output: data/training/embedding_pairs_v2.jsonl (merged, deduplicated)

Usage:
    python scripts/mine_hard_negatives.py
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peatlearn.rag.vector_search import PineconeVectorSearch
from peatlearn.rag.query_normalizer import _EXPANSIONS

# --- Config ---
EXISTING_PAIRS = PROJECT_ROOT / "data" / "training" / "embedding_pairs.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "training" / "embedding_pairs_v2.jsonl"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"


def gemini_generate(prompt: str, max_tokens: int = 400) -> str | None:
    """Call Gemini for text generation."""
    if not GEMINI_API_KEY:
        return None
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": max_tokens},
    }
    for attempt in range(3):
        try:
            r = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
    return None


def load_cross_encoder():
    """Load the cross-encoder for relevance scoring."""
    from sentence_transformers.cross_encoder import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
    return ce


async def retrieve_passages(vs: PineconeVectorSearch, query: str, top_k: int = 20) -> list[dict]:
    """Retrieve top passages from the fine-tuned index."""
    results = await vs.search(query, top_k=top_k, min_similarity=0.2)
    return [
        {
            "source_file": r.source_file,
            "context": r.context,
            "text": r.ray_peat_response,
            "score": r.similarity_score,
        }
        for r in results
    ]


def score_with_cross_encoder(ce, query: str, passages: list[dict]) -> list[dict]:
    """Score each passage with the cross-encoder and add ce_score field."""
    if not passages:
        return passages
    pairs = [(query, f"Context: {p['context']}\nRay Peat: {p['text']}") for p in passages]
    scores = ce.predict(pairs)
    for p, s in zip(passages, scores):
        p["ce_score"] = float(s)
    return passages


async def mine_from_queries(vs, ce, queries: list[str]) -> list[dict]:
    """Mine hard negatives for a set of queries.

    For each query:
    - Positive: passage with highest cross-encoder score
    - Hard negative: passage with high cosine but LOW cross-encoder score
    """
    pairs = []
    for query in queries:
        passages = await retrieve_passages(vs, query, top_k=20)
        if len(passages) < 4:
            continue

        scored = score_with_cross_encoder(ce, query, passages)
        scored.sort(key=lambda p: p["ce_score"], reverse=True)

        # Best CE-scored passage = positive
        positive = scored[0]
        if positive["ce_score"] < 0:
            continue  # even the best passage isn't relevant

        # Hard negative: high cosine (top 10) but lowest CE score, different source file
        cosine_top = [p for p in passages[:10] if p["source_file"] != positive["source_file"]]
        if not cosine_top:
            cosine_top = passages[5:15]
        cosine_top.sort(key=lambda p: p.get("ce_score", 0))
        hard_neg = cosine_top[0] if cosine_top else scored[-1]

        # Only keep if there's a meaningful CE gap (positive is much more relevant than negative)
        ce_gap = positive["ce_score"] - hard_neg.get("ce_score", 0)
        if ce_gap < 2.0:
            continue

        pairs.append({
            "anchor": query,
            "positive": f"Context: {positive['context']}\nRay Peat: {positive['text']}",
            "negative": f"Context: {hard_neg['context']}\nRay Peat: {hard_neg['text']}",
            "source": "hard_negative_mined",
            "ce_gap": round(ce_gap, 2),
        })

    return pairs


async def generate_expanded_colloquial(vs, ce) -> list[dict]:
    """Generate a broader set of colloquial queries across more topics."""
    topic_areas = [
        # Original 20 topics already covered — these are NEW topics
        "liver health and detox",
        "pregnancy and nutrition",
        "children nutrition and development",
        "bone density and osteoporosis",
        "water retention and edema",
        "autoimmune conditions",
        "brain health and cognition",
        "testosterone and men's health",
        "menopause symptoms",
        "blood sugar and insulin",
        "cholesterol and heart health",
        "respiratory health and CO2",
        "digestive enzymes and digestion",
        "iron and anemia",
        "vitamin A and retinol",
        "salt and sodium",
        "calcium metabolism",
        "light therapy and red light",
        "alcohol and liver",
        "protein requirements and sources",
        "weight gain and underweight",
        "headaches and migraines",
        "dental health",
        "eye health and vision",
        "antibiotics and infections",
    ]

    prompt_template = (
        "Generate 8 short, casual questions a regular person would type into a health chatbot "
        "about: {topic}. Use very informal language — misspellings OK, text-speak OK, "
        "like someone searching on their phone. Mix question styles: some as full questions, "
        "some as search phrases, some with slang. One per line, no numbering."
    )

    pairs = []
    for topic in topic_areas:
        result = gemini_generate(prompt_template.format(topic=topic))
        if not result:
            continue

        questions = [q.strip() for q in result.strip().split("\n") if q.strip() and len(q.strip()) > 8]

        # Mine hard negatives for each generated question
        batch_pairs = await mine_from_queries(vs, ce, questions[:8])
        pairs.extend(batch_pairs)

        # Rate limit
        time.sleep(0.5)
        if len(pairs) % 20 == 0:
            print(f"    ... {len(pairs)} mined pairs so far")

    return pairs


async def mine_from_benchmark_weaknesses(vs, ce) -> list[dict]:
    """Mine hard negatives for queries that the benchmark showed were weak."""
    # Single-word and short queries that need better coverage
    weak_queries = [
        # E-category (edge/ambiguous) — single word or vague
        "diabetes", "stress", "aging", "cancer", "inflammation",
        "hormones", "thyroid", "fasting", "keto", "supplements",
        # Paraphrases of B3 (thyroid support) — scored 7.88
        "how do I support my thyroid naturally",
        "best foods for thyroid health",
        "what supplements help thyroid function",
        "my thyroid is slow what can I do",
        "thyroid support without medication",
        # Paraphrases of weak colloquial patterns
        "is dairy bad", "should I eat fruit", "what about coconut oil",
        "red meat good or bad", "is coffee healthy", "best oils for cooking",
        "how to speed up metabolism", "natural ways to balance hormones",
        "is sugar really that bad", "what causes inflammation in the body",
        # HRT/bioidentical — was historically weak
        "bioidentical hormones vs synthetic",
        "is HRT safe long term",
        "natural hormone replacement options",
        "progesterone cream benefits",
    ]

    return await mine_from_queries(vs, ce, weak_queries)


async def mine_from_normalizer_expanded(vs, ce) -> list[dict]:
    """Expand normalizer entries with more diverse templates."""
    templates = [
        "what does ray peat say about {term}?",
        "is {term} good or bad?",
        "{term} ray peat opinion",
        "should I avoid {term}?",
        "{term} benefits and risks",
        "how does {term} affect my health?",
        "ray peat {term} advice",
        "what's the deal with {term}?",
    ]

    queries = []
    for term in _EXPANSIONS:
        chosen = random.sample(templates, min(3, len(templates)))
        queries.extend(t.format(term=term) for t in chosen)

    return await mine_from_queries(vs, ce, queries)


async def main():
    print("Loading cross-encoder...")
    ce = load_cross_encoder()

    print("Initializing PineconeVectorSearch...")
    vs = PineconeVectorSearch()

    all_new_pairs = []

    # Source 1: Benchmark weakness mining
    print("\n[1/3] Mining hard negatives from benchmark weak spots...")
    weak_pairs = await mine_from_benchmark_weaknesses(vs, ce)
    print(f"  -> {len(weak_pairs)} pairs from benchmark weaknesses")
    all_new_pairs.extend(weak_pairs)

    # Source 2: Expanded normalizer templates
    print("\n[2/3] Mining from expanded normalizer templates...")
    norm_pairs = await mine_from_normalizer_expanded(vs, ce)
    print(f"  -> {len(norm_pairs)} pairs from normalizer expansion")
    all_new_pairs.extend(norm_pairs)

    # Source 3: Broader colloquial topics via Gemini
    print("\n[3/3] Generating broader colloquial queries + mining hard negatives...")
    colloquial_pairs = await generate_expanded_colloquial(vs, ce)
    print(f"  -> {len(colloquial_pairs)} pairs from expanded colloquial topics")
    all_new_pairs.extend(colloquial_pairs)

    # Load existing pairs
    existing = []
    if EXISTING_PAIRS.exists():
        with open(EXISTING_PAIRS, "r", encoding="utf-8") as f:
            existing = [json.loads(line) for line in f]
        print(f"\nLoaded {len(existing)} existing pairs from v1")

    # Merge and deduplicate
    merged = existing + all_new_pairs
    seen = set()
    deduped = []
    for p in merged:
        key = p["anchor"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    print(f"\nTotal unique pairs: {len(deduped)} (was {len(existing)})")
    print(f"New pairs added: {len(deduped) - len(existing)}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in deduped:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUTPUT_FILE}")

    # Summary by source
    sources = {}
    for p in deduped:
        src = p.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("\nBreakdown:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # CE gap stats for mined pairs
    mined = [p for p in deduped if "ce_gap" in p]
    if mined:
        gaps = [p["ce_gap"] for p in mined]
        print(f"\nHard negative quality (CE gap):")
        print(f"  Mean: {sum(gaps)/len(gaps):.2f}")
        print(f"  Min:  {min(gaps):.2f}")
        print(f"  Max:  {max(gaps):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
