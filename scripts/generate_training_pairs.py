#!/usr/bin/env python3
"""
Generate training pairs for embedding fine-tuning.

Creates (query, positive_passage, negative_passage) triples for contrastive
learning. Uses three data sources:

1. Eval questions (55) — match to top Pinecone passages
2. Query normalizer entries (53) — generate natural questions from colloquial terms
3. Synthetic colloquial queries — Gemini generates casual questions per topic

Output: data/training/embedding_pairs.jsonl
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
OUTPUT_FILE = PROJECT_ROOT / "data" / "training" / "embedding_pairs.jsonl"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval" / "questions.json"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"


def gemini_generate(prompt: str, max_tokens: int = 300) -> str | None:
    """Call Gemini for text generation. Returns None on failure."""
    if not GEMINI_API_KEY:
        return None
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": max_tokens},
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


async def retrieve_top_passages(
    vs: PineconeVectorSearch, query: str, top_k: int = 5
) -> list[dict]:
    """Retrieve top passages from Pinecone for a query."""
    results = await vs.search(query, top_k=top_k, min_similarity=0.3)
    return [
        {
            "source_file": r.source_file,
            "context": r.context,
            "text": r.ray_peat_response,
            "score": r.similarity_score,
        }
        for r in results
    ]


async def generate_from_eval_questions(vs: PineconeVectorSearch) -> list[dict]:
    """Source 1: Use eval questions as queries, top passages as positives."""
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = [
        q for q in data["questions"] if q.get("expected_behavior") == "answer"
    ]

    pairs = []
    for q in questions:
        passages = await retrieve_top_passages(vs, q["question"], top_k=8)
        if len(passages) < 2:
            continue

        # Top passage = positive, bottom passage = hard negative
        positive = passages[0]
        # Pick a hard negative: high cosine but different source file
        hard_neg = None
        for p in reversed(passages):
            if p["source_file"] != positive["source_file"]:
                hard_neg = p
                break
        if not hard_neg:
            hard_neg = passages[-1]

        pairs.append(
            {
                "anchor": q["question"],
                "positive": f"Context: {positive['context']}\nRay Peat: {positive['text']}",
                "negative": f"Context: {hard_neg['context']}\nRay Peat: {hard_neg['text']}",
                "source": "eval_question",
            }
        )
    return pairs


async def generate_from_normalizer(vs: PineconeVectorSearch) -> list[dict]:
    """Source 2: Generate colloquial questions from normalizer entries."""
    # Map colloquial terms to natural questions
    term_question_templates = [
        "what does ray peat think about {term}?",
        "is {term} good or bad according to peat?",
        "{term} ray peat",
        "what should I know about {term}?",
        "how does {term} affect health?",
    ]

    pairs = []
    for term, expansion in _EXPANSIONS.items():
        # Pick 2 random templates
        templates = random.sample(term_question_templates, min(2, len(term_question_templates)))
        for template in templates:
            query = template.format(term=term)
            passages = await retrieve_top_passages(vs, query, top_k=6)
            if len(passages) < 2:
                continue

            positive = passages[0]
            hard_neg = None
            for p in reversed(passages):
                if p["source_file"] != positive["source_file"]:
                    hard_neg = p
                    break
            if not hard_neg:
                hard_neg = passages[-1]

            pairs.append(
                {
                    "anchor": query,
                    "positive": f"Context: {positive['context']}\nRay Peat: {positive['text']}",
                    "negative": f"Context: {hard_neg['context']}\nRay Peat: {hard_neg['text']}",
                    "source": "normalizer_term",
                }
            )
    return pairs


async def generate_synthetic_colloquial(vs: PineconeVectorSearch) -> list[dict]:
    """Source 3: Use Gemini to generate casual user questions per topic."""
    topic_areas = [
        "thyroid and metabolism",
        "polyunsaturated fats and seed oils",
        "sugar and carbohydrates",
        "progesterone and estrogen",
        "stress hormones and cortisol",
        "dairy and milk",
        "gelatin and amino acids",
        "coconut oil and saturated fats",
        "serotonin and depression",
        "aging and longevity",
        "cancer and metabolism",
        "fasting and ketogenic diets",
        "inflammation and prostaglandins",
        "vitamin D and calcium",
        "hypothyroidism diagnosis",
        "aspirin and anti-inflammatory",
        "coffee and caffeine",
        "gut health and endotoxin",
        "sleep and melatonin",
        "hair loss and hormones",
    ]

    prompt_template = (
        "Generate 5 short, casual questions that a regular person (not a researcher) "
        "would type into a health chatbot about: {topic}. "
        "Use informal language — like texting a friend. No clinical terms. "
        "One question per line, no numbering, no quotes."
    )

    pairs = []
    for topic in topic_areas:
        result = gemini_generate(prompt_template.format(topic=topic))
        if not result:
            continue

        questions = [q.strip() for q in result.strip().split("\n") if q.strip() and len(q.strip()) > 10]

        for question in questions[:5]:
            passages = await retrieve_top_passages(vs, question, top_k=6)
            if len(passages) < 2:
                continue

            positive = passages[0]
            hard_neg = None
            for p in reversed(passages):
                if p["source_file"] != positive["source_file"]:
                    hard_neg = p
                    break
            if not hard_neg:
                hard_neg = passages[-1]

            pairs.append(
                {
                    "anchor": question,
                    "positive": f"Context: {positive['context']}\nRay Peat: {positive['text']}",
                    "negative": f"Context: {hard_neg['context']}\nRay Peat: {hard_neg['text']}",
                    "source": "synthetic_colloquial",
                }
            )

        # Rate limit
        time.sleep(1)

    return pairs


async def main():
    print("Initializing PineconeVectorSearch...")
    vs = PineconeVectorSearch()

    all_pairs = []

    # Source 1: Eval questions
    print("\n[1/3] Generating pairs from eval questions...")
    eval_pairs = await generate_from_eval_questions(vs)
    print(f"  -> {len(eval_pairs)} pairs from eval questions")
    all_pairs.extend(eval_pairs)

    # Source 2: Normalizer entries
    print("\n[2/3] Generating pairs from normalizer entries...")
    norm_pairs = await generate_from_normalizer(vs)
    print(f"  -> {len(norm_pairs)} pairs from normalizer entries")
    all_pairs.extend(norm_pairs)

    # Source 3: Synthetic colloquial
    print("\n[3/3] Generating synthetic colloquial pairs via Gemini...")
    synth_pairs = await generate_synthetic_colloquial(vs)
    print(f"  -> {len(synth_pairs)} pairs from synthetic generation")
    all_pairs.extend(synth_pairs)

    # Deduplicate by anchor
    seen = set()
    deduped = []
    for p in all_pairs:
        key = p["anchor"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    print(f"\nTotal unique pairs: {len(deduped)}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in deduped:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUTPUT_FILE}")

    # Summary
    sources = {}
    for p in deduped:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
    print("\nBreakdown:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
