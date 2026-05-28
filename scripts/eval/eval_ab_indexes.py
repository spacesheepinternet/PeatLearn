#!/usr/bin/env python3
"""
A/B Eval: ray-peat-corpus-v3 (Gemini 3072-dim) vs ray-peat-corpus-v3-ft (fine-tuned 768-dim)

For each question in the eval set, queries both indexes with:
  - Reranker DISABLED (raw embedding retrieval)
  - Reranker ENABLED (cross-encoder re-scored)

Metrics per question:
  - Top-k chunk IDs and scores from each index
  - Chunk overlap % between indexes (pre and post rerank)
  - Rank correlation between indexes (Spearman)
  - Answer grounding flag (manual review prompt printed)

Output:
  data/eval/results_ab_<timestamp>.json   — full results
  data/eval/report_ab_<timestamp>.txt     — human-readable summary

Usage:
    python scripts/eval/eval_ab_indexes.py
    python scripts/eval/eval_ab_indexes.py --questions data/eval/questions_adversarial_v1.jsonl
    python scripts/eval/eval_ab_indexes.py --top-k 10 --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

INDEX_A = "ray-peat-corpus-v3"       # Gemini 3072-dim
INDEX_B = "ray-peat-corpus-v3-ft"    # Fine-tuned 768-dim
TOP_K = 10


# ── Embedders ─────────────────────────────────────────────────────────────────

async def embed_for_index_a(query: str) -> list[float]:
    """Gemini gemini-embedding-001, 3072-dim."""
    from peatlearn.rag.embedder import get_embedding_async
    return await get_embedding_async(query)


def embed_for_index_b(query: str) -> list[float]:
    """Fine-tuned peat-embeddinggemma-ft, 768-dim."""
    global _ft_model
    if _ft_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        model_dir = PROJECT_ROOT / "data" / "models" / "embeddings" / "peat-embeddinggemma-ft"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ft_model = SentenceTransformer(str(model_dir), trust_remote_code=True, device=device)
        print(f"  [FT model loaded on {device}]")
    return _ft_model.encode([query], normalize_embeddings=True)[0].tolist()

_ft_model = None


# ── Pinecone query ─────────────────────────────────────────────────────────────

def query_index(index, embedding: list[float], top_k: int) -> list[dict]:
    """Query a Pinecone index, return list of {id, score, metadata}."""
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    hits = []
    for m in results.matches:
        hits.append({
            "id": m.id,
            "score": float(m.score),
            "context": (m.metadata or {}).get("context", ""),
            "ray_peat_response": (m.metadata or {}).get("text", ""),
            "source_file": (m.metadata or {}).get("source_file", ""),
            "primary_topic": (m.metadata or {}).get("primary_topic", ""),
            "source_type": (m.metadata or {}).get("source_type", ""),
        })
    return hits


# ── Reranker ──────────────────────────────────────────────────────────────────

def apply_reranker(query: str, hits: list[dict]) -> list[dict]:
    from peatlearn.rag.reranker import rerank
    return rerank(query, hits)


# ── Metrics ───────────────────────────────────────────────────────────────────

def chunk_overlap(ids_a: list[str], ids_b: list[str]) -> float:
    """Jaccard overlap of two ID lists."""
    set_a, set_b = set(ids_a), set(ids_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def spearman_rank_correlation(ids_a: list[str], ids_b: list[str]) -> float:
    """Spearman correlation of rank positions for shared chunks."""
    shared = list(set(ids_a) & set(ids_b))
    if len(shared) < 2:
        return float("nan")
    rank_a = {id_: i for i, id_ in enumerate(ids_a)}
    rank_b = {id_: i for i, id_ in enumerate(ids_b)}
    ra = [rank_a[x] for x in shared]
    rb = [rank_b[x] for x in shared]
    n = len(shared)
    d2 = sum((a - b) ** 2 for a, b in zip(ra, rb))
    rho = 1 - (6 * d2) / (n * (n ** 2 - 1))
    return round(rho, 4)


# ── Answer generation ─────────────────────────────────────────────────────────

async def generate_answer(query: str, hits: list[dict], session) -> str:
    """Generate a Gemini answer from top hits."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[NO API KEY]"

    context_block = "\n\n".join(
        f"[{i+1}] {h.get('context','')}\n{h.get('ray_peat_response','')}"
        for i, h in enumerate(hits[:5])
    )
    prompt = (
        f"You are answering a health question based strictly on Ray Peat's work.\n"
        f"Only use information from the sources below. If the answer is not in the sources, say so.\n\n"
        f"Question: {query}\n\nSources:\n{context_block}\n\nAnswer:"
    )
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with session.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return f"[HTTP {resp.status}]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args):
    from pinecone import Pinecone

    # Load questions
    q_path = Path(args.questions)
    if not q_path.is_absolute():
        q_path = PROJECT_ROOT / q_path
    questions = []
    with open(q_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    print(f"\n{'='*70}")
    print(f"  A/B EVAL: {INDEX_A} vs {INDEX_B}")
    print(f"{'='*70}")
    print(f"  Questions: {len(questions)}")
    print(f"  Top-k:     {args.top_k}")
    print(f"  Dry run:   {args.dry_run}\n")

    if args.dry_run:
        print("  DRY RUN — questions loaded, no API calls.")
        for q in questions:
            print(f"  [{q['id']}] {q['question'][:80]}")
        return

    # Init Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx_a = pc.Index(INDEX_A)
    idx_b = pc.Index(INDEX_B)
    print(f"  Index A ({INDEX_A}): {idx_a.describe_index_stats().total_vector_count:,} vectors")
    print(f"  Index B ({INDEX_B}): {idx_b.describe_index_stats().total_vector_count:,} vectors\n")

    # Warm up reranker
    print("  Warming up reranker...")
    from peatlearn.rag.reranker import _load_done
    _load_done.wait(timeout=30)
    from peatlearn.rag.reranker import is_cross_encoder_available
    print(f"  Cross-encoder available: {is_cross_encoder_available()}\n")

    results = []

    async with aiohttp.ClientSession() as session:
        for i, q in enumerate(questions):
            qid = q["id"]
            question = q["question"]
            print(f"  [{i+1:02d}/{len(questions)}] {qid} — {question[:60]}...")

            t0 = time.time()

            # Embed for both indexes
            emb_a = await embed_for_index_a(question)
            emb_b = embed_for_index_b(question)

            # Query both indexes (raw)
            hits_a_raw = query_index(idx_a, emb_a, args.top_k)
            hits_b_raw = query_index(idx_b, emb_b, args.top_k)

            ids_a_raw = [h["id"] for h in hits_a_raw]
            ids_b_raw = [h["id"] for h in hits_b_raw]

            # Rerank both
            hits_a_reranked = apply_reranker(question, hits_a_raw)
            hits_b_reranked = apply_reranker(question, hits_b_raw)

            ids_a_reranked = [h["id"] for h in hits_a_reranked]
            ids_b_reranked = [h["id"] for h in hits_b_reranked]

            # Metrics
            overlap_raw = chunk_overlap(ids_a_raw, ids_b_raw)
            overlap_reranked = chunk_overlap(ids_a_reranked, ids_b_reranked)
            spearman_raw = spearman_rank_correlation(ids_a_raw, ids_b_raw)
            spearman_reranked = spearman_rank_correlation(ids_a_reranked, ids_b_reranked)

            # Generate answers (reranked top-5)
            ans_a = await generate_answer(question, hits_a_reranked, session)
            ans_b = await generate_answer(question, hits_b_reranked, session)

            elapsed = time.time() - t0

            result = {
                "id": qid,
                "question": question,
                "category": q.get("category", ""),
                "false_premise": q.get("false_premise", False),
                "elapsed_s": round(elapsed, 2),
                "index_a": {
                    "raw_ids": ids_a_raw,
                    "raw_scores": [round(h["score"], 4) for h in hits_a_raw],
                    "raw_topics": [h["primary_topic"] for h in hits_a_raw],
                    "reranked_ids": ids_a_reranked,
                    "reranked_scores": [round(h.get("rerank_score", 0), 4) for h in hits_a_reranked],
                    "answer": ans_a,
                },
                "index_b": {
                    "raw_ids": ids_b_raw,
                    "raw_scores": [round(h["score"], 4) for h in hits_b_raw],
                    "raw_topics": [h["primary_topic"] for h in hits_b_raw],
                    "reranked_ids": ids_b_reranked,
                    "reranked_scores": [round(h.get("rerank_score", 0), 4) for h in hits_b_reranked],
                    "answer": ans_b,
                },
                "metrics": {
                    "overlap_raw": round(overlap_raw, 4),
                    "overlap_reranked": round(overlap_reranked, 4),
                    "spearman_raw": spearman_raw,
                    "spearman_reranked": spearman_reranked,
                    "shared_raw": len(set(ids_a_raw) & set(ids_b_raw)),
                    "shared_reranked": len(set(ids_a_reranked) & set(ids_b_reranked)),
                },
            }
            results.append(result)

            # Print per-question summary
            print(f"       overlap raw={overlap_raw:.2f} reranked={overlap_reranked:.2f}  "
                  f"spearman raw={spearman_raw:.3f} reranked={spearman_reranked:.3f}  "
                  f"({elapsed:.1f}s)")

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

    # Aggregate metrics
    overlaps_raw = [r["metrics"]["overlap_raw"] for r in results]
    overlaps_reranked = [r["metrics"]["overlap_reranked"] for r in results]
    spearmans_raw = [r["metrics"]["spearman_raw"] for r in results if not (isinstance(r["metrics"]["spearman_raw"], float) and r["metrics"]["spearman_raw"] != r["metrics"]["spearman_raw"])]
    spearmans_reranked = [r["metrics"]["spearman_reranked"] for r in results if not (isinstance(r["metrics"]["spearman_reranked"], float) and r["metrics"]["spearman_reranked"] != r["metrics"]["spearman_reranked"])]

    summary = {
        "avg_overlap_raw": round(float(np.mean(overlaps_raw)), 4),
        "avg_overlap_reranked": round(float(np.mean(overlaps_reranked)), 4),
        "avg_spearman_raw": round(float(np.mean(spearmans_raw)), 4) if spearmans_raw else None,
        "avg_spearman_reranked": round(float(np.mean(spearmans_reranked)), 4) if spearmans_reranked else None,
        "n_questions": len(results),
        "high_overlap_raw": sum(1 for x in overlaps_raw if x >= 0.8),
        "high_overlap_reranked": sum(1 for x in overlaps_reranked if x >= 0.8),
    }

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = EVAL_DIR / f"results_ab_{ts}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Questions:              {summary['n_questions']}")
    print(f"  Avg overlap (raw):      {summary['avg_overlap_raw']:.3f}")
    print(f"  Avg overlap (reranked): {summary['avg_overlap_reranked']:.3f}")
    print(f"  Avg Spearman (raw):     {summary['avg_spearman_raw']}")
    print(f"  Avg Spearman (reranked):{summary['avg_spearman_reranked']}")
    print(f"  High overlap (≥0.8) raw:      {summary['high_overlap_raw']}/{summary['n_questions']}")
    print(f"  High overlap (≥0.8) reranked: {summary['high_overlap_reranked']}/{summary['n_questions']}")
    print(f"\n  Results saved: {results_path.relative_to(PROJECT_ROOT)}")

    # Print answers for manual grounding review
    print(f"\n{'='*70}")
    print("  ANSWERS FOR MANUAL GROUNDING REVIEW")
    print(f"{'='*70}")
    for r in results:
        print(f"\n[{r['id']}] {r['question']}")
        print(f"  --- Index A ---")
        print(f"  {r['index_a']['answer'][:300]}...")
        print(f"  --- Index B ---")
        print(f"  {r['index_b']['answer'][:300]}...")

    import aiohttp as _aiohttp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/eval/questions_adversarial_v1.jsonl")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    import aiohttp
    main()
