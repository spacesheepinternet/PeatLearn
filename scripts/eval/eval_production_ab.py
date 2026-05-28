#!/usr/bin/env python3
"""
Production pipeline A/B eval.

Runs all 25 adversarial questions through the full production RayPeatRAG
(citation gate, false-premise prompt, cross-encoder reranker, confidence
scoring, grounding verifier) for both indexes:

  Index A: ray-peat-corpus-v3   — Gemini gemini-embedding-001, 3072-dim
  Index B: ray-peat-corpus-v3-ft — fine-tuned peat-embeddinggemma-ft, 768-dim

Usage:
    python scripts/eval/eval_production_ab.py
    python scripts/eval/eval_production_ab.py --api-key AIzaSy...   # override Gemini key
    python scripts/eval/eval_production_ab.py --index-b-only        # skip Index A (no API quota needed)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

Q_PATH = EVAL_DIR / "questions_adversarial_v1.jsonl"
INDEX_A = "ray-peat-corpus-v3"
INDEX_B = "ray-peat-corpus-v3-ft"
FT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "embeddings" / "peat-embeddinggemma-ft"


# ── Embedders ─────────────────────────────────────────────────────────────────

def make_ft_embedder():
    """Load local fine-tuned SentenceTransformer, return embed function."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(FT_MODEL_DIR), trust_remote_code=True, device=device)
    print(f"  [FT embedder loaded on {device}]")
    def embed(text: str) -> list:
        return model.encode([text], normalize_embeddings=True)[0].tolist()
    return embed


# ── RAG instances ──────────────────────────────────────────────────────────────

def make_rag(index_name: str):
    """Create a RayPeatRAG instance pointing at the given Pinecone index."""
    from peatlearn.rag.vector_search import PineconeVectorSearch
    from peatlearn.adaptive.rag_system import RayPeatRAG
    search = PineconeVectorSearch(index_name=index_name)
    rag = RayPeatRAG(search_engine=search)
    return rag


# ── Run one question through production pipeline ───────────────────────────────

def run_question(rag, question: str, embed_override=None) -> dict:
    """
    Run a question through RayPeatRAG._get_rag_response_sync.
    If embed_override is provided, temporarily patches peatlearn.rag.embedder.get_embedding.
    Returns dict with answer, elapsed_s, abstained flag.
    """
    import peatlearn.rag.embedder as _emb_mod

    original = _emb_mod.get_embedding
    if embed_override:
        _emb_mod.get_embedding = embed_override

    t0 = time.time()
    try:
        answer = rag.get_rag_response(question)
    except Exception as e:
        answer = f"[ERROR: {e}]"
    finally:
        _emb_mod.get_embedding = original

    elapsed = round(time.time() - t0, 2)

    abstained = (
        "ABSTAIN" in answer
        or "I don't have sufficient information" in answer
        or "can't answer this question" in answer
        or "corpus doesn't consistently include" in answer
    )

    return {
        "answer": answer,
        "elapsed_s": elapsed,
        "abstained": abstained,
        "answer_preview": answer[:300].replace("\n", " "),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", help="Override GEMINI_API_KEY (for Index A embeddings)")
    parser.add_argument("--index-b-only", action="store_true", help="Skip Index A")
    parser.add_argument("--questions", default=str(Q_PATH))
    args = parser.parse_args()

    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key
        from config import settings as _settings_mod
        _settings_mod.settings.__dict__["GEMINI_API_KEY"] = args.api_key
        print(f"  [Using provided API key]")

    # Load questions
    questions = []
    with open(args.questions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"\n{'='*70}")
    print(f"  PRODUCTION PIPELINE A/B EVAL")
    print(f"{'='*70}")
    print(f"  Questions:   {len(questions)}")
    print(f"  Index A:     {INDEX_A} (Gemini 3072-dim)")
    print(f"  Index B:     {INDEX_B} (fine-tuned 768-dim)")
    print(f"  Index B only: {args.index_b_only}")
    print(f"  Defenses:    citation gate, false-premise prompt, reranker,")
    print(f"               confidence scoring, entity grounding, verifier")
    print()

    # Warm up FT embedder
    print("  Loading fine-tuned embedder...")
    ft_embed = make_ft_embedder()

    # Create RAG instances
    print("  Connecting to Pinecone indexes...")
    rag_b = make_rag(INDEX_B)
    rag_a = make_rag(INDEX_A) if not args.index_b_only else None
    print()

    results = []

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        cat = q["category"]
        false_premise = q.get("false_premise", False)

        print(f"  [{i+1:02d}/{len(questions)}] {qid} ({cat}) {'[FP]' if false_premise else ''}")
        print(f"    Q: {question[:70]}...")

        result = {
            "id": qid,
            "question": question,
            "category": cat,
            "false_premise": false_premise,
        }

        # Index B
        b = run_question(rag_b, question, embed_override=ft_embed)
        result["index_b"] = b
        b_flag = "ABSTAIN" if b["abstained"] else "answer"
        print(f"    B [{b_flag}] {b['answer_preview'][:120]}")

        # Index A
        if not args.index_b_only:
            time.sleep(0.5)  # avoid rate limiting between questions
            a = run_question(rag_a, question)  # uses Gemini embedder (default)
            result["index_a"] = a
            a_flag = "ABSTAIN" if a["abstained"] else "answer"
            print(f"    A [{a_flag}] {a['answer_preview'][:120]}")

        results.append(result)
        print()

        time.sleep(1.0)  # avoid Gemini LLM rate limits between questions

    # ── Summary ────────────────────────────────────────────────────────────────

    b_abstains = sum(1 for r in results if r["index_b"]["abstained"])
    a_abstains = sum(1 for r in results if "index_a" in r and r["index_a"]["abstained"]) if not args.index_b_only else None

    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Index B abstentions: {b_abstains}/{len(results)}")
    if not args.index_b_only:
        print(f"  Index A abstentions: {a_abstains}/{len(results)}")

    # Citation gate fired?
    citation_qs = [r for r in results if r["category"] == "hallucination_trap"]
    b_citation_blocked = sum(1 for r in citation_qs if "corpus doesn't consistently include" in r["index_b"]["answer"])
    print(f"\n  Citation gate fired (Index B, hallucination_trap): {b_citation_blocked}/{len(citation_qs)}")
    if not args.index_b_only:
        a_citation_blocked = sum(1 for r in citation_qs if "index_a" in r and "corpus doesn't consistently include" in r["index_a"]["answer"])
        print(f"  Citation gate fired (Index A, hallucination_trap): {a_citation_blocked}/{len(citation_qs)}")

    # False premise handling
    fp_qs = [r for r in results if r["false_premise"]]
    b_fp_rejected = sum(
        1 for r in fp_qs
        if any(phrase in r["index_b"]["answer"].lower() for phrase in [
            "premise", "doesn't reflect", "does not reflect", "not recommend",
            "does not recommend", "wouldn't say", "would not say",
            "incorrect", "misinformation", "not the case",
        ])
    )
    print(f"\n  False premise rejected (Index B): {b_fp_rejected}/{len(fp_qs)}")
    if not args.index_b_only:
        a_fp_rejected = sum(
            1 for r in fp_qs
            if "index_a" in r and any(phrase in r["index_a"]["answer"].lower() for phrase in [
                "premise", "doesn't reflect", "does not reflect", "not recommend",
                "does not recommend", "wouldn't say", "would not say",
                "incorrect", "misinformation", "not the case",
            ])
        )
        print(f"  False premise rejected (Index A): {a_fp_rejected}/{len(fp_qs)}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"results_production_ab_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "index_a": INDEX_A,
                "index_b": INDEX_B,
                "index_b_only": args.index_b_only,
                "n_questions": len(results),
                "b_abstains": b_abstains,
                "a_abstains": a_abstains,
                "b_citation_blocked": b_citation_blocked,
                "b_fp_rejected": b_fp_rejected,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Print full answers for manual review
    print(f"\n{'='*70}")
    print("  ANSWERS FOR MANUAL REVIEW")
    print(f"{'='*70}")
    for r in results:
        print(f"\n[{r['id']}] ({r['category']}) {r['question']}")
        print(f"  B: {r['index_b']['answer'][:400]}")
        if not args.index_b_only and "index_a" in r:
            print(f"  A: {r['index_a']['answer'][:400]}")


if __name__ == "__main__":
    main()
