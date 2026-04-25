# Handoff — 2026-04-25

## Current Branch & Git State
- Branch: `main` (up to date with origin/main)
- Uncommitted changes: 7 modified, ~20 untracked (same set as last session — nothing committed yet)
- Last 5 commits:
  - `057580e` exp: disable HyDE — raw query embedding only (9.25/10, stabilizes B4/F1/E3)
  - `c9d6642` feat: add 10 colloquial eval questions (category I) — 9.32/10
  - `1e6ff91` fix: F3 false-ABSTAIN — meta-question framing words in entity stop list
  - `b95b8f0` feat: query vocabulary normalization — colloquial terms map to Peat's corpus
  - `9432e27` fix: reject false premises in first sentence (H14 whole grain weakness)

## What Was Being Worked On

### A/B Eval: v3 corpus, Gemini 3072 vs EmbeddingGemma ft 768

Both v3 Pinecone indexes were confirmed fully uploaded (14,407 vectors each) from the prior session. This session ran the full production A/B eval to get scored results.

**Eval flow:**
1. `eval_production_ab.py` — qualitative full-pipeline run (25 adversarial questions, both indexes). Results in `data/eval/results_production_ab_20260425_060617.json`.
2. `eval_rag_quality.py` — LLM-judged 55-question eval, Index A first, then Index B.

**Bug discovered and fixed mid-session:** `rag_system.py` called `get_embedding()` directly from `peatlearn.rag.embedder` (hardcoded Gemini 3072-dim), bypassing `vector_search.py` entirely. When pointed at the 768-dim ft index, this sent 3072-dim vectors to a 768-dim Pinecone index → dimension mismatch → SHA-256 hash fallback → garbage retrieval → ~1.00/10 scores. Fixed by:
- Adding `embed_query()` sync method to `PineconeVectorSearch` that routes by `self.embedding_dimensions` (768 → local EmbeddingGemma, 3072 → Gemini)
- Replacing the hardcoded `get_embedding()` call in `rag_system.py` line 207 with `self.search_engine.embed_query(search_query)`

### Final Scored Results

| Index | Overall | vs Baseline (8.6) |
|-------|---------|-------------------|
| **Index A — ray-peat-corpus-v3 (Gemini 3072)** | **9.28** | **+0.68** |
| v2 active — ray-peat-corpus-v2 (EmbeddingGemma ft) | 9.09 | +0.49 |
| Index B — ray-peat-corpus-v3-ft (EmbeddingGemma ft) | 8.69 | +0.09 |

Per-category breakdown:

| Category | v2 | A (Gemini) | B (Gemma) |
|----------|----|-----------|-----------|
| core_bioenergetics | 9.27 | 9.28 | 9.27 |
| hormones_endocrine | 9.12 | 9.32 | 8.94 |
| nutrition_foods | 9.24 | 8.79 | 7.48 |
| disease_clinical | 9.36 | 9.29 | 9.22 |
| edge_ambiguous | 9.07 | 9.11 | 8.93 |
| edge_nuanced | 7.63 | 8.69 | 7.56 |
| cross_concept | 9.08 | 9.43 | 9.21 |
| adversarial | 9.18 | 9.59 | 8.59 |
| colloquial_user | 9.07 | 9.24 | 8.80 |

## Key Decisions Made

- **Index A (Gemini 3072, v3) wins — promote to production.** +0.19 over current v2, dominant across adversarial, edge_nuanced, cross_concept. Only weakness: nutrition_foods (8.79 vs 9.24 on v2) — worth investigating but not a blocker.

- **Fine-tuned EmbeddingGemma is underperforming on v3 corpus.** Index B scores *worse* than v2 despite using the same clean corpus. avg_top_relevance 0.479 vs 0.703 for Index A — the ft model is pulling less relevant chunks. The council was right: the fine-tuning pairs (256 triples) are insufficient. Either retrain with more data or accept Gemini as the embedding model going forward.

- **`embed_query()` is now the canonical embedding entry point** in `PineconeVectorSearch`. All callers should use `self.search_engine.embed_query()` — never import `get_embedding` from `embedder.py` directly in `rag_system.py`.

- **HyDE remains disabled.** Bypassed at lines 319-321 of `rag_system.py`. The HyDE REST calls in the codebase still hardcode `gemini-embedding-001` — if HyDE is ever re-enabled for the 768-dim index, those calls also need to route through `embed_query()`.

## Active Plan
None.

## What Needs to Happen Next

1. **Promote Index A to production** — change `PINECONE_INDEX_NAME=ray-peat-corpus-v3` in `.env`. Run a quick smoke test (`python scripts/eval_rag_quality.py --limit 5`) to confirm it works end-to-end with the new index. Also update `EMBEDDING_DIMENSIONS=3072` and `EMBEDDING_MODEL=gemini-embedding-001` in `.env` if they differ.

2. **Investigate nutrition_foods regression** — Index A scores 8.79 vs v2's 9.24. Pull the worst-scoring nutrition questions from `data/eval/results_20260425_062746.json` and inspect which chunks were retrieved vs expected. Likely a corpus coverage gap in v3 for food-specific topics, not an embedding issue.

3. **Commit this session's work** — large uncommitted diff. Key new files: `peatlearn/rag/embedder.py`, `peatlearn/rag/vector_search.py` (embed_query fix), `peatlearn/adaptive/rag_system.py` (embed_query routing), all new scripts, corpus_v3, eval scripts.

4. **Decide on EmbeddingGemma fine-tuning roadmap** — 256 training triples are clearly insufficient (Index B underperforms v2). Options: (a) generate more pairs and retrain, (b) accept Gemini as primary embedder and retire the ft model for RAG, (c) keep ft model only for offline/quota-free fallback. Recommend (b) given the eval results.

5. **Fix HyDE embedding calls if re-enabling HyDE** — lines ~261 and ~289 in `rag_system.py` still call Gemini REST API directly. Route through `self.search_engine.embed_query()` before re-enabling.

## Open Questions / Blockers

- **nutrition_foods regression in Index A**: Is it a corpus gap (v3 missing certain food content from v2) or an embedding alignment issue? Needs chunk-level inspection.
- **EmbeddingGemma ft future**: With only 256 training triples and avg_top_relevance of 0.479, the model isn't production-ready for RAG retrieval. Retraining requires either more training data generation or a different fine-tuning strategy.
- **Gifted API key**: Still in use (`AIzaSyDAY9UQHyghfmgF0gCQT2j3JKI4z08OWDI`). Should be reverted to original key (`AIzaSyCohwsvZTpLa_jALA5gZYEgQlWG3zgrgcU`) in `.env` after promoting Index A.

## Files Modified This Session

### Modified (tracked)
```
.claude/rules/embedding.md       — updated corpus stats, model paths, rollback info
CLAUDE.md                        — tech stack table updates
HANDOFF.md                       — this file
config/settings.py               — settings update
peatlearn/adaptive/rag_system.py — embed_query routing (line 207); previous session changes
peatlearn/rag/reranker.py        — reranker changes (prior session)
peatlearn/rag/vector_search.py   — embed_query() + _embed_local() + generate_query_embedding routing
```

### New (untracked, key files)
```
peatlearn/rag/embedder.py                        — Gemini embedding singleton, 3072-dim, taskType: RETRIEVAL_QUERY
scripts/build_corpus_v3.py                       — smart parser, dedup, 24-topic taxonomy
scripts/embed_and_upload_v3.py                   — Gemini embed + Pinecone upload; taskType: RETRIEVAL_DOCUMENT
scripts/embed_and_upload_v3_ft.py                — EmbeddingGemma embed + upload to v3-ft index
scripts/eval_ab_indexes.py                       — retrieval-level A/B comparison (overlap, Spearman)
scripts/eval_production_ab.py                    — full-pipeline A/B eval, 25 adversarial Qs
data/corpus_v3/corpus_v3_20260423_184732.jsonl   — 14,407-record clean corpus
data/eval/results_production_ab_20260425_060617.json — qualitative A/B results (25 Qs)
data/eval/results_20260425_062746.json           — Index A scored eval (9.28/10)
data/eval/results_20260425_065741.json           — Index B scored eval (8.69/10)
```

## Embedding Architecture (current state)

| Index | Model | Dim | Status | Notes |
|-------|-------|-----|--------|-------|
| `ray-peat-corpus` | Gemini (legacy) | 3072 | Live — rollback only | Old broken parser data |
| `ray-peat-corpus-v2` | EmbeddingGemma ft | 768 | Live — **current production** | 26,431 vectors; 9.09/10 |
| `ray-peat-corpus-v3` | Gemini | 3072 | Live — **promote to production** | 14,407 vectors; 9.28/10 |
| `ray-peat-corpus-v3-ft` | EmbeddingGemma ft | 768 | Live — underperforming | 14,407 vectors; 8.69/10 |

Active index in `.env`: `PINECONE_INDEX_NAME=ray-peat-corpus-v2` — change to `ray-peat-corpus-v3` to promote.
