# RAG Quality Evaluation

Reproducible benchmark for the Ray Peat RAG chatbot. Measures retrieval + generation
quality against a stable question set using both LLM-as-judge and automated metrics.

## Files

| File | Purpose |
|------|---------|
| `questions.json` | 30 benchmark questions across 7 categories (A-G) with expected topics |
| `results_<timestamp>.json` | Per-run results: per-question scores, judge reasoning, automated metrics, aggregate report |

## Run it

From the project root with the venv activated:

```bash
# Full run — 30 questions, LLM-judged, ~5–10 min
python scripts/eval_rag_quality.py

# Dry run — single category, no LLM judge
python scripts/eval_rag_quality.py --subset A --no-judge

# Multiple categories
python scripts/eval_rag_quality.py --subset A,B,E

# Debug — only run first 3 questions
python scripts/eval_rag_quality.py --limit 3
```

The script prints a summary to stdout and writes the full results JSON to
`data/eval/results_<timestamp>.json`.

## What gets scored

### Track 1 — LLM-as-judge (Gemini 2.5-flash)

Each answer is scored 1-10 on 5 dimensions, final score is a weighted average:

| Dimension | Weight | Measures |
|-----------|--------|----------|
| accuracy | 30% | No hallucinations, correctly reflects Peat's views |
| grounding | 25% | Inline `[S1]`, `[S2]` citations that map to real sources |
| domain_fluency | 15% | Natural use of Peat vocabulary (bioenergetics, T3, PUFAs...) |
| completeness | 15% | Covers the question's key aspects, not superficial |
| attribution_style | 15% | Attributes to Peat explicitly, no filler openings |

### Track 2 — Automated metrics

No LLM needed, deterministic:

- **Citation count** — regex `\[S\d+\]` hits
- **Source coverage** — num sources returned vs. expected minimum
- **Vocab hit rate** — fraction of 25 Peat-specific terms present
- **Expected topic coverage** — fraction of question's `expected_topics` present
- **Source diversity** — unique source files / total sources
- **Top / average relevance** — Pinecone scores from the sources footer
- **Length sanity** — 150 ≤ word_count ≤ 400

## Baseline

Current baseline: **8.6/10** (from commit `ed84cf1`, "RAG chatbot quality overhaul —
7.0 to 8.6/10 avg score"). That score was computed ad-hoc and its exact question
set was not preserved, so this benchmark establishes a **new**, reproducible
baseline on top of the same target number.

The script exits with code 0 if the new run meets or beats the baseline,
1 if it regresses.

## Question categories

| Code | Category | # | What it tests |
|------|----------|---|---------------|
| A | core_bioenergetics | 6 | Foundational topics the RAG should nail |
| B | hormones_endocrine | 5 | Well-covered corpus area (57 files) |
| C | nutrition_foods | 5 | Specific-food retrieval (milk, OJ, coconut...) |
| D | disease_clinical | 5 | Multi-hop reasoning across papers + interviews |
| E | edge_ambiguous | 3 | Single-word queries — stress-tests HyDE expansion |
| F | edge_nuanced | 3 | Contradictory / nuanced — stress-tests cross-encoder |
| G | cross_concept | 3 | Synthesis across multiple concepts |

## History

| Date | Commit | Score | Notes |
|------|--------|-------|-------|
| 2026-03-xx | ed84cf1 | 8.6 | HyDE + two-pass Pinecone + MMR diversity penalty (original score, ad-hoc) |
| 2026-04-11 | (post cross-encoder) | **8.95** | + cross-encoder rerank (ms-marco-MiniLM-L-6-v2) + MMR float('-inf') fix. 29/30 judged (G3 lost to transient 503 before retry patch) |
