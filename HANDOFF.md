# Handoff — 2026-03-22

## Current Branch & Git State
- Branch: `main`
- Uncommitted changes: 37 files (32 modified, 5 deleted, 2 new scripts)
- Last 5 commits:
  1. `552ab16` feat: major refactor — consolidate into peatlearn/ package + app/ entry points
  2. `1ee25a3` Delete GEMINI.md
  3. `9a6ff32` Final Presentation
  4. `19f72e5` feat: Comprehensive memorial page enhancement
  5. `a1c2ecf` Add MF recommender loader and blended recommendations

---

## What Was Being Worked On

**RAG chatbot quality testing and optimization** — completed all 10 quality test questions, scoring the chatbot across accuracy, source diversity, writing quality, completeness, and follow-up questions.

**Final scorecard: 8.6/10 average** (up from 7.0 before fixes).

| Q | Topic | Score |
|---|-------|-------|
| 1 | Thyroid & metabolism | 8/10 |
| 2 | Estrogen effects | 9/10 |
| 3 | PUFAs | 7/10 (was 4/10) |
| 4 | CO2 | 9/10 |
| 5 | Sugar & glucose | 8/10 |
| 6 | Cortisol & stress | 9/10 (was 1/10) |
| 7 | Serotonin | 9/10 |
| 8 | Progesterone | 9/10 |
| 9 | Light therapy | 9/10 (was 1/10) |
| 10 | Multi-topic synthesis | 9/10 |

**7 bugs/improvements fixed this session:**

1. **Groq API key added** (`.env`) — Groq fallback now active for Gemini rate limits.
2. **Gibberish classifier false rejections** (`app/dashboard.py`) — Q6 "cortisol and stress" and Q9 "light therapy and red light" were rejected because the 5+ consonant check ran on concatenated words. Fixed to check per-word.
3. **Source deduplication in display** (`app/dashboard.py`) — `_split_answer_and_sources()` now dedupes by filename and renumbers.
4. **Graceful port 8001 message** (`app/dashboard.py`) — Instead of ugly connection error, shows quiet caption when ML server is down.
5. **Content-type diversity penalty** (`peatlearn/adaptive/rag_system.py`) — MMR now penalizes repeated content types (transcripts, papers, etc.) at -0.05 per same-type source.
6. **Two-pass Pinecone query** (`peatlearn/adaptive/rag_system.py`) — Pass 1 fetches 80 results with 4-chunk-per-file cap; Pass 2 excludes dominant files and fetches 40 more. Jumped PUFA query from 4 to 25 unique source files.
7. **Completeness improvements** (`peatlearn/adaptive/rag_system.py`) — Word count bumped (80-130 to 120-180), added prompt rule to surface practical recommendations.

**Data quality fix — deduplicated 22 bloated processed files:**
- AI cleaning had duplicated content (up to 403x!) in 22 transcript/publication files
- Ran `scripts/dedup_processed_files.py` — removed 5,343 duplicate blocks, saved 2.4 MB
- Deleted 4,380 old vectors from Pinecone
- Re-embedded 720 of 840 pairs (93%) — 2 files still missing due to Gemini daily quota

---

## Key Decisions Made
- **Per-word gibberish check** — concatenated-string approach caused false positives on valid multi-word queries
- **Two-pass Pinecone query** — single-pass couldn't overcome files with 50+ identical-score chunks
- **Content-type diversity at -0.05** — light enough to not force irrelevant sources, strong enough to break transcript dominance
- **Dedup at display level** — RAG still sends 2 chunks per file to the LLM for context, but display deduplicates filenames
- **Gemini embedding quota is 1,000/day free tier** — need to pace re-embedding work across sessions

---

## Active Plan
`parallel-marinating-pearl.md` — session summary and remaining tasks

---

## What Needs to Happen Next

1. **Finish re-embedding 2 missing files** (9 pairs) — run `python scripts/reembed_deduped_files.py --embed-upload` after Gemini quota resets. Missing files:
   - `kmud-160715-the-metabolism-of-cancer.mp3-transcript_processed.txt` (4 blocks)
   - `09.21.21 Peat Ray [1128846532].mp3-transcript_processed.txt` (5 blocks)
2. **Clean up duplicate vectors** from the second embed run — the script ran twice, creating duplicates for some files
3. **Commit all pending changes** — 37 files unstaged
4. **Migrate `quiz_generator.py`** from deprecated `google.generativeai` to `google.genai` SDK
5. **Investigate Pinecone dimension mismatch** — index reports 3072, config says 768; embeddings are zero-padded to 3072 on upload

---

## Open Questions / Blockers
- **Gemini embedding daily quota exhausted** — 1,000 req/day free tier. Resets tomorrow.
- **Pinecone dimension mismatch** — 3072 vs 768. Functional (zero-padded) but wasteful and confusing.
- **3 bloated files not deduped** — `jf-190427-stress-health.mp3`, `2005 - November`, and `Dr. Ray Peat Day One` are bloated (>200%) but have no duplicate `RAY PEAT:` blocks — different bloat cause, needs investigation.
- **`torch_geometric` not installed** — advanced ML personalization runs degraded.

---

## Files Modified This Session

```
app/dashboard.py                    |  38 changes (gibberish fix, source dedup, port 8001 message)
peatlearn/adaptive/rag_system.py    | 275 changes (two-pass query, content-type diversity, prompt improvements)
config/settings.py                  |   8 changes (GROQ_API_KEY)
.env                                |   2 changes (GROQ_API_KEY value added)
scripts/dedup_processed_files.py    | new (dedup bloated processed files)
scripts/reembed_deduped_files.py    | new (delete/re-embed/upload Pinecone vectors)
data/processed/ai_cleaned/...       |  22 files deduped (2.4 MB of duplicates removed)
data/artifacts/vectors_to_delete.json | new (scan results for Pinecone cleanup)
```

## Servers Running
- Port 8000: RAG FastAPI — running
- Port 8001: Advanced ML FastAPI — running
- Port 8501: Streamlit dashboard — NOT running (killed for restart, needs `streamlit run app/dashboard.py`)
