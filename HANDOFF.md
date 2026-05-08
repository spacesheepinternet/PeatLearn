# Handoff — 2026-05-05

## Current Branch & Git State
- Branch: `main` (5 commits ahead of origin/main — not pushed)
- Uncommitted changes: 11 files modified, 1 file deleted, 5 new files (script, council reports, processed files, staging dir)
- Last 5 commits:
  - `df33558` feat: eval 9.42/10 (+0.82 vs baseline); normalizer gaps from user queries
  - `144c8b5` fix: gate HyDE behind _HYDE_ENABLED flag — stop burning embedding quota
  - `c3cdcff` fix: add cruciferous/broccoli/kale normalizer entries → surface goitrogen chunks
  - `598b2cc` feat: v3 corpus pipeline, A/B eval, embed_query routing, cross-encoder rerank
  - `057580e` exp: disable HyDE — raw query embedding only (9.25/10, stabilizes B4/F1/E3)

## What Was Being Worked On

**Three threads ran in parallel this session:**

**1. Preprocessing audit + missing-files recovery (completed):** An Explore agent audited `data/processed/ai_cleaned/` and flagged a +94 raw-vs-processed file count discrepancy. Investigation revealed `preprocessing/optimized_pipeline.py:67` had `extensions = ['.txt', '.pdf', '.docx', '.md', '.json']` — **no `.html`**. This silently dropped 16 HTML files (entire `09_Miscellaneous/` directory + `02_Publications/Articles/` subdir). Plus the Generative Energy book PDF in `new_content_2026/` was never scanned because the pipeline was invoked with `input_dir=data/raw/raw_data/`, not `data/raw/`. Wrote `scripts/process_missing_files.py` that strips HTML (BeautifulSoup) / extracts PDF (pdfplumber) → feeds clean text to existing `EnhancedSignalProcessor` → writes marker-format chunks. Processed all 17 files. Rebuilt corpus_v3 → 14,407 → 14,591 chunks (+184). Re-embedded entire corpus to Pinecone (took 100 min — see "Active Issues" below). Discovered `EnhancedSignalProcessor` has a real bug: references `self.ai_model` which is never set in constructor (constructor sets `self.client`); patched in `process_missing_files.py` with `processor.ai_model = processor.client`.

**2. Council on RAG strategies (completed):** Ran the LLM Council skill on which of 11 ottomator-agents RAG strategies to add. Unanimous: Multi-Query RAG and Hierarchical RAG top two; Agentic RAG and Knowledge Graphs unanimously dropped. **All 5 peer reviewers independently flagged the same blind spot — the 50% premise rejection rate had never been audited.** Reports saved to `council-report-rag-strategies-20260504.html` + `council-transcript-rag-strategies-20260504.md`.

**3. Confidence gate audit (completed, surprising result):** The Council's Phase 0 prescription was to audit `confidence.py`. Audit findings: **(a) the gate fired ABSTAIN 0/55 times in eval — works as designed, (b) all 5 actual abstentions came from the separate `temporal_guard` (Ozempic, CGM, etc.), (c) the "weak answers" suspected of false confidence were actually silent empty responses from the verifier bug already patched this session.** Re-ran A4 sugar, C2 OJ, C3 coconut oil, D1 diabetes live — all four now return real grounded answers (549–1130 chars) instead of empty strings. Council's prescribed audit revealed the audit itself was the wrong frame; symptoms had a different root cause.

**4. Verifier-guard fix verified end-to-end:** The "if revised_answer < 50 chars, keep original" guard added at the start of session works — H6 keto reject_premise spot-check returns proper rejection answer. Subset H eval improved 0/10 → 5/10 reject_premise correct after also strengthening the prompt instruction (added "CRITICAL — False premise check" with affirmative-support rule).

**5. Dashboard & API resilience (in progress, blocked):** Pinecone API has been unreachable from this network for 30+ minutes. Wrapped `RayPeatRAG()` init in `try/except` in three places (`init_adaptive_system`, `_get_rag_system` cache, `app/advanced_api.py`) so quizzes/profile features keep working when Pinecone is down. Dashboard now shows friendly "chat unavailable" message instead of crashing. **Advanced API (port 8001) still fails to start** because `peatlearn/rag/vector_search.py:396` instantiates `PineconeVectorSearch()` at module import — exception propagates before the try/except wrapper can catch it. Need to remove that module-level instantiation or wrap it.

## Key Decisions Made

- **Real baseline remains 7.13/10:** Couldn't run a fresh eval to measure today's cumulative gains because Pinecone is down. Expected uplift from today's fixes: +0.5–1.0 (verifier guard alone fixes 4 zero-scoring queries → 7+).
- **Confidence gate doesn't need recalibration:** Audit revealed the gate works correctly given its design (only ABSTAINs on weak retrieval). The Council's prescription to audit was based on a misdiagnosis.
- **Drop transcription-gap chunks rather than repair them:** 16 chunks with `________________` placeholders deleted from Pinecone (14,610 → 14,594). Permanent `_{8,}` filter added to `build_corpus_v3.py` quality_gate so they can't return on rebuild.
- **CORRUPTED file is non-issue:** The `*_processed.txt` glob already excludes `.txt.CORRUPTED`. The original audit was wrong on this point. 0 chunks from it in active corpus.
- **17 missing files handled with full AI cleaning:** Used existing `EnhancedSignalProcessor` (Gemini 2.5 Flash Lite) for consistency with rest of corpus, not a faster bs4-only path. Cost: ~$0.06 for all 17 files.
- **Eval health check added:** `scripts/eval_rag_quality.py` now `sys.exit(2)` if Pinecone unreachable or index has < 1000 vectors. Today's failed eval (entire 55 questions scored 1.00 with no failure signal) proved Priority 2 #5 is real and dangerous — the silent fallback to SHA-256 hash embeddings produces garbage that looks like a model regression.

## Active Plan
None (no `.claude/plans/` directory).

## What Needs to Happen Next

1. **Wait for Pinecone to come back** — `api.pinecone.io:443` still timing out. `generativelanguage.googleapis.com` and `google.com` both reachable, so it's Pinecone-specific. Could be a Pinecone outage or local network/firewall issue. Retry `socket.create_connection(('api.pinecone.io', 443), timeout=10)` periodically.
2. **Fix `peatlearn/rag/vector_search.py:396`** — remove `search_engine = PineconeVectorSearch()` at module level, or wrap in try/except. This is what's blocking the advanced API server (port 8001) from starting when Pinecone is down. Quizzes need port 8001.
3. **Run fresh full eval** — once Pinecone is back, run `python scripts/eval_rag_quality.py` to measure cumulative uplift from: verifier guard, prompt strengthening, 17 new files (Generative Energy book + 6 interviews + Ray Peat's Brain etc.), 16 transcription-gap chunks removed.
4. **Commit the work** — 11 modified files + new script + council reports + 17 new processed files. Suggested split: (a) preprocessing fix + missing-files recovery, (b) verifier guard + prompt strengthening, (c) confidence audit findings + eval health check, (d) dashboard/API resilience, (e) corpus quality filter.
5. **Fix `embed_and_upload_v3.py --resume`** — the checkpoint is keyed by JSONL filename. Every new corpus build → new JSONL → new (empty) checkpoint → re-embed all 14k chunks. Today this cost 100 minutes when it should have taken 30 seconds. Either diff against Pinecone or use a stable checkpoint name.
6. **Council Phase 1 — Multi-Query RAG** — once eval baseline is re-established, implement constrained reformulations through `query_normalizer.py` (NOT free-form LLM). Should crush nutrition_foods category (5.69) by bridging colloquial-to-clinical phrasing.
7. **Same-content dedup** — newly discovered: the Estrogen-Age Stress Hormone HTML in `09_Miscellaneous/` is the same article as `04_Health_Topics/.../estrogen-and-brain-aging-in-men-and-women...`. Now duplicated in corpus. Dedup is exact-match on `source_file + context[:100]`; needs content-hash variant.
8. **Council Phase 2 — Hierarchical RAG** — needs `parent_doc_id`, `position_in_doc`, `section` metadata on chunks. Currently absent. Either backfill via Pinecone metadata-only update or accept it as re-indexing work.

## Open Questions / Blockers

- **Pinecone connectivity** — sustained outage from this network. Can't test anything live, can't run eval. No ETA. Blocks: fresh eval, dashboard chat, advanced API quizzes.
- **Why module-level `search_engine = PineconeVectorSearch()` in `vector_search.py:396`?** — looks like a backward-compat singleton. Removing it breaks any callers that import it directly. Need to grep before deleting; safer to wrap in try/except.
- **Cumulative impact of today's fixes is unmeasured.** Spot checks confirm the 4 silent-empty-response queries now answer correctly, premise rejection improved 0→5/10 on subset H, but no full-corpus number to anchor against the 7.13 baseline.
- **Item bank is thin.** Only 6 quiz items in SQLite. With Pinecone down, no new items can be seeded. Quizzes for new topics will return "No quiz items available" 400 errors until Pinecone returns.
- **`council-report-*.html` and `council-transcript-*.md` files** clutter the repo root (10 of each). Should be moved to `.claude/council/` or `data/artifacts/` and gitignored or organized.

## Files Modified This Session

```
HANDOFF.md                                                                | regenerated
app/advanced_api.py                                                       | RayPeatRAG init wrapped in try/except (rag_system=None on Pinecone failure)
app/dashboard.py                                                          | init_adaptive_system + _get_rag_system both wrap RayPeatRAG; chat handler shows "chat unavailable" friendly message; full document reader added to source expanders (st.text_area, cached _load_full_document)
config/settings.py                                                        | (carried from prior session — OPENROUTER_API_KEY field)
data/processed/ai_cleaned/01_Audio_Transcripts/Other_Interviews/sourcenutritional-...-2.txt | DELETED (was already CORRUPTED, reflagged)
peatlearn/adaptive/rag_system.py                                          | strengthened false-premise prompt instruction (CRITICAL pre-check + affirmative-support rule for partial-match traps)
peatlearn/rag/confidence.py                                               | (carried — no new edits)
peatlearn/rag/query_normalizer.py                                         | (carried)
peatlearn/rag/reranker.py                                                 | (carried — Cohere timeout fix)
preprocessing/optimized_pipeline.py                                       | added '.html' to extensions list (line 67)
scripts/build_corpus_v3.py                                                | quality_gate now drops chunks containing _{8,} (transcription-gap placeholders)
scripts/eval_rag_quality.py                                               | health check after RayPeatRAG init: aborts with sys.exit(2) if search_engine is None or index has <1000 vectors
```

```
NEW FILES (untracked):
scripts/process_missing_files.py                                          | 17-file recovery script (HTML/PDF → AI clean → marker format)
data/processed/ai_cleaned/09_Miscellaneous/*.txt                          | 10 newly processed files (Estrogen-Age Stress, Ray Peat's Brain Pt I/II, etc.)
data/processed/ai_cleaned/02_Publications/Articles/*.txt                  | 6 newly processed Peat interview files
data/processed/ai_cleaned/02_Publications/Books/generative-energy-...-life_processed.txt | Generative Energy book (Peat's published book, 166KB cleaned)
data/_missing_files_staging/                                              | intermediate clean-text staging (can be gitignored or deleted)
data/corpus_v3/corpus_v3_20260504_211041.jsonl + summary                  | 14,591-chunk regenerated corpus (untracked — Priority 3 #11)
council-report-rag-strategies-20260504.html                               | council HTML report on RAG strategies
council-transcript-rag-strategies-20260504.md                             | council full transcript
```

## Servers Running
- Streamlit dashboard: http://localhost:8501 (running, restarted this session, RAG-resilient)
- FastAPI RAG (port 8000): not running
- FastAPI ML (port 8001): not running (failed to start due to Pinecone module-level instantiation in vector_search.py:396)

## Pinecone State
- Index: `ray-peat-corpus-v3`, 3072-dim
- Vector count: **14,594** (was 14,398 → +212 from new files re-embed → −16 transcription-gap deletes)
- Currently unreachable from this network (TimeoutError, sustained 30+ min)
