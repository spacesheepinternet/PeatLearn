# Handoff — 2026-05-16

## Current Branch & Git State
- Branch: `main` (8 commits ahead of origin)
- Uncommitted changes: `HANDOFF.md` (this file) + 6 untracked knowledge-graph files from prior session (unchanged this session)
- Last 5 commits:
  - `19b0f40` fix: revert 1.4-point quality regression — restore thinking + verifier
  - `3c5dcf4` fix: revive broken chatbot — wrong index + NameError + add auth
  - `6676c5f` fix: address professor's 3 architecture concerns
  - `41c9c72` feat: RAG hardening, quiz fixes, corpus expansion, Pinecone resilience
  - `df33558` feat: eval 9.42/10 (+0.82 vs baseline); normalizer gaps from user queries

## What Was Being Worked On

**Launch readiness audit + emergency chatbot revival.** The session started as a "what's left before I can launch" architecture review and immediately uncovered that the chatbot had been completely broken in production. Two independent bugs since commit `6676c5f` were causing every dashboard chat query to return the canned "Sorry, technical issue" error message:

1. `app/api.py:22–23` hardcoded the legacy `ray-peat-corpus` rollback index instead of the active `ray-peat-corpus-v3` (14,594 vectors).
2. `peatlearn/adaptive/rag_system.py` called `logger.info()` at lines 414 and 620 but never imported `logging` — every query crashed with `NameError`. The 2026-05-05 eval recording 1.01/10 was originally misattributed to a Gemini outage; it was actually this bug.

After fixing those, the first clean eval came in at **8.04/10** — alive but down 1.38 points from the previous peak of 9.42 at commit `df33558`. Investigation found two more regression sources from commits `41c9c72` + `6676c5f`:

3. `"thinkingConfig": {"thinkingBudget": 0}` had been added as a cost-saver, disabling Gemini 2.5-flash's reasoning pass. Removed.
4. The grounding verifier was gated to LOW-confidence only on the theory that HIGH/MEDIUM had strong retrieval. In practice adversarial questions retrieve HIGH (the topic IS in the corpus, only the framing is false), so the verifier was being skipped exactly when it was needed. Restored to all tiers with the existing "keep original if revision is gutted" guard.

An A/B confirmed Cohere rerank-4-pro beats local ms-marco-MiniLM cross-encoder on this corpus (9.64 vs 9.42, with edge_nuanced 9.56 vs 6.17 the biggest gap). Cohere stays.

Also added the auth/rate-limit prep for a private-beta launch: bearer token (env-driven, off in dev) + per-IP in-memory rate limiter on `/api/ask`, `/api/search`, `/api/stats`, `/api/related`; env-driven `CORS_ORIGINS`.

**Final eval: 9.64/10 — a new all-time high.** (+1.04 vs baseline 8.6, +0.22 vs previous peak 9.42, +1.60 vs broken 8.04.)

## Key Decisions Made

- **Index name reads from `settings.PINECONE_INDEX_NAME`, not hardcoded.** Also hardened `PineconeRAG.__init__` default. Anyone constructing the RAG without args is now safe.
- **Cohere stays as top of reranker cascade.** A/B'd against MiniLM with thinking + verifier both restored: 9.64 vs 9.42. Skipped the HANDOFF item to swap MiniLM → MedCPT — Cohere already beats both, and MedCPT would only be the fallback's fallback.
- **`thinkingBudget` left unset on the main RAG call.** Cost saving is not worth ~1.4 eval points for a health-grounded chatbot. Saved to project memory.
- **Verifier runs on every tier**, not just LOW. The professor's coherence concern is handled by the pre-existing `len(revised.strip()) > 50` guard — if the verifier would gut the answer, we keep the original. Skipping the verifier entirely cost real quality on adversarial questions.
- **Auth is opt-in via env var** (`API_BEARER_TOKEN`). Unset means dev mode — no auth check. Production set the env var to enable. Rate limit uses in-memory per-IP token bucket; fine for single-worker private beta, upgrade to Redis-backed slowapi for multi-worker prod.
- **CORS made env-driven** but default still includes localhost (`:3000`, `:8000`, `:8501`) so local dev is unaffected.
- **Smoke test verified** via FastAPI TestClient: open `/api/health` returns 200, missing/wrong bearer token returns 401, rate limit boundary returns 429 with `Retry-After`.

## Active Plan
`C:\Users\rehan\.claude\plans\first-thing-i-want-wondrous-petal.md` — Scope 1 (private beta) was selected. Scope 1 coding is complete; hosting decision deferred.

## What Needs to Happen Next

**Launch prep — remaining Scope 1 items:**
1. **Pick a hosting target.** Options previously surfaced: Streamlit Community Cloud (free, fastest, bundles UI+backend), fly.io/Render/Railway (Docker required, proper separation), self-hosted VM (you write systemd/nginx). Note: the existing `docker-compose.yml` is stale from the pre-2026-03 reorg — references `inference/backend/Dockerfile` and `web_ui/frontend/` which no longer exist. Will need rewrite or deletion if going the Docker route.
2. **Set `API_BEARER_TOKEN` in prod env** and document the value somewhere the beta testers can find it.
3. **Smoke test end-to-end after deploy:** hit `/api/ask?question=what+causes+hypothyroidism` with bearer header, confirm sources + answer + status 200.

**Optional polish (H-tier from the launch audit, not blockers):**
4. **H1**: Populate `confidence_tier` + `confidence_reasons` in the `/api/ask` `QuestionResponse`. The dashboard badge at `app/dashboard.py:1548–1562` always renders empty tier because `peatlearn/rag/rag_system.py:96` only sets the float `confidence`. ~30 min.
5. **H3**: Add structured request logging in `app/api.py` so prod incidents are visible. ~1 h.
6. **H4**: Wrap `_call_gemini_llm` to return a clean 503 on missing key/outage instead of bare 500. ~20 min.

**Defer to v1.0.1 (not chatbot-launch scope):**
7. Knowledge graph: 3 pre-flight fixes (max_output_tokens, em-dash filename, l-glycine alias) + full 568-doc run (~$5, ~4 h). Not integrated into chatbot.
8. Personalization (RL agent, neural CF, MF recommender) — all scaffolded but untrained; ship in v1.0.1 once real user data accumulates.
9. Add chat-path integration tests (currently zero; `tests/integration/test_api.py` covers import + RAG init only).
10. Add Pinecone/Gemini keys to `.env.example` with docs.

## Open Questions / Blockers

- **No hosting target chosen.** Blocks deployment. Streamlit Community Cloud is the fastest path; pick one when ready.
- **Existing `docker-compose.yml` is unusable.** References pre-reorg paths. If going Docker, must rewrite or delete.
- **6 untracked knowledge-graph files from prior session.** `scripts/extract_graph_triples.py`, `scripts/init_graph_db.py`, `scripts/render_graph_preview.py`, `scripts/run_graph_extraction.py`, `scripts/store_graph_triples.py`, and `data/knowledge_graph/`. Decide whether to commit separately or stash until the full graph run happens.
- **`reject_premise` keyword-heuristic still reports ~30% defense rate** in the eval log — but this is the dumb keyword counter, not the LLM judge. The LLM judge scored all reject_premise questions ≥9.30 in eval A. The keyword heuristic likely needs updating to match the new (more nuanced) rejection phrasing. Not blocking, but worth tightening if you want the headline number to match reality.

## Files Modified This Session

```
app/api.py                            — B1 index fix; B4 bearer token + rate limit; CORS dependency wiring
config/settings.py                    — API_BEARER_TOKEN env var; CORS_ORIGINS env-driven (comma-separated)
peatlearn/adaptive/rag_system.py      — B5 logger import; thinkingBudget=0 removed; verifier restored to all tiers
peatlearn/rag/rag_system.py           — PineconeRAG default index_name now reads settings.PINECONE_INDEX_NAME
data/eval/results_20260516_181111.json — first post-B1+B5 eval (8.04, gitignored)
data/eval/results_20260516_190013.json — pre-regression-fix eval (8.04, gitignored)
data/eval/results_20260516_203512.json — Eval A: Cohere on + fixes (9.64 🏆, gitignored)
data/eval/results_20260516_205012.json — Eval B: Cohere off + fixes (9.42, gitignored)
HANDOFF.md                            — this file
```

Memory updates: `project_thinking_budget.md`, `project_reranker_choice.md`.
