# Handoff — 2026-05-17

## Current Branch & Git State
- Branch: `main` (synced with origin — `7ffa889` is on GitHub)
- Uncommitted changes: none in tracked files
- Untracked (carried from prior session — unchanged): 6 knowledge-graph files (`scripts/extract_graph_triples.py`, `scripts/init_graph_db.py`, `scripts/render_graph_preview.py`, `scripts/run_graph_extraction.py`, `scripts/store_graph_triples.py`, `data/knowledge_graph/`)
- Untracked from this session (gitignored): `council-report-20260516-214206.html`, `council-transcript-20260516-214206.md`
- Last 5 commits:
  - `7ffa889` docs: handoff for 2026-05-16 chatbot revival + 9.64 eval
  - `19b0f40` fix: revert 1.4-point quality regression — restore thinking + verifier
  - `3c5dcf4` fix: revive broken chatbot — wrong index + NameError + add auth
  - `6676c5f` fix: address professor's 3 architecture concerns
  - `41c9c72` feat: RAG hardening, quiz fixes, corpus expansion, Pinecone resilience

## What Was Being Worked On

Two threads carried over from yesterday's chatbot-revival session. Both about getting safely from "working locally at 9.64/10" to "live URL someone else can use."

**Thread 1 — Council review of the whole project.** Ran an LLM Council (5 advisors + peer review + chairman synthesis) on the entire project state. Output is two files at the project root: `council-report-20260516-214206.html` and `council-transcript-20260516-214206.md` (both gitignored). Every reviewer named the Contrarian as the sharpest voice and the Expansionist as the biggest blind spot. Chairman verdict: **ship Friday, not Monday — spend Mon–Tue on a safety net first.** The two things every advisor missed in round one (only surfaced in peer review): hard provider-side spend caps, and the Ray Peat estate / corpus copyright question.

**Thread 2 — Leaked API key incident.** GitHub Secret Scanning emailed about an exposed key in the last push. Investigation: a Gemini key (prefix `AIzaSyCL_q...`, REDACTED) is in commit `0e217b1` (initial commit, 2025-07-25, author `thewildofficial / abanstampy@gmail.com`). It is **not the user's key** — current local `.env` has a different key. `.env` is correctly in `.gitignore` and is not currently tracked. The push didn't introduce the leak; it just re-triggered the scanner against history that's been public for ~10 months. The user's wallet is not at risk from this leak — the old collaborator's is.

## Key Decisions Made

- **Don't rewrite git history to scrub the leaked key.** Once the collaborator revokes it, the value in history is worthless. Rewriting would force-push and break any clones, with no real security benefit. Option remains open if the user wants to silence future scanner emails.
- **Council recommendation accepted in spirit: ship Friday, not Monday.** Use Mon–Tue for safety net (spend caps, monitoring, disclaimer, rename), Wed for the auth-hole fix, Thu for 5 real Peat readers, Fri for one targeted public post.
- **Expansionist ideas (knowledge graph as viral product, "Daily Peat" Duolingo, adjacent-figure franchise) are explicitly v1.2 — not for launch.** Every peer reviewer flagged them as growth strategy poison before validation.
- **The leaked-key triage is the collaborator's problem to solve.** User's responsibility is limited to (a) notifying them, and (b) capping the user's own key.

## Active Plan
`C:\Users\rehan\.claude\plans\first-thing-i-want-wondrous-petal.md` — Scope 1 (private beta) selected and underway. The council shifted the order of operations within Scope 1: spend caps + monitoring + disclaimer **before** hosting, not after.

## What Needs to Happen Next

**TODAY (before closing the laptop):**
1. **Set hard spend caps on the user's own Gemini key** (`AIzaSyDAY9...`). Google Cloud Console → Billing → Budgets & alerts → $50/month with email alerts; APIs & Services → Quotas → "Generative Language API" requests/day → 5,000. Restrict the key to the Generative Language API only.
2. **Cap OpenRouter** (where Cohere rerank-4-pro lives). Buy $10–20 in pre-paid credits, auto-recharge OFF. When credits run out the cascade falls back to local MiniLM (eval B confirmed 9.42 with MiniLM-only — still shippable).
3. **Pinecone**: free Starter tier auto-caps at 100k vectors / limited QPS. Currently using 14,594 vectors. No action needed unless upgrading.
4. **Send the leaked-key email to `abanstampy@gmail.com`** — one sentence: "GitHub flagged a Gemini key from your initial commit on PeatLearn, you'll want to revoke it." Then mark task complete.

**MON–TUE:**
5. Add Sentry to `app/api.py` and `app/dashboard.py` (or simpler: free Uptime Robot ping every 5 min hitting `/api/health`).
6. Big red medical disclaimer at the top of the Streamlit chat. Suggested text in the council report.
7. Consider renaming "PeatLearn" → "Bioenergetic Q&A — inspired by Ray Peat's writings" to soften the estate/likeness risk the council flagged. Folder name can stay; this is product-name only.
8. Add a footer: "Corpus compiled from publicly available transcripts and articles. Contact us for takedown requests."

**WED:**
9. **Fix the auth hole** the Contrarian caught. The Streamlit dashboard talks to `peatlearn.rag.rag_system` in-process — it bypasses the bearer-token-protected FastAPI on port 8000. So the auth work added in commit `3c5dcf4` protects an endpoint nobody uses. Two clean options: (a) route the dashboard through `requests.post('http://localhost:8000/api/ask', headers={'Authorization': 'Bearer ...'})`, or (b) put Streamlit Community Cloud's built-in password protection in front of the whole app. Option (b) is faster.

**THU:**
10. Send the URL to 5 actual Peat readers (forums, Discord, Twitter DMs). Watch them break it. Take notes. Don't fix in real-time — let them finish.

**FRI:**
11. Public post in ONE Peat forum thread. Not Hacker News. Not Twitter mass-post.

**Optional polish (not blockers):**
- H1: populate `confidence_tier` / `confidence_reasons` in the `/api/ask` response (currently declared but never set, dashboard badge renders blank)
- H4: wrap `_call_gemini_llm` to return a clean 503 on missing key / outage instead of bare 500
- Add Pinecone/Gemini/OpenRouter keys to `.env.example` with placeholder values + docs
- Update `.claude/rules/rag.md` — it still says the reranker is "ms-marco-MiniLM-L-6-v2" but as of 41c9c72 the top of the cascade is Cohere rerank-4-pro

**Deferred to v1.0.1 or later:**
- Knowledge graph 3 pre-flight fixes + full 568-doc run (~$5 + 4h)
- Personalization stack training (needs real user data first)
- Chat-path integration tests
- Migrating `app/api.py` rate limiter from in-memory to Redis-backed for multi-worker
- Optional: `git filter-repo` to scrub the leaked key from history (only needed to silence future scanner emails — the key itself is worthless once revoked)

## Open Questions / Blockers

- **Did the old collaborator's leaked key get abused over the last 10 months?** Only they can check, in their Google Cloud Console billing history. Worth asking when you message them.
- **Hosting still not picked.** Streamlit Community Cloud is the path of least resistance and the chairman's pick. fly.io / Render / Railway / self-hosted VM all on the table but require Docker (which is broken — `docker-compose.yml` is stale from pre-March-2026 reorg).
- **6 untracked knowledge-graph files have been sitting for two sessions.** Decide: commit on a feature branch, leave untracked, or delete. Not blocking launch but the working tree is noisy.
- **Council flagged the corpus copyright question.** Did the user (or the original collaborator) have the right to redistribute `Ray Peat Anthology.xlsx` and the various PDFs in `data/raw/`? If not, that's a real takedown risk independent of the medical liability. No action specified yet — likely worth one hour of research.
- **The `_get_rag_system()._last_sources` pattern in `app/dashboard.py:1682`** works but is fragile — it relies on a side-effect attribute populated by the last call. If you ever go multi-user (more than one Streamlit session), this will return the wrong sources. Not blocking for private beta.

## Files Modified This Session

```
(no tracked code files modified this session — work was investigation + planning + external incident triage)

New artifacts at project root (gitignored):
  council-report-20260516-214206.html       — visual council verdict
  council-transcript-20260516-214206.md     — full transcript (5 advisors + 5 reviews + chairman)
  HANDOFF.md                                — this file
```

No commits this session. Next commit will likely be the disclaimer + safety-net work from Mon–Tue.
