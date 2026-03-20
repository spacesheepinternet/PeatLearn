# PeatLearn — Global Rules

## What This Project Is
AI-powered adaptive learning platform built around Dr. Ray Peat's bioenergetic philosophy.
Domain: bioenergetic medicine, nutrition, hormonal science.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (`app/dashboard.py`) |
| RAG Backend | FastAPI, port 8000 (`app/api.py`) |
| ML Backend | FastAPI, port 8001 (`app/advanced_api.py`) |
| LLM | Google Gemini (`gemini-2.5-flash`, `gemini-2.5-flash-lite`) |
| Embeddings | `gemini-embedding-001`, 768 dimensions |
| Vector DB | Pinecone (index: `ray-peat-corpus`) |
| Local DB | SQLite (`data/user_interactions/interactions.db`) |
| Python | 3.12, venv at `venv/` |

---

## Entry Points

| File | Purpose |
|------|---------|
| `app/dashboard.py` | Main Streamlit dashboard |
| `app/api.py` | RAG FastAPI server (port 8000) |
| `app/advanced_api.py` | Advanced ML FastAPI (port 8001) |
| `peatlearn_master.py` | Thin backward-compat launcher → `app/dashboard.py` |
| `scripts/run_servers.py` | Launches all 3 servers at once |

---

## Package Structure

```
peatlearn/           ← importable package (project root is on PYTHONPATH)
  rag/               ← PineconeVectorSearch, PineconeRAG
  adaptive/          ← QuizGenerator, CorpusTopicModel, DataLogger, ContentSelector
  personalization/   ← PersonalizationEngine, RL agent, knowledge graph, neural layer
  embedding/         ← CorpusEmbedder, HuggingFace sync, Pinecone upload
  recommendation/    ← MF trainer (SGD, 32-dim)
app/                 ← FastAPI + Streamlit entry points
config/              ← settings.py (pydantic-settings, reads .env)
preprocessing/       ← pipeline, checkpoint system, cleaning modules
scripts/             ← utility runners
tests/
  unit/
  integration/
data/
  raw/               ← source xlsx, pdfs, txts
  processed/         ← AI-cleaned chunks
  embeddings/        ← local .npy/.pkl vector files
  models/            ← topic model, MF model artifacts
  user_interactions/ ← SQLite DB
```

---

## Launch Commands

```bash
# Activate venv first (Windows)
venv/Scripts/activate

# Individual services
streamlit run app/dashboard.py
uvicorn app.api:app --port 8000 --reload
uvicorn app.advanced_api:app --port 8001 --reload

# All at once
python scripts/run_servers.py
```

---

## Universal Conventions

### Code
- Always activate `venv/` before running Python; never install to system Python.
- Import from `peatlearn.*` package — never use `sys.path` hacks.
- `config/settings.py` is the single source of truth for settings; read via `from config.settings import settings`.
- Secrets live in `.env` — never hardcode API keys.

### LLM / Gemini
- Default generation model: `gemini-2.5-flash-lite` (fast, cheap).
- Upgrade to `gemini-2.5-flash` only when response quality matters (RAG answers, quiz generation).
- Embedding model: `gemini-embedding-001` (768-dim). Do not change dimensions without re-embedding the corpus.

### Pinecone
- Index name: `ray-peat-corpus`. Do not create new indices without explicit user instruction.
- `PineconeVectorSearch` has a SHA-256 hash fallback for offline use — do not remove it.

### Data
- Never mutate `data/raw/` — it is the source of truth.
- Processed outputs go to `data/processed/ai_cleaned/`.
- All model artifacts go under `data/models/`.

### Testing
- Unit tests in `tests/unit/`, integration tests in `tests/integration/`.
- No `sys.path` manipulation in test files.
- Run with `pytest tests/` from the project root.

### Git
- Branch from `main` for all features.
- Legacy RAG (file-based) is fully deprecated — do not resurrect it.

---

## Context Engineering (WISC)

This project uses a 3-tier context system. Do not load Tier 3 docs unless genuinely needed.

| Tier | Location | Load trigger |
|------|----------|-------------|
| 1 — Global rules | `CLAUDE.md` (this file) | Always loaded |
| 2 — Subsystem rules | `.claude/rules/*.md` | Auto-loaded by path patterns |
| 3 — Deep reference | `.claude/docs/*.md` | Sub-agent / on-demand only |

Slash commands: `/prime`, `/prime-rag`, `/prime-ml`, `/plan-feature`, `/handoff`
