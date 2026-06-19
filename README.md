<div align="center">

# 🧬 PeatLearn

**A grounded, citation-backed AI chatbot for exploring Dr. Ray Peat's bioenergetic work.**

Ask questions in plain language and get answers retrieved from a curated corpus of Ray Peat's
transcripts, papers, newsletters, and health writings — with inline citations and source documents.

<br>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)
![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-000000)
![RAG Score](https://img.shields.io/badge/RAG%20Benchmark-9.64%2F10-success)

[**Live app → peatlearn.streamlit.app**](https://peatlearn.streamlit.app)

</div>

---

## Table of Contents

- [Overview](#overview)
- [What Ships](#what-ships)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Corpus & Data Pipeline](#corpus--data-pipeline)
- [RAG Quality Benchmark](#rag-quality-benchmark)
- [Testing](#testing)
- [In the Codebase (Not Shipped)](#in-the-codebase-not-shipped)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## Overview

PeatLearn turns a large archive of Dr. Ray Peat's recorded and written work into an interactive,
grounded chatbot. A retrieval-augmented generation (RAG) pipeline answers questions using only the
source corpus, with inline citations and the underlying documents one click away.

The domain is **bioenergetic medicine, nutrition, and hormonal science** — a health-critical
context, so the system is built to ground every claim in the corpus and to **abstain** when the
corpus doesn't support an answer, rather than improvise.

---

## What Ships

The deployed app is a single Streamlit dashboard with **two tabs**:

| Tab | Description |
|-----|-------------|
| 💬 **Chat** | Ask questions about Ray Peat's work. Answers run through the full multi-stage RAG pipeline (below), are returned with inline citations and relevance-scored sources, and each source has a "Read full document" expander. Benchmark avg **9.64/10**. |
| 🕊️ **Memorial** | A tribute page honoring Dr. Ray Peat. |

> Other components (quizzes, recommender, personalization, knowledge graph, standalone FastAPI
> backends) exist in the repository but are **not wired into the live app** — see
> [In the Codebase (Not Shipped)](#in-the-codebase-not-shipped).

---

## Quick Start

The live app runs a single Streamlit process:

```bash
# 1. Activate the virtual environment
venv\Scripts\activate          # Windows (PowerShell / CMD)
source venv/Scripts/activate   # Git Bash

# 2. Create your environment file and add API keys
cp config/env_template.txt .env

# 3. Run the dashboard
streamlit run app/dashboard.py   # → http://localhost:8501
```

---

## Setup

### Prerequisites

- Python **3.12**
- A **Google Gemini** API key
- A **Pinecone** API key

### Installation

```bash
git clone <repository-url>
cd PeatLearn

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```bash
cp config/env_template.txt .env
```

Then edit `.env`:

```ini
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional — enables the Cohere rerank-4-pro retrieval reranker (recommended in prod).
# Without it, retrieval falls back to the local cross-encoder.
OPENROUTER_API_KEY=your_openrouter_api_key
```

`config/settings.py` (pydantic-settings) is the single source of truth for configuration and reads
these values from `.env`. **Never hardcode API keys.**

### Embeddings

The Pinecone index (`ray-peat-corpus-v3`) is pre-populated with **22,457** native 3072-dim vectors,
so no local embedding setup is required to run the app.

To pull the local embedding artifacts (optional), set `HF_DATASET_REPO` in `.env` and run:

```bash
python peatlearn/embedding/hf_download.py
```

---

## Architecture

The deployed app is Streamlit-only — `app/dashboard.py` calls the RAG pipeline in
`peatlearn/adaptive/rag_system.py` directly (no separate backend service in production).

```
   ┌────────────────────────────┐
   │   Streamlit Dashboard      │   app/dashboard.py  (Chat · Memorial)
   └─────────────┬──────────────┘
                 │
                 ▼
   ┌────────────────────────────────────────────────────────────┐
   │   RAG pipeline  (peatlearn/adaptive/rag_system.py)          │
   │                                                            │
   │   query normalize → temporal guard → citation gate →       │
   │   Pinecone two-pass retrieval → reranker → MMR diversity → │
   │   confidence tiers + entity grounding → grounding verifier │
   └─────────────┬───────────────────────────────┬──────────────┘
                 ▼                               ▼
          ┌─────────────┐                 ┌──────────────┐
          │  Pinecone   │                 │   Gemini     │
          │  (vectors)  │                 │ (→ Groq      │
          │             │                 │  fallback)   │
          └─────────────┘                 └──────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (`app/dashboard.py`) |
| RAG pipeline | `peatlearn/adaptive/rag_system.py` (called in-process) |
| LLM | Google Gemini (`gemini-2.5-flash`, `gemini-2.5-flash-lite`), Groq fallback |
| Embeddings | `gemini-embedding-001` · 3072 dimensions |
| Reranker | Cohere `rerank-4-pro` (via OpenRouter) → local cross-encoder fallback |
| Vector DB | Pinecone · index `ray-peat-corpus-v3` · 22,457 vectors |
| Language | Python 3.12 |

---

## Corpus & Data Pipeline

The corpus draws from **552 source documents** spanning Ray Peat's recorded and written work:

| Type | Count |
|------|------:|
| Audio transcripts | 188 |
| Academic papers | 96 |
| Health topics | 98 |
| Newsletters | 59 |
| Other | 111 |
| **Total** | **552** |

These are processed into **22,457 QA pairs**, embedded at 3072 dimensions, and stored in Pinecone.

```
data/raw/  →  preprocessing/cleaning/  →  data/processed/ai_cleaned/
           →  peatlearn/embedding/      →  Pinecone (ray-peat-corpus-v3)
```

- **Tier 1 (~27%)** — rules-based cleaning for already-clean documents.
- **Tier 2 (~73%)** — AI-powered cleaning: OCR correction, speaker attribution, and segmentation.

---

## RAG Quality Benchmark

The chatbot is evaluated against a fixed **30-question benchmark** with dual scoring:
LLM-as-judge (Gemini 2.5-flash on a 5-dimension rubric) **plus** automated metrics
(citations, vocabulary hit rate, source diversity, and topic coverage).

**Retrieval pipeline:** queries run through HyDE expansion → two-pass Pinecone retrieval →
a tiered reranker → MMR diversity → confidence-gated abstention. The reranker tries
**Cohere `rerank-4-pro`** (via OpenRouter) first, then falls back to a local cross-encoder
(`peat-reranker-ft` if present, otherwise `ms-marco-MiniLM-L-6-v2`), and finally to keyword overlap.

```bash
python scripts/eval/eval_rag_quality.py               # full 30-question run
python scripts/eval/eval_rag_quality.py --subset A,B  # only specific categories
python scripts/eval/eval_rag_quality.py --no-judge    # automated metrics only
```

The question set lives in `data/eval/questions.json`; results are written to
`data/eval/results_<timestamp>.json`. See `data/eval/README.md` for the full rubric.

### Score history

| Date | Score | Notes |
|------|------:|-------|
| commit `ed84cf1` | 8.60 / 10 | Baseline — HyDE + two-pass Pinecone + MMR diversity |
| 2026-04-11 | 8.95 / 10 | +0.35 — cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) + MMR fix |
| 2026-04-14 | 9.05 / 10 | +0.10 — dynamic `max_sources` heuristic + three-tier prompt depth |
| 2026-05-16 | **9.64 / 10** | +0.59 — swapped reranker to **Cohere `rerank-4-pro`** (A/B win over local MiniLM, 9.64 vs 9.42) |

**Per-category & per-dimension breakdown** (from the 9.05 judged run, 30/30, pre-Cohere reranker):

| Category | Score | | Rubric dimension | Score |
|----------|------:|---|------------------|------:|
| core_bioenergetics | 9.11 | | accuracy | 9.47 |
| disease_clinical | 9.20 | | grounding | 9.18 |
| cross_concept | 9.07 | | attribution_style | 9.05 |
| hormones_endocrine | 9.05 | | domain_fluency | 8.72 |
| edge_ambiguous | 9.05 | | completeness | 8.37 |
| edge_nuanced | 9.03 | | | |
| nutrition_foods | 8.86 | | | |

Automated metrics: **source diversity 0.91** · expected-topic coverage 0.76 ·
100% of answers returned ≥ expected sources · avg **5.3 inline citations** per answer.

---

## Testing

```bash
pytest tests/              # all tests
pytest tests/unit/         # unit tests only
pytest tests/integration/  # integration tests only
```

Run from the project root. Tests import from the `peatlearn.*` package — no `sys.path` hacks.

---

## In the Codebase (Not Shipped)

The repository contains additional components that are **not part of the live app**. They are kept
for local development and future work — do not treat them as current features:

- **FastAPI backends** — `app/api.py` (RAG, port 8000) and `app/advanced_api.py` (ML, port 8001).
  Useful for local development; the production deploy runs `app/dashboard.py` directly without them.
- **Adaptive quizzes** — `QuizGenerator` exists but is not wired into the UI (Quiz tab parked).
- **Personalized recommendations** — matrix factorization recommender and RL content selector exist
  as code/artifacts, not user-facing.
- **Learning profiles / analytics** — parked tabs.
- **Topic model** — TF-IDF + KMeans clustering over the corpus, not surfaced in the live UI.
- **Knowledge graph** — concept-map work, parked.

---

## Project Structure

```
peatlearn/               ← importable package (project root on PYTHONPATH)
  rag/                   ← PineconeVectorSearch, PineconeRAG, reranker, confidence
  adaptive/              ← rag_system.py (live RAG pipeline) + parked: QuizGenerator, topic model
  personalization/       ← engine, RL agent, knowledge graph  (not shipped)
  embedding/             ← CorpusEmbedder, HuggingFace sync
  recommendation/        ← matrix factorization trainer       (not shipped)
app/
  dashboard.py           ← live Streamlit app (Chat · Memorial)
  api.py / advanced_api.py ← FastAPI backends (local dev only)
config/                  ← settings.py (pydantic-settings, reads .env)
preprocessing/           ← cleaning pipeline + quality analysis
scripts/                 ← utility runners (launch, setup, eval)
tests/
  unit/                  ← unit tests
  integration/           ← integration tests
data/
  raw/                   ← source xlsx, pdfs, txts (source of truth — never mutate)
  processed/             ← AI-cleaned chunks
  embeddings/            ← local .npy/.pkl vector files
  models/                ← topic model & MF model artifacts
  user_interactions/     ← SQLite DB
```

---

## Acknowledgments

- **Dr. Ray Peat** — for his pioneering work in bioenergetic medicine.
- **The Ray Peat community** — researchers and enthusiasts who keep his ideas alive.

<div align="center">

<br>

*"Energy and structure are interdependent at every level."*
— Ray Peat

</div>
