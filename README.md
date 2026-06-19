<div align="center">

# 🧬 PeatLearn

**An AI-powered adaptive learning platform built around Dr. Ray Peat's bioenergetic philosophy.**

Ask grounded questions, take quizzes that adapt to you, and get personalized recommendations — all backed by a retrieval-augmented corpus of Ray Peat's life work.

<br>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)
![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-000000)
![RAG Score](https://img.shields.io/badge/RAG%20Benchmark-9.64%2F10-success)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Corpus & Data Pipeline](#corpus--data-pipeline)
- [API Reference](#api-reference)
- [RAG Quality Benchmark](#rag-quality-benchmark)
- [Testing](#testing)
- [Acknowledgments](#acknowledgments)

---

## Overview

PeatLearn turns a large archive of Dr. Ray Peat's transcripts, papers, newsletters, and health
writings into an interactive learning experience. A retrieval-augmented generation (RAG) engine
answers questions with grounded, citation-backed responses, while an adaptive ML layer tailors
quizzes and recommendations to each learner.

The domain is **bioenergetic medicine, nutrition, and hormonal science** — a health-critical
context, so the system is built to ground every claim in the source corpus rather than improvise.

---

## Features

| Feature | Description |
|---------|-------------|
| 💬 **RAG Chatbot** | Ask questions about Ray Peat's work; answers are retrieved from 22,457 embedded passages with inline citations (benchmark avg **9.64/10**). |
| 🧠 **Adaptive Quizzes** | Gemini-powered quizzes that scale difficulty to your demonstrated knowledge level. |
| 🗂️ **Topic Browser** | A TF-IDF + KMeans topic model that auto-selects 12–36 clusters across the corpus. |
| 🎯 **Personalized Recommendations** | Matrix factorization (SGD, 32-dim) combined with a reinforcement-learning content selector. |
| 👤 **Learning Profiles** | AI-generated learner profiles backed by a concept knowledge graph. |

---

## Quick Start

```bash
# 1. Activate the virtual environment
venv\Scripts\activate          # Windows (PowerShell / CMD)
source venv/Scripts/activate   # Git Bash

# 2. Create your environment file and add API keys
cp config/env_template.txt .env

# 3. Launch all services at once
python scripts/launch/run_servers.py
```

Or launch each service individually:

```bash
streamlit run app/dashboard.py             # Dashboard      → http://localhost:8501
uvicorn app.api:app --port 8000 --reload   # RAG backend    → http://localhost:8000
uvicorn app.advanced_api:app --port 8001   # ML backend     → http://localhost:8001
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

```
                        ┌──────────────────────────┐
                        │   Streamlit Dashboard     │   app/dashboard.py  (:8501)
                        └────────────┬──────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                          ▼
   ┌──────────────────────────┐              ┌──────────────────────────┐
   │   RAG Backend (FastAPI)  │              │   ML Backend (FastAPI)   │
   │   app/api.py     (:8000) │              │ app/advanced_api.py(:8001)│
   └────────────┬─────────────┘              └────────────┬─────────────┘
                │                                         │
       ┌────────┴────────┐                    ┌───────────┴───────────┐
       ▼                 ▼                    ▼                       ▼
  ┌─────────┐      ┌──────────┐        ┌────────────┐         ┌──────────────┐
  │ Pinecone│      │  Gemini  │        │ Topic Model│         │ SQLite        │
  │ vectors │      │   LLM    │        │ MF / RL    │         │ interactions  │
  └─────────┘      └──────────┘        └────────────┘         └──────────────┘
```

---

## Project Structure

```
peatlearn/               ← importable package (project root on PYTHONPATH)
  rag/                   ← PineconeVectorSearch, PineconeRAG
  adaptive/              ← QuizGenerator, CorpusTopicModel, ContentSelector
  personalization/       ← PersonalizationEngine, RL agent, knowledge graph
  embedding/             ← CorpusEmbedder, HuggingFace sync
  recommendation/        ← Matrix factorization trainer
app/                     ← FastAPI + Streamlit entry points
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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (`app/dashboard.py`) |
| RAG Backend | FastAPI · port 8000 (`app/api.py`) |
| ML Backend | FastAPI · port 8001 (`app/advanced_api.py`) |
| LLM | Google Gemini (`gemini-2.5-flash`, `gemini-2.5-flash-lite`) |
| Embeddings | `gemini-embedding-001` · 3072 dimensions |
| Vector DB | Pinecone · index `ray-peat-corpus-v3` · 22,457 vectors |
| Local DB | SQLite (`data/user_interactions/interactions.db`) |
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

## API Reference

### RAG service · `:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question; returns a grounded RAG answer with citations |
| `GET`  | `/search` | Semantic search over the corpus |
| `GET`  | `/health` | Health check |

### ML service · `:8001`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/quiz/generate` | Generate an adaptive quiz |
| `POST` | `/profile/analyze` | Generate a user learning profile |
| `GET`  | `/recommendations` | Personalized content recommendations |

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

## Acknowledgments

- **Dr. Ray Peat** — for his pioneering work in bioenergetic medicine.
- **The Ray Peat community** — researchers and enthusiasts who keep his ideas alive.

<div align="center">

<br>

*"Energy and structure are interdependent at every level."*
— Ray Peat

</div>
