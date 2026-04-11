# PeatLearn

AI-powered adaptive learning platform built around Dr. Ray Peat's bioenergetic philosophy. Features a RAG chatbot, adaptive quizzes, personalized recommendations, and a full ML backend.

---

## Quick Start

```bash
# 1. Activate venv
venv\Scripts\activate          # Windows
source venv/Scripts/activate   # Git Bash

# 2. Copy and fill in API keys
cp config/env_template.txt .env

# 3. Launch everything
python scripts/run_servers.py
```

Or launch individually:

```bash
streamlit run app/dashboard.py             # Dashboard (port 8501)
uvicorn app.api:app --port 8000 --reload   # RAG backend
uvicorn app.advanced_api:app --port 8001   # Advanced ML backend
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (`app/dashboard.py`) |
| RAG Backend | FastAPI, port 8000 (`app/api.py`) |
| ML Backend | FastAPI, port 8001 (`app/advanced_api.py`) |
| LLM | Google Gemini (`gemini-2.5-flash`, `gemini-2.5-flash-lite`) |
| Embeddings | `gemini-embedding-001`, 3072 dimensions |
| Vector DB | Pinecone (index: `ray-peat-corpus`, 22,457 vectors) |
| Local DB | SQLite (`data/user_interactions/interactions.db`) |
| Python | 3.12 |

---

## Project Structure

```
peatlearn/               ← importable package
  rag/                   ← PineconeVectorSearch, PineconeRAG
  adaptive/              ← QuizGenerator, CorpusTopicModel, ContentSelector
  personalization/       ← PersonalizationEngine, RL agent, knowledge graph
  embedding/             ← CorpusEmbedder, HuggingFace sync
  recommendation/        ← Matrix factorization trainer
app/                     ← FastAPI + Streamlit entry points
config/                  ← settings.py (pydantic-settings, reads .env)
preprocessing/           ← cleaning pipeline + quality analysis
scripts/                 ← utility runners
tests/
  unit/
  integration/
data/
  raw/                   ← source xlsx, pdfs, txts (do not mutate)
  processed/             ← AI-cleaned chunks
  embeddings/            ← local .npy/.pkl vector files
  models/                ← topic model, MF model artifacts
  user_interactions/     ← SQLite DB
```

---

## Setup

### Prerequisites
- Python 3.12
- Gemini API key
- Pinecone API key

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

Edit `.env`:
```
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### Embeddings

The Pinecone index (`ray-peat-corpus`) is pre-populated with 22,457 native 3072-dim vectors. No local embedding setup required to run the app.

For local embedding artifacts, set `HF_DATASET_REPO` in `.env` and run:
```bash
python peatlearn/embedding/hf_download.py
```

---

## Features

- **RAG Chatbot** — Ask questions about Ray Peat's work; retrieves from 22,457 embedded QA pairs (avg score 8.6/10)
- **Adaptive Quizzes** — Gemini-powered quizzes that adapt to your knowledge level
- **Topic Browser** — TF-IDF + KMeans topic model auto-selects 12–36 clusters from the corpus
- **Personalized Recommendations** — Matrix factorization (SGD, 32-dim) + RL content selector
- **User Profiles** — AI-generated learning profiles with knowledge graph

---

## Corpus

| Type | Count |
|------|-------|
| Audio transcripts | 188 |
| Academic papers | 96 |
| Health topics | 98 |
| Newsletters | 59 |
| Other | 111 |
| **Total** | **552 documents** |

Processed into **22,457 QA pairs**, embedded at 3072 dimensions in Pinecone.

---

## Data Pipeline

```
data/raw/  →  preprocessing/cleaning/  →  data/processed/ai_cleaned/
          →  peatlearn/embedding/      →  Pinecone (ray-peat-corpus)
```

- **Tier 1** (~27%): Rules-based cleaning for high-quality documents
- **Tier 2** (~73%): AI-powered cleaning (OCR correction, speaker attribution, segmentation)

---

## API Endpoints

**RAG service (port 8000)**
```
POST /ask          — Ask a question, returns RAG answer
GET  /search       — Semantic search over corpus
GET  /health       — Health check
```

**ML service (port 8001)**
```
POST /quiz/generate        — Generate adaptive quiz
POST /profile/analyze      — Generate user learning profile
GET  /recommendations      — Personalized content recommendations
```

---

## Testing

```bash
pytest tests/              # all tests
pytest tests/unit/         # unit only
pytest tests/integration/  # integration only
```

---

## RAG Quality Benchmark

The RAG chatbot is evaluated against a fixed 30-question benchmark with dual scoring: LLM-as-judge (Gemini 2.5-flash on a 5-dimension rubric) + automated metrics (citations, vocab hit rate, source diversity, topic coverage).

```bash
python scripts/eval_rag_quality.py               # full 30-question run
python scripts/eval_rag_quality.py --subset A,B  # only specific categories
python scripts/eval_rag_quality.py --no-judge    # automated metrics only
```

Question set lives in `data/eval/questions.json`; results are saved to `data/eval/results_<timestamp>.json`. See `data/eval/README.md` for the full rubric and category breakdown.

### Score history

| Date | Score | Notes |
|------|------:|-------|
| commit `ed84cf1` | 8.60 / 10 | Baseline — HyDE + two-pass Pinecone + MMR diversity (ad-hoc score) |
| 2026-04-11 | **8.95 / 10** | **+0.35** — cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) + MMR `float('-inf')` fix |

**Latest run (8.95/10, 29/30 judged):**

| Category | Score | | Rubric dimension | Score |
|----------|------:|---|------------------|------:|
| core_bioenergetics | 9.10 | | accuracy | 9.40 |
| disease_clinical | 9.10 | | grounding | 9.24 |
| cross_concept | 9.03 | | attribution_style | 8.95 |
| hormones_endocrine | 8.94 | | domain_fluency | 8.45 |
| nutrition_foods | 8.90 | | completeness | 8.07 |
| edge_ambiguous | 8.82 | | | |
| edge_nuanced | 8.56 | | | |

Automated metrics: **source diversity 0.92**, expected-topic coverage 0.71, 100% of answers returned ≥ expected sources, avg 5.0 inline citations per answer.

---

## Dataset Hosting

Embeddings are hosted on HuggingFace to keep the repo lightweight. Set `HF_DATASET_REPO` in `.env` to your dataset repo. The Pinecone index is the primary retrieval backend and requires no local downloads to use.

---

## Acknowledgments

- **Dr. Ray Peat** — For his groundbreaking work in bioenergetic medicine
- **Community** — Ray Peat researchers and enthusiasts worldwide

---

*"Energy and structure are interdependent at every level."* — Ray Peat
