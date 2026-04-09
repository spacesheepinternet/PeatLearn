# Data Assets Reference

> Summary: All data paths, corpus breakdown (552 docs), embedding files, model artifacts, and SQLite schema.
> Load this when: working with data files, embeddings, model artifacts, or debugging data pipeline issues.
> Do NOT load for code-only tasks.

---

## Directory Map

```
data/
├── raw/
│   ├── Ray Peat Anthology.xlsx     ← primary corpus (552 docs)
│   └── new_content_2026/           ← new PDFs added 2026
│       └── *.pdf
│
├── processed/
│   └── ai_cleaned/                 ← cleaned + chunked output from preprocessing pipeline
│
├── embeddings/
│   ├── vectors/
│   │   ├── embeddings_20250728_221826.npy   ← latest embedding matrix (552 × 768)
│   │   └── embeddings_20250728_221826.pkl   ← metadata (id, source_file, context, etc.)
│   └── cache/                      ← per-doc cache (saves re-embedding cost)
│
├── models/
│   ├── topics/                     ← CorpusTopicModel artifacts
│   │   ├── vectorizer.pkl
│   │   ├── topic_model.pkl
│   │   └── cluster_labels.npy
│   └── recs/
│       └── mf_model.npz            ← MF recommender (32-dim SGD)
│
├── checkpoints/                    ← preprocessing pipeline resume points
├── artifacts/                      ← JSON quality reports from preprocessing
├── user_interactions/
│   └── interactions.db             ← SQLite: all quiz + interaction logs
└── vectorstore/
    └── chroma/                     ← ChromaDB artifacts (legacy, not in use)
```

---

## Corpus Breakdown (552 total docs)

| Type | Count |
|------|-------|
| Transcripts | 188 |
| Papers | 96 |
| Health topics | 98 |
| Newsletters | 59 |
| Other / misc | 111 |
| **Total** | **552** |

Source: `data/raw/Ray Peat Anthology.xlsx`

---

## Embedding Files

| File | Description |
|------|-------------|
| `embeddings_20250728_221826.npy` | Float32 matrix, shape `(552, 768)` |
| `embeddings_20250728_221826.pkl` | List of dicts: `{id, context, ray_peat_response, source_file, tokens}` |

HuggingFace mirror: set via `HF_DATASET_REPO` env variable

To download:
```bash
python peatlearn/embedding/download_from_hf.py
```

To upload after re-embedding:
```bash
python peatlearn/embedding/upload_to_hf.py
```

---

## Model Artifacts

### Topic Model (`data/models/topics/`)
- `vectorizer.pkl` — sklearn `TfidfVectorizer`
- `topic_model.pkl` — sklearn `KMeans` (k auto-selected from [12, 36] by silhouette score)
- `cluster_labels.npy` — per-doc cluster assignments

Retrain: `python -m peatlearn.adaptive.topic_model` (or via training script)

### MF Recommender (`data/models/recs/mf_model.npz`)
- Fields: `user_factors`, `item_factors`, `user_index`, `item_index`
- 32 latent dimensions, trained with SGD (lr=0.01, decay=0.95/epoch, 15 epochs)

---

## SQLite: interactions.db

Table: `interactions`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | |
| `user_id` | TEXT | |
| `topic` | TEXT | whitelist-enforced topic label |
| `score` | REAL | quiz/interaction score 0–1 |
| `difficulty` | REAL | difficulty level 0–1 |
| `timestamp` | TEXT | ISO 8601 |
| `cosine_sim` | REAL | similarity score from RAG retrieval |
| `fuzzy_jargon_score` | REAL | jargon match score from topic assignment |
| `content_id` | TEXT | |
| `interaction_type` | TEXT | e.g. "quiz", "search", "view" |
| `time_spent` | REAL | seconds |

Access via `peatlearn/personalization/quiz_logger.py → log_quiz_outcome()`.

---

## Checkpoints

`data/checkpoints/` stores JSON files keyed by pipeline run ID.
Each checkpoint records: last processed file index, quality scores so far, errors.
The preprocessing pipeline automatically resumes from the latest checkpoint.

---

## Git-ignored Assets

These are **never committed to git**:
- `data/embeddings/vectors/*.npy` and `*.pkl`
- `data/models/**/*.pkl`, `*.npz`, `*.npy`
- `data/user_interactions/interactions.db`
- `data/checkpoints/`
- `.env`
