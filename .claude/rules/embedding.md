---
paths:
  - peatlearn/embedding/**
  - scripts/embedding/**
---

# Embedding Subsystem Rules

## Key Classes / Scripts

| File | Purpose |
|------|---------|
| `peatlearn/embedding/embed_corpus.py` | `CorpusEmbedder` — batches docs, calls Gemini, saves `.npy`+`.pkl` |
| `peatlearn/embedding/download_from_hf.py` | Downloads embeddings from HuggingFace |
| `peatlearn/embedding/upload_to_hf.py` | Uploads embeddings to HuggingFace |
| `peatlearn/embedding/check_vectors.py` | Validates local embedding files |
| `peatlearn/embedding/setup_env.py` | Environment/dependency checker |
| `peatlearn/embedding/monitor_progress.py` | Live progress monitor for long embed runs |

## Embedding Spec

- Model: `gemini-embedding-001`
- Dimensions: **768** — hardcoded in Pinecone index and `config/settings.py`
- Batch size: 10 (Gemini rate limit safe)
- Output format: `data/embeddings/vectors/embeddings_<timestamp>.npy` + matching `.pkl` metadata

## HuggingFace Sync

- Repository: `abanwild/peatlearn-embeddings`
- Download before a fresh embed run to avoid re-embedding already-done docs.
- Upload after a full embed run completes.

## Pinecone Upload

- Pinecone upload scripts are in `peatlearn/embedding/pinecone/` (legacy) and `scripts/embedding/`.
- Use upsert, not insert — duplicate IDs are safe to re-upload.
- Max batch size for Pinecone upsert: 100 vectors.

## Corpus Stats

- 552 total docs: 188 transcripts, 96 papers, 98 health topics, 59 newsletters, + misc.
- Source: `data/raw/Ray Peat Anthology.xlsx` + `data/raw/new_content_2026/*.pdf`
- Latest embedding file: `data/embeddings/vectors/embeddings_20250728_221826.npy`

## Do Not
- Do not change embedding dimensions from 768 — would require re-creating the Pinecone index and re-embedding all 552 docs.
- Do not embed in the main thread during a server request — it blocks.
- Do not delete `data/embeddings/cache/` — it saves API cost on re-runs.
- Do not commit `.npy`/`.pkl` embedding files to git — they are large binary assets.
