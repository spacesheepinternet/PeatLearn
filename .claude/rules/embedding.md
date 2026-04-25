---
paths:
  - peatlearn/embedding/**
  - peatlearn/rag/embedder.py
  - scripts/embedding/**
  - scripts/finetune_embeddings.py
  - scripts/reembed_corpus.py
  - scripts/generate_training_pairs.py
---

# Embedding Subsystem Rules

## Key Classes / Scripts

| File | Purpose |
|------|---------|
| `peatlearn/rag/embedder.py` | **Active** — singleton loader for fine-tuned EmbeddingGemma; `get_embedding()` |
| `peatlearn/embedding/embed_corpus.py` | Legacy `CorpusEmbedder` — batches docs via Gemini API (no longer primary) |
| `scripts/generate_training_pairs.py` | Generates (query, positive, negative) triples for contrastive fine-tuning |
| `scripts/finetune_embeddings.py` | Fine-tunes EmbeddingGemma with MultipleNegativesRankingLoss |
| `scripts/reembed_corpus.py` | Re-embeds full corpus with fine-tuned model |
| `scripts/upload_gemma_pinecone.py` | Uploads 768-dim vectors to `ray-peat-corpus-v2` |
| `peatlearn/embedding/download_from_hf.py` | Downloads embeddings from HuggingFace |
| `peatlearn/embedding/upload_to_hf.py` | Uploads embeddings to HuggingFace |

## Embedding Spec

- Model: Fine-tuned `google/embeddinggemma-300m` → `data/models/embeddings/peat-embeddinggemma-ft/`
- Dimensions: **768** — configured in `config/settings.py` and Pinecone index `ray-peat-corpus-v2`
- Runs locally on GPU (RTX 4070, ~1400 texts/sec) or CPU
- Training data: 256 (query, positive, negative) triples in `data/training/embedding_pairs.jsonl`
- Output format: `data/embeddings/vectors/embeddings_gemma_ft_<timestamp>.npy` + `.json` metadata

## HuggingFace Sync

- Repository: set via `HF_DATASET_REPO` env variable
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

## Rollback
- Old index `ray-peat-corpus` (3072-dim, Gemini) is still alive for rollback.
- Set `PINECONE_INDEX_NAME=ray-peat-corpus` + `EMBEDDING_MODEL=gemini-embedding-001` + `EMBEDDING_DIMENSIONS=3072` to revert.

## Do Not
- Do not change embedding dimensions from 768 without re-embedding the corpus and creating a new Pinecone index.
- Do not delete `data/models/embeddings/peat-embeddinggemma-ft/` — it is the active model.
- Do not embed in the main thread during a server request — it blocks.
- Do not delete `data/embeddings/cache/` — it saves API cost on legacy re-runs.
- Do not commit `.npy`/`.pkl` embedding files to git — they are large binary assets.
