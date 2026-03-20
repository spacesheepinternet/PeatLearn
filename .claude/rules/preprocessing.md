---
paths:
  - preprocessing/**
---

# Preprocessing Subsystem Rules

## Key Files

| File | Purpose |
|------|---------|
| `preprocessing/optimized_pipeline.py` | Main pipeline orchestrator — reads raw, cleans, chunks, outputs |
| `preprocessing/parallel_processor.py` | Multi-process wrapper for the pipeline |
| `preprocessing/checkpoint_system.py` | Saves/loads progress checkpoints to `data/checkpoints/` |
| `preprocessing/cleaning/` | Rule-based + AI-powered cleaning modules |

## Pipeline Flow

```
data/raw/  →  cleaning/  →  chunking  →  data/processed/ai_cleaned/
```

- Raw sources: `data/raw/Ray Peat Anthology.xlsx`, `data/raw/new_content_2026/*.pdf`
- Output: cleaned, chunked text files in `data/processed/ai_cleaned/`
- Checkpoints: `data/checkpoints/` — resume a failed run without reprocessing

## Quality Thresholds (from `config/settings.py`)

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `TIER1_FIDELITY_THRESHOLD` | 4.0 | Min fidelity score to pass |
| `TIER1_ATOMICITY_THRESHOLD` | 5.0 | Min atomicity score |
| `TIER1_NOISE_THRESHOLD` | 5.0 | Max noise score to pass |

Chunks below threshold are filtered out before writing to `data/processed/`.

## Chunking Settings (from `config/settings.py`)

- `MAX_TOKENS_PER_CHUNK`: 1000
- `CHUNK_OVERLAP`: 200
- `BATCH_SIZE`: 10

## Do Not
- Do not mutate `data/raw/` — it is the source of truth; preprocessing only reads it.
- Do not skip the checkpoint system for long runs — re-running from scratch is expensive.
- Do not use the parallel processor for small batches (< 50 docs) — overhead not worth it.
- AI-powered cleaning uses Gemini API calls; always check rate limits before large runs.
