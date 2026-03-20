---
paths:
  - peatlearn/rag/**
  - app/api.py
---

# RAG Subsystem Rules

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `PineconeVectorSearch` | `peatlearn/rag/vector_search.py` | Async semantic search against Pinecone |
| `PineconeRAG` | `peatlearn/rag/rag_system.py` | Full RAG: retrieve → rerank → generate |
| `SearchResult` | `peatlearn/rag/vector_search.py` | Dataclass: id, context, ray_peat_response, source_file, similarity_score, tokens |
| `RAGResponse` | `peatlearn/rag/rag_system.py` | Dataclass: answer, sources, confidence, query, search_stats |

## Critical Behaviours

### Offline Fallback
`PineconeVectorSearch` falls back to SHA-256 hash-based pseudo-embeddings when the Gemini API is unavailable. Never remove this fallback — it keeps tests runnable without API keys.

### Rerank & Dedupe
`PineconeRAG._rerank_and_dedupe` scores candidates as:
```
score = 0.7 × vector_similarity + 0.3 × keyword_overlap
```
Dedupe is exact-match on `source_file + context[:100]`. Do not change weights without benchmarking.

### LLM Selection
- `PineconeRAG` uses `gemini-2.5-flash` (not lite) — quality matters for answers.
- Has both Gemini SDK path and aiohttp HTTP fallback. Keep both.

## API Endpoints (port 8000)

| Method | Path | Params |
|--------|------|--------|
| GET | `/` | — |
| GET | `/api/search` | `q`, `limit` (1-50), `min_similarity` (0-1) |
| GET | `/api/ask` | `question`, `max_sources` (1-10), `min_similarity` |
| GET | `/api/stats` | — |

## Import Pattern
```python
from peatlearn.rag.vector_search import PineconeVectorSearch
from peatlearn.rag.rag_system import PineconeRAG, RAGResponse

# In app/api.py these are aliased for backward compat:
from peatlearn.rag.vector_search import PineconeVectorSearch as RayPeatVectorSearch
from peatlearn.rag.rag_system import PineconeRAG as RayPeatRAG
```

## Do Not
- Do not reintroduce file-based / ChromaDB RAG — it is deprecated.
- Do not change the Pinecone index name `ray-peat-corpus` without re-uploading embeddings.
- Do not make `answer_question` synchronous — it is `async` throughout.
