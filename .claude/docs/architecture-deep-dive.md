# Architecture Deep Dive

> Summary: Full data flow, package dependency map, port map, and all class names.
> Load this when: debugging cross-subsystem issues, planning major refactors, or onboarding to the full system.
> Do NOT load for single-subsystem tasks — use `.claude/rules/<subsystem>.md` instead.

---

## Port Map

| Service | Port | File | Command |
|---------|------|------|---------|
| RAG backend | 8000 | `app/api.py` | `uvicorn app.api:app --port 8000` |
| Advanced ML backend | 8001 | `app/advanced_api.py` | `uvicorn app.advanced_api:app --port 8001` |
| Streamlit dashboard | 8501 (default) | `app/dashboard.py` | `streamlit run app/dashboard.py` |

---

## Full Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  RAW DATA                                                       │
│  data/raw/Ray Peat Anthology.xlsx                               │
│  data/raw/new_content_2026/*.pdf                                │
└────────────────────────────┬────────────────────────────────────┘
                             │ preprocessing/optimized_pipeline.py
                             │ preprocessing/parallel_processor.py
                             │ preprocessing/checkpoint_system.py
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROCESSED                                                      │
│  data/processed/ai_cleaned/  (cleaned, chunked text)           │
└────────────────────────────┬────────────────────────────────────┘
                             │ peatlearn/embedding/embed_corpus.py
                             │ CorpusEmbedder → gemini-embedding-001
                             ▼
┌───────────────────────────────┐    ┌──────────────────────────────┐
│  LOCAL VECTORS                │    │  PINECONE                    │
│  data/embeddings/vectors/     │    │  index: ray-peat-corpus      │
│  embeddings_*.npy + .pkl      │    │  768-dim, ~552 docs          │
└───────────────────────────────┘    └──────────────┬───────────────┘
                                                    │ PineconeVectorSearch
                                                    │ PineconeRAG
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  RAG BACKEND  (port 8000)                                       │
│  app/api.py                                                     │
│  GET /api/search  GET /api/ask  GET /api/stats                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STREAMLIT DASHBOARD  (port 8501)                               │
│  app/dashboard.py                                               │
│  Also calls port 8001 for ML features                           │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │ HTTP
┌────────────────────────────┴────────────────────────────────────┐
│  ADVANCED ML BACKEND  (port 8001)                               │
│  app/advanced_api.py                                            │
│  POST /api/users  POST /api/interactions  POST /api/quiz        │
│  GET /api/recommendations/{user_id}                             │
│  GET /api/learning-path/{user_id}                               │
│  GET /api/knowledge-graph                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Package Dependency Map

```
peatlearn/
├── rag/
│   ├── vector_search.py   ← PineconeVectorSearch (depends: pinecone, google-generativeai)
│   └── rag_system.py      ← PineconeRAG (depends: vector_search, google-generativeai, aiohttp)
│
├── adaptive/
│   ├── quiz_generator.py  ← QuizGenerator (depends: rag, google-generativeai)
│   ├── topic_model.py     ← CorpusTopicModel (depends: sklearn, numpy)
│   ├── data_logger.py     ← DataLogger (depends: sqlite3)
│   ├── content_selector.py← ContentSelector (depends: topic_model, data_logger)
│   ├── profile_analyzer.py← LearnerProfiler (depends: data_logger)
│   ├── ai_profile_analyzer.py ← AIProfileAnalyzer (depends: google-generativeai)
│   └── rag_system.py      ← AdaptiveRAGSystem (depends: rag)
│
├── personalization/
│   ├── engine.py          ← PersonalizationEngine, QuizRecommendationSystem (depends: recommendation)
│   ├── neural.py          ← AdvancedPersonalizationEngine, UserInteraction, LearningState (depends: torch)
│   ├── rl_agent.py        ← AdaptiveLearningAgent, LearningEnvironmentState, adaptive_agent (depends: torch)
│   ├── knowledge_graph.py ← AdvancedKnowledgeGraph, ray_peat_knowledge_graph (depends: torch-geometric)
│   ├── quiz_logger.py     ← log_quiz_outcome (depends: sqlite3)
│   └── utils.py           ← generate_mcq_from_passage (depends: google-generativeai)
│
├── embedding/
│   ├── embed_corpus.py    ← CorpusEmbedder (depends: google-generativeai, numpy, pickle)
│   ├── download_from_hf.py← (depends: huggingface_hub)
│   ├── upload_to_hf.py    ← (depends: huggingface_hub)
│   ├── check_vectors.py   ← validation utility
│   ├── setup_env.py       ← dependency checker
│   ├── monitor_progress.py← progress monitor
│   └── pinecone/          ← legacy upload scripts (use scripts/embedding/ for new work)
│
└── recommendation/
    └── mf_trainer.py      ← MFTrainer (depends: numpy, scipy)
```

---

## All Class Names (quick reference)

### peatlearn/rag/
- `PineconeVectorSearch` — async search, SHA-256 offline fallback
- `SearchResult` — dataclass: id, context, ray_peat_response, source_file, similarity_score, tokens
- `PineconeRAG` — retrieve → rerank → generate
- `RAGResponse` — dataclass: answer, sources, confidence, query, search_stats

### peatlearn/adaptive/
- `QuizGenerator`
- `CorpusTopicModel`
- `DataLogger`
- `ContentSelector`
- `LearnerProfiler`
- `AIProfileAnalyzer`
- `AdaptiveRAGSystem`

### peatlearn/personalization/
- `PersonalizationEngine`
- `QuizRecommendationSystem`
- `AdvancedPersonalizationEngine`
- `UserInteraction`
- `LearningState`
- `personalization_engine` (module-level singleton)
- `AdaptiveLearningAgent`
- `LearningEnvironmentState`
- `adaptive_agent` (module-level singleton)
- `AdvancedKnowledgeGraph`
- `ray_peat_knowledge_graph` (module-level singleton)

### peatlearn/recommendation/
- `MFTrainer`

---

## Config: `config/settings.py`

`Settings` (pydantic-settings) reads from `.env`. Key fields:

```python
GEMINI_API_KEY          # required for embeddings + LLM
PINECONE_API_KEY        # required for vector search
OPENAI_API_KEY          # optional
EMBEDDING_MODEL         # gemini-embedding-001
EMBEDDING_DIMENSIONS    # 768
DEFAULT_LLM_MODEL       # gemini-2.5-flash-lite
VECTOR_DB_TYPE          # pinecone
CORS_ORIGINS            # [localhost:3000, localhost:8000]
```

---

## Deprecated / Legacy (do not use)

- `embedding/` (root-level) → moved to `peatlearn/embedding/`
- `src/adaptive_learning/` → moved to `peatlearn/adaptive/`
- `inference/backend/personalization/` → moved to `peatlearn/personalization/`
- File-based / ChromaDB RAG → replaced by Pinecone
- `inference/backend/rag/` → replaced by `peatlearn/rag/`
