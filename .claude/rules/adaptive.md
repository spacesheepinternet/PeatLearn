---
paths:
  - peatlearn/adaptive/**
---

# Adaptive Learning Subsystem Rules

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `QuizGenerator` | `peatlearn/adaptive/quiz_generator.py` | Gemini-powered adaptive quiz generation |
| `CorpusTopicModel` | `peatlearn/adaptive/topic_model.py` | TF-IDF + KMeans + SVD topic clustering |
| `DataLogger` | `peatlearn/adaptive/data_logger.py` | Logs interactions to SQLite |
| `ContentSelector` | `peatlearn/adaptive/content_selector.py` | Selects next content based on mastery |
| `LearnerProfiler` / `AIProfileAnalyzer` | `peatlearn/adaptive/profile_analyzer.py`, `ai_profile_analyzer.py` | Builds learner profiles |
| `AdaptiveRAGSystem` | `peatlearn/adaptive/rag_system.py` | Adaptive wrapper around PineconeRAG |

## Topic Model

- Uses **silhouette score** to auto-select k in range `[12, 36]`.
- Artifacts (vectorizer, model, cluster labels) saved to `data/models/topics/`.
- Topic whitelist is enforced — topics outside the whitelist are rejected.
- Corpus-driven: trained on `data/processed/ai_cleaned/`.

## Quiz Generator

- Calls `gemini-2.5-flash` via `call_llm_api` helper.
- Weakest-topic targeting: selects topics where mastery score is lowest.
- MCQ format: 4 options, 1 correct, difficulty adapts to learner history.
- Quiz outcomes logged via `peatlearn/personalization/quiz_logger.py`.

## Data Logger / SQLite Schema

Interactions are stored in `data/user_interactions/interactions.db`.
Key columns: `user_id`, `topic`, `score`, `difficulty`, `timestamp`, `cosine_sim`, `fuzzy_jargon_score`.

## Import Pattern
```python
from peatlearn.adaptive.quiz_generator import QuizGenerator, call_llm_api
from peatlearn.adaptive.topic_model import CorpusTopicModel
from peatlearn.adaptive.data_logger import DataLogger
```

## Do Not
- Do not generate quizzes synchronously inside request handlers — wrap in `asyncio.to_thread` if needed.
- Do not skip the topic whitelist filter when assigning topics.
- Do not retrain the topic model on every server start — load from `data/models/topics/` if artifacts exist.
