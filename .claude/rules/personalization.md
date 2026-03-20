---
paths:
  - peatlearn/personalization/**
  - app/advanced_api.py
---

# Personalization Subsystem Rules

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `PersonalizationEngine` / `QuizRecommendationSystem` | `peatlearn/personalization/engine.py` | Stub engine; blends MF recs with adaptive signals |
| `AdvancedPersonalizationEngine` | `peatlearn/personalization/neural.py` | Neural collaborative filtering layer |
| `AdaptiveLearningAgent` | `peatlearn/personalization/rl_agent.py` | Deep RL agent for content sequencing |
| `AdvancedKnowledgeGraph` | `peatlearn/personalization/knowledge_graph.py` | GNN-backed concept graph |
| `log_quiz_outcome` | `peatlearn/personalization/quiz_logger.py` | Writes quiz results to SQLite |
| `generate_mcq_from_passage` | `peatlearn/personalization/utils.py` | Utility: extract MCQ from passage text |

## Advanced ML Availability

`app/advanced_api.py` wraps advanced components in a `try/except ImportError` block controlled by `ADVANCED_ML_AVAILABLE`. This is intentional — `torch-geometric` is an optional heavy dependency.

```python
# Pattern used everywhere:
if ADVANCED_ML_AVAILABLE:
    # use neural / RL / KG components
else:
    # fallback to simpler heuristics
```

Do not remove this guard.

## MF Recommender

- Matrix factorization model at `data/models/recs/mf_model.npz`.
- SGD trainer, 32 latent dimensions, 15 epochs, lr decay 0.95/epoch, weight decay.
- Loaded at startup; blended with adaptive signals for final recommendations.
- Source: `peatlearn/recommendation/mf_trainer.py`.

## API Endpoints (port 8001)

| Method | Path | Body / Params |
|--------|------|---------------|
| GET | `/` | — |
| POST | `/api/users` | `UserProfile` |
| POST | `/api/interactions` | `InteractionData` |
| POST | `/api/quiz` | `QuizRequest` |
| GET | `/api/recommendations/{user_id}` | — |
| GET | `/api/learning-path/{user_id}` | — |
| GET | `/api/knowledge-graph` | — |

## Import Pattern
```python
from peatlearn.personalization.engine import PersonalizationEngine
from peatlearn.personalization.quiz_logger import log_quiz_outcome
from peatlearn.personalization.utils import generate_mcq_from_passage
```

## Do Not
- Do not hard-require `torch-geometric` — always guard with `ADVANCED_ML_AVAILABLE`.
- Do not bypass `log_quiz_outcome` — all quiz outcomes must be persisted.
- Do not change MF latent dimensions without retraining and saving a new `mf_model.npz`.
