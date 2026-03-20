---
paths:
  - tests/**
---

# Testing Rules

## Test Layout

```
tests/
  __init__.py
  unit/
    test_pipeline.py
    test_mf_trainer.py
    test_personalization_engine.py
    test_adaptive_system.py
    test_ai_system.py
  integration/
    test_api.py
    test_basic.py
    test_vector_search.py
    test_advanced_ml.py
    test_advanced_ml_full.py
    verify_system.py
```

## Hard Rules

1. **No `sys.path` hacks** — the `peatlearn` package is importable from the project root. Never add `sys.path.insert(0, ...)` in test files.
2. **Unit tests must not require API keys** — mock or stub external services (Gemini, Pinecone).
3. **Integration tests may require live services** — mark them with `@pytest.mark.integration` and guard with `pytest.importorskip` or `skipIf` when keys are absent.
4. **No database mutations in unit tests** — use temp dirs or in-memory SQLite.

## Running Tests

```bash
# All tests
pytest tests/

# Unit only (no API keys needed)
pytest tests/unit/

# Integration (needs .env)
pytest tests/integration/

# Single file
pytest tests/unit/test_mf_trainer.py -v
```

## Pytest Config

- Config lives in `pyproject.toml` or `pytest.ini` at project root (check which exists).
- Fixtures for common objects (settings, temp db) should live in `tests/conftest.py`.

## What to Test

| Component | Test type | Key assertions |
|-----------|-----------|---------------|
| MF trainer | Unit | Convergence, correct output shape, loss decreasing |
| Personalization engine | Unit | Returns recommendations without crashing |
| Pipeline | Unit | Chunks are within token limits, quality filters applied |
| API endpoints | Integration | Status 200, correct response schema |
| Vector search | Integration | Returns results with similarity scores |

## Do Not
- Do not write tests that embed real documents — too slow and costly.
- Do not write tests that call `PineconeVectorSearch.search()` without mocking — requires live Pinecone.
- Do not use `assert response == exact_string` for LLM outputs — they are non-deterministic.
