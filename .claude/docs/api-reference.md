# API Reference

> Summary: All FastAPI endpoints on port 8000 (RAG) and port 8001 (advanced ML), with request/response models and CORS config.
> Load this when: building clients, writing integration tests, or debugging API issues.
> Do NOT load for backend implementation tasks — use `.claude/rules/rag.md` or `.claude/rules/personalization.md` instead.

---

## CORS Configuration

Both APIs share these CORS settings (from `config/settings.py`):
```python
allow_origins = ["http://localhost:3000", "http://localhost:8000"]
allow_credentials = True
allow_methods = ["*"]
allow_headers = ["*"]
```

---

## Port 8000 — RAG API (`app/api.py`)

Base URL: `http://localhost:8000`

### `GET /`
Health check.

**Response:**
```json
{
  "message": "Welcome to Ray Peat Legacy",
  "status": "healthy",
  "version": "1.0.0",
  "corpus_loaded": true
}
```

---

### `GET /api/search`
Semantic search against the corpus.

**Query params:**
| Param | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `q` | string | required | — | Search query |
| `limit` | int | 10 | 1–50 | Number of results |
| `min_similarity` | float | 0.1 | 0.0–1.0 | Minimum similarity threshold |

**Response: `SearchResponse`**
```json
{
  "query": "thyroid and metabolism",
  "results": [
    {
      "id": "doc_123",
      "context": "...",
      "ray_peat_response": "...",
      "source_file": "transcripts/interview_2019.txt",
      "similarity_score": 0.87,
      "tokens": 245
    }
  ],
  "total_results": 5
}
```

---

### `GET /api/ask`
RAG question answering — retrieves context then generates an answer with Gemini.

**Query params:**
| Param | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `question` | string | required | — | Question to answer |
| `max_sources` | int | 5 | 1–10 | Max source passages |
| `min_similarity` | float | 0.3 | 0.0–1.0 | Min source similarity |

**Response: `QuestionResponse`**
```json
{
  "question": "What does Ray Peat say about thyroid?",
  "answer": "According to Ray Peat...",
  "confidence": 0.82,
  "sources": [ /* same shape as search results */ ]
}
```

---

### `GET /api/stats`
Corpus statistics (Pinecone-adapted).

**Response: `CorpusStatsResponse`**
```json
{
  "total_embeddings": 552,
  "total_tokens": 0,
  "embedding_dimensions": 768,
  "source_files": 552,
  "files_breakdown": {}
}
```

---

## Port 8001 — Advanced ML API (`app/advanced_api.py`)

Base URL: `http://localhost:8001`

Note: Advanced ML endpoints degrade gracefully when `torch-geometric` is unavailable (`ADVANCED_ML_AVAILABLE = False`).

### `GET /`
Health check + feature availability flags.

---

### `POST /api/users`
Create or update a user profile.

**Body: `UserProfile`**
```json
{
  "user_id": "user_abc",
  "name": "Alice",
  "email": "alice@example.com",
  "learning_style": "visual",
  "preferences": {}
}
```

---

### `POST /api/interactions`
Log a user interaction.

**Body: `InteractionData`**
```json
{
  "user_id": "user_abc",
  "content_id": "doc_123",
  "interaction_type": "quiz",
  "performance_score": 0.8,
  "time_spent": 45.0,
  "difficulty_level": 0.5,
  "topic_tags": ["thyroid", "metabolism"],
  "context": {}
}
```

---

### `POST /api/quiz`
Generate an adaptive quiz for a user.

**Body: `QuizRequest`**
```json
{
  "user_id": "user_abc",
  "topic": "thyroid"
}
```

**Response:** quiz questions array with MCQ format.

---

### `GET /api/recommendations/{user_id}`
Get personalized content recommendations.

**Response:** array of recommended content items with scores.

---

### `GET /api/learning-path/{user_id}`
Get a sequenced learning path based on RL agent + knowledge graph.

---

### `GET /api/knowledge-graph`
Return the knowledge graph structure (nodes + edges for concept map).

---

## Pydantic Models Summary

### Port 8000
| Model | Fields |
|-------|--------|
| `SearchResponse` | query, results[], total_results |
| `QuestionResponse` | question, answer, confidence, sources[] |
| `CorpusStatsResponse` | total_embeddings, total_tokens, embedding_dimensions, source_files, files_breakdown |

### Port 8001
| Model | Fields |
|-------|--------|
| `UserProfile` | user_id, name, email?, learning_style?, preferences{} |
| `InteractionData` | user_id, content_id, interaction_type, performance_score, time_spent, difficulty_level, topic_tags[], context{} |
| `QuizRequest` | user_id, topic? |

---

## Example: Python client

```python
import httpx

async def ask_peat(question: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "http://localhost:8000/api/ask",
            params={"question": question, "max_sources": 5}
        )
        r.raise_for_status()
        return r.json()
```
