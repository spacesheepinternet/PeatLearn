#!/usr/bin/env python3
"""
Ray Peat Legacy - Backend API Server (Pinecone-backed)

FastAPI application providing RAG-powered search and question answering using
Pinecone for vector search. The legacy, file-based RAG has been sunset.
"""

import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from peatlearn.rag.vector_search import PineconeVectorSearch as RayPeatVectorSearch
from peatlearn.rag.rag_system import PineconeRAG as RayPeatRAG

# --- Auth ---------------------------------------------------------------------
# If settings.API_BEARER_TOKEN is unset (None/empty), auth is bypassed — handy
# for local dev. In production, set API_BEARER_TOKEN in the environment.
def verify_bearer_token(authorization: Optional[str] = Header(default=None)) -> None:
    expected = settings.API_BEARER_TOKEN
    if not expected:
        return  # auth disabled
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

# --- Rate limiting ------------------------------------------------------------
# In-process token bucket per client IP. Fine for single-worker private beta;
# upgrade to Redis-backed slowapi or an external gateway for multi-worker prod.
_RATE_WINDOW_SECONDS = 60
_rate_lock = Lock()
_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)

def rate_limit(request: Request) -> None:
    limit = settings.API_RATE_LIMIT
    if limit <= 0:
        return
    client_ip = (request.client.host if request.client else "anon") or "anon"
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_SECONDS
    with _rate_lock:
        bucket = _rate_buckets[client_ip]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(_RATE_WINDOW_SECONDS - (now - bucket[0])))
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limit}/min). Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)

# Initialize RAG components (Pinecone) — use the active index from settings,
# not the legacy "ray-peat-corpus" rollback index.
search_engine = RayPeatVectorSearch(index_name=settings.PINECONE_INDEX_NAME)
rag_system = RayPeatRAG(search_engine, index_name=settings.PINECONE_INDEX_NAME)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    results: List[dict]
    total_results: int

class QuestionResponse(BaseModel):
    """Response model for question answering endpoint."""
    question: str
    answer: str
    confidence: float
    sources: List[dict]
    confidence_tier: Optional[str] = None
    confidence_reasons: Optional[List[str]] = None

class CorpusStatsResponse(BaseModel):
    """Response model for corpus statistics."""
    total_embeddings: int
    total_tokens: int
    embedding_dimensions: int
    source_files: int
    files_breakdown: dict

@app.get("/")
async def root():
    """Health check endpoint."""
    try:
        stats = search_engine.get_corpus_stats()
        # Pinecone returns total_vectors; treat >0 as loaded
        vector_count = stats.get("total_vectors") or stats.get("total_embeddings") or 0
    except Exception:
        vector_count = 0
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "status": "healthy",
        "version": settings.VERSION,
        "corpus_loaded": vector_count > 0
    }

@app.get(
    "/api/search",
    response_model=SearchResponse,
    dependencies=[Depends(verify_bearer_token), Depends(rate_limit)],
)
async def search_corpus(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of results to return"),
    min_similarity: float = Query(0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """Search the Ray Peat corpus using semantic search."""
    try:
        results = await search_engine.search(
            query=q,
            top_k=limit,
            min_similarity=min_similarity
        )
        
        # Convert results to dict format
        results_dict = [
            {
                "id": result.id,
                "context": result.context,
                "ray_peat_response": result.ray_peat_response,
                "source_file": result.source_file,
                "similarity_score": result.similarity_score,
                "tokens": result.tokens
            }
            for result in results
        ]
        
        return SearchResponse(
            query=q,
            results=results_dict,
            total_results=len(results_dict)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get(
    "/api/ask",
    response_model=QuestionResponse,
    dependencies=[Depends(verify_bearer_token), Depends(rate_limit)],
)
async def ask_question(
    question: str = Query(..., description="Question to ask Ray Peat's knowledge base"),
    max_sources: int = Query(5, ge=1, le=10, description="Maximum number of sources to use"),
    min_similarity: float = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity for sources")
):
    """Ask a question and get an AI-generated answer based on Ray Peat's work."""
    try:
        rag_response = await rag_system.answer_question(
            question=question,
            max_sources=max_sources,
            min_similarity=min_similarity
        )
        
        # Convert sources to dict format
        sources_dict = [
            {
                "id": source.id,
                "context": source.context,
                "ray_peat_response": source.ray_peat_response,
                "source_file": source.source_file,
                "similarity_score": source.similarity_score,
                "tokens": source.tokens
            }
            for source in rag_response.sources
        ]
        
        return QuestionResponse(
            question=question,
            answer=rag_response.answer,
            confidence=rag_response.confidence,
            sources=sources_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering error: {str(e)}")

@app.get(
    "/api/stats",
    response_model=CorpusStatsResponse,
    dependencies=[Depends(verify_bearer_token), Depends(rate_limit)],
)
async def get_corpus_stats():
    """Get statistics about the loaded Ray Peat corpus (adapted for Pinecone)."""
    try:
        stats = search_engine.get_corpus_stats()
        if "error" in stats:
            raise HTTPException(status_code=503, detail=stats["error"])

        # Adapt Pinecone stats to legacy response schema expected by UI
        total_embeddings = stats.get("total_vectors", stats.get("total_embeddings", 0))
        embedding_dimensions = stats.get("embedding_dimensions", 0)

        # Tokens/source breakdown not tracked in Pinecone stats endpoint
        total_tokens = 0
        source_files = 0
        files_breakdown = {}

        return CorpusStatsResponse(
            total_embeddings=total_embeddings,
            total_tokens=total_tokens,
            embedding_dimensions=embedding_dimensions,
            source_files=source_files,
            files_breakdown=files_breakdown,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.get(
    "/api/related",
    dependencies=[Depends(verify_bearer_token), Depends(rate_limit)],
)
async def get_related_topics(
    query: str = Query(..., description="Query to find related topics for"),
    limit: int = Query(8, ge=1, le=20, description="Number of related topics to return")
):
    """Get topics related to the query."""
    try:
        topics = await rag_system.get_related_questions(query, max_questions=limit)
        return {
            "query": query,
            "related_topics": topics,
            "total_topics": len(topics)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Related topics error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Health check (for orchestrator)
@app.get("/api/health")
async def health_check():
    try:
        stats = search_engine.get_corpus_stats()
        vector_count = stats.get("total_vectors") or stats.get("total_embeddings") or 0
        return {
            "status": "healthy",
            "pinecone": True,
            "vectors": vector_count,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }

@app.get("/api/topics")
async def get_topics():
    """Get available topics and categories."""
    return {
        "topics": [
            "Thyroid Function",
            "Nutrition",
            "Hormones",
            "Metabolism",
            "Stress",
            "Supplements"
        ],
        "message": "Topics will be dynamically generated from corpus analysis"
    }