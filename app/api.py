#!/usr/bin/env python3
"""
Ray Peat Legacy - Backend API Server (Pinecone-backed)

FastAPI application providing RAG-powered search and question answering using
Pinecone for vector search. The legacy, file-based RAG has been sunset.
"""

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from peatlearn.rag.vector_search import PineconeVectorSearch as RayPeatVectorSearch
from peatlearn.rag.rag_system import PineconeRAG as RayPeatRAG

# Initialize RAG components (Pinecone)
search_engine = RayPeatVectorSearch(index_name="ray-peat-corpus")
rag_system = RayPeatRAG(search_engine, index_name="ray-peat-corpus")

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

@app.get("/api/search", response_model=SearchResponse)
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

@app.get("/api/ask", response_model=QuestionResponse)
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

@app.get("/api/stats", response_model=CorpusStatsResponse)
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

@app.get("/api/related")
async def get_related_topics(
    query: str = Query(..., description="Query to find related topics for"),
    limit: int = Query(8, ge=1, le=20, description="Number of related topics to return")
):
    """Get topics related to the query."""
    try:
        topics = await rag_system.get_related_topics(query, max_topics=limit)
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