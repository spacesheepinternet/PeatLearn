#!/usr/bin/env python3
"""
Pinecone-based RAG (Retrieval-Augmented Generation) System for Ray Peat Knowledge

Combines Pinecone vector search with LLM generation for accurate Q&A.
"""

import sys
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging


from .vector_search import PineconeVectorSearch, SearchResult
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from the Pinecone-based RAG system."""
    answer: str
    sources: List[SearchResult]
    confidence: float
    query: str
    search_stats: Dict[str, Any]

class PineconeRAG:
    """RAG system for answering questions about Ray Peat's work using Pinecone."""
    
    def __init__(self, search_engine: PineconeVectorSearch = None, index_name: str = "ray-peat-corpus"):
        """Initialize the Pinecone-based RAG system."""
        self.search_engine = search_engine or PineconeVectorSearch(index_name=index_name)
        self.llm_model = "gemini-2.5-flash"  # Fast and cost-effective for RAG
        
    async def answer_question(
        self, 
        question: str, 
        max_sources: int = 5,
        min_similarity: float = 0.3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Answer a question using Pinecone-based RAG approach.
        
        Args:
            question: The user's question
            max_sources: Maximum number of source passages to use
            min_similarity: Minimum similarity threshold for sources
            metadata_filter: Optional metadata filters for Pinecone search
            
        Returns:
            RAGResponse with answer and sources
        """
        
        search_stats = {"method": "pinecone_vector_search"}
        
        try:
            # Step 1: Retrieve relevant passages from Pinecone
            raw_results = await self.search_engine.search(
                query=question,
                top_k=max_sources,
                min_similarity=min_similarity,
                filter_dict=metadata_filter
            )
            # Step 1.1: Rerank and deduplicate for quality and diversity
            search_results = self._rerank_and_dedupe(question, raw_results, max_sources)

            search_stats.update({
                "results_found": len(search_results),
                "min_similarity": min_similarity,
                "metadata_filter": metadata_filter
            })
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information about that topic in Ray Peat's work.",
                    sources=[],
                    confidence=0.0,
                    query=question,
                    search_stats=search_stats
                )
            
            # Step 2: Generate answer using retrieved context
            answer = await self._generate_answer(question, search_results)
            
            # Step 3: Calculate confidence based on similarity scores
            avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
            confidence = min(avg_similarity * 1.2, 1.0)  # Scale similarity to confidence
            
            search_stats.update({
                "avg_similarity": avg_similarity,
                "confidence": confidence,
                "sources_used": len(search_results)
            })
            
            return RAGResponse(
                answer=answer,
                sources=search_results,
                confidence=confidence,
                query=question,
                search_stats=search_stats
            )
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                query=question,
                search_stats={"error": str(e)}
            )

    def _rerank_and_dedupe(self, query: str, results: List[SearchResult], max_sources: int) -> List[SearchResult]:
        """Rerank by combining vector similarity with simple keyword overlap, and deduplicate by source.

        - Encourages diversity across `source_file`
        - Prefers passages with higher query term overlap
        """
        if not results:
            return []

        import re
        from collections import defaultdict

        def tokenize(text: str) -> List[str]:
            return re.findall(r"[a-zA-Z][a-zA-Z\-']+", (text or "").lower())

        stop = {
            "the","a","an","and","or","of","to","in","is","it","on","for","with","as","by","that","this","are","be","at","from","about","into","over","under","than","then","but","if","so","not"
        }
        query_tokens = [t for t in tokenize(query) if t not in stop]
        query_vocab = set(query_tokens)
        if not query_vocab:
            query_vocab = set(tokenize(query))

        scored: List[tuple[float, SearchResult]] = []
        for r in results:
            text = f"{r.context} {r.ray_peat_response}"
            toks = [t for t in tokenize(text) if t not in stop]
            if toks:
                overlap = len(query_vocab.intersection(toks)) / max(1, len(query_vocab))
            else:
                overlap = 0.0
            score = 0.7 * float(r.similarity_score) + 0.3 * float(overlap)
            scored.append((score, r))

        # Sort by score desc
        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by (source_file, normalized excerpt) for diversity
        seen_sources: set[str] = set()
        selected: List[SearchResult] = []
        for _score, r in scored:
            key = (r.source_file or "").strip()
            if key in seen_sources:
                continue
            selected.append(r)
            seen_sources.add(key)
            if len(selected) >= max_sources:
                break

        # If we still have room (few unique sources), fill remaining ignoring source dedupe
        if len(selected) < max_sources:
            for _score, r in scored:
                if r not in selected:
                    selected.append(r)
                    if len(selected) >= max_sources:
                        break

        return selected
    
    async def answer_with_source_filter(
        self,
        question: str,
        source_files: List[str],
        max_sources: int = 5,
        min_similarity: float = 0.3
    ) -> RAGResponse:
        """
        Answer a question using only specific source files.
        
        Args:
            question: The user's question
            source_files: List of source file names to search within
            max_sources: Maximum number of source passages to use
            min_similarity: Minimum similarity threshold
            
        Returns:
            RAGResponse with answer and sources
        """
        # Create metadata filter for specific source files
        metadata_filter = {
            "source_file": {"$in": source_files}
        }
        
        return await self.answer_question(
            question=question,
            max_sources=max_sources,
            min_similarity=min_similarity,
            metadata_filter=metadata_filter
        )
    
    async def get_related_questions(
        self,
        topic: str,
        max_questions: int = 8,
        min_similarity: float = 0.2
    ) -> List[str]:
        """
        Get related questions/contexts based on a topic.
        
        Args:
            topic: The topic to find related questions for
            max_questions: Maximum number of questions to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of related question contexts
        """
        try:
            search_results = await self.search_engine.search(
                query=topic,
                top_k=max_questions * 2,
                min_similarity=min_similarity
            )
            
            # Extract unique contexts/questions
            questions = []
            seen_contexts = set()
            
            for result in search_results:
                context = result.context.strip()
                if context and context not in seen_contexts and len(context) > 10:
                    questions.append(context)
                    seen_contexts.add(context)
                    
                    if len(questions) >= max_questions:
                        break
            
            logger.info(f"Found {len(questions)} related questions for topic: {topic}")
            return questions
            
        except Exception as e:
            logger.error(f"Error getting related questions: {e}")
            return []
    
    async def find_similar_responses(
        self,
        response_text: str,
        max_similar: int = 5,
        min_similarity: float = 0.4
    ) -> List[SearchResult]:
        """
        Find Ray Peat responses similar to given text.
        
        Args:
            response_text: Text to find similar responses for
            max_similar: Maximum number of similar responses
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar SearchResult objects
        """
        try:
            # Use the response text as a query to find similar responses
            search_results = await self.search_engine.search(
                query=response_text,
                top_k=max_similar,
                min_similarity=min_similarity
            )
            
            logger.info(f"Found {len(search_results)} similar responses")
            return search_results
            
        except Exception as e:
            logger.error(f"Error finding similar responses: {e}")
            return []
    
    async def _generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        """Generate an answer using the LLM with retrieved context."""
        
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            truncation_note = " [Note: truncated]" if source.truncated else ""
            # Trim very long fields to keep prompt concise
            ctx = (source.context or "")[:800]
            resp = (source.ray_peat_response or "")[:1200]
            context_parts.append(
                f"SOURCE {i} | file: {source.source_file} | sim: {source.similarity_score:.3f}{truncation_note}\n"
                f"Context:\n{ctx}\n"
                f"Response:\n{resp}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""You are an expert on Ray Peat's bioenergetic approach. Answer the user's question strictly from the SOURCES below.

Question: {question}

SOURCES:
{context}

Requirements:
- Use only the SOURCES. Do not add external knowledge.
- Synthesize a clear, well-structured answer with headings and bullet points where helpful.
- Include 2-4 short quoted snippets in quotes when directly citing.
- Cite sources inline as [S1], [S2], etc., matching SOURCE indices.
- If information is insufficient or contradictory, state this explicitly.
- End with a short 1-2 sentence summary.

Output format:
1) Answer
2) Key citations (list, e.g., [S1], [S3])
3) Source mapping: [S1]=<file>, [S2]=<file>, ...
"""

        try:
            answer = await self._call_gemini_llm(prompt)
            return answer or "I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    async def _call_gemini_llm(self, prompt: str) -> Optional[str]:
        """Call Gemini LLM with smart continuation to avoid truncation."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        async def call_once(user_prompt: str) -> tuple[str, str]:
            """One-shot call, return (text, finish_reason)."""
            # Prefer official SDK if available; fallback to HTTP
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model_name = self.llm_model if self.llm_model.startswith("gemini") else f"models/{self.llm_model}"
                model = genai.GenerativeModel(model_name)
                resp = await asyncio.to_thread(
                    model.generate_content,
                    user_prompt,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 4096,
                        "top_p": 0.8,
                        "top_k": 40,
                    },
                )
                text = getattr(resp, "text", "") or ""
                finish = ""
                try:
                    finish = str(resp.candidates[0].finish_reason)  # type: ignore[attr-defined]
                except Exception:
                    finish = ""
                if text:
                    return text, finish
            except Exception as _sdk_err:
                logger.debug(f"Gemini SDK unavailable or failed, using HTTP fallback: {_sdk_err}")

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": settings.GEMINI_API_KEY}
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 4096,
                    "topP": 0.8,
                    "topK": 40,
                },
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            text = ""
                            finish = ""
                            try:
                                candidates = result.get("candidates", []) if isinstance(result, dict) else []
                                if candidates:
                                    first = candidates[0]
                                    finish = str(first.get("finishReason", ""))
                                    content = first.get("content", {})
                                    parts = content.get("parts", []) if isinstance(content, dict) else []
                                    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                                    text = "".join(texts)
                            except Exception:
                                pass
                            if not text and isinstance(result, dict):
                                text = result.get("text") or result.get("output_text") or ""
                            return text, finish
                        else:
                            error_text = await response.text()
                            logger.error(f"LLM API Error {response.status}: {error_text}")
                            return "", ""
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                return "", ""

        # Continuation loop
        accumulated = ""
        loops = 0
        max_chars = 12000
        text, finish = await call_once(prompt)
        accumulated += text or ""
        while loops < 4 and len(accumulated) < max_chars:
            # If finish indicates max tokens or the text seems cut mid-sentence, continue
            seems_cut = not accumulated.strip().endswith(('.', '"', "'", '}', ']', ')'))
            if finish and finish.upper() != 'MAX_TOKENS' and not seems_cut:
                break
            loops += 1
            tail = accumulated[-600:]
            cont_prompt = (
                "Continue the previous answer. Continue seamlessly without repeating.\n"
                f"Context tail: {tail}"
            )
            more, finish = await call_once(cont_prompt)
            if not more:
                break
            accumulated += ("\n" if not accumulated.endswith("\n") else "") + more
        return accumulated or None
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the underlying search engine."""
        return self.search_engine.get_corpus_stats()

# Global instance for the API (same pattern as the original)
rag_system = PineconeRAG()


