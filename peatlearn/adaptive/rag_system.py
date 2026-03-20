#!/usr/bin/env python3
"""
Real RAG System using Ray Peat Knowledge Base with Vector Search + LLM
Provides detailed, source-based responses with proper retrieval.

Note: Uses Pinecone-backed vector search by default. Legacy file-based
RAG under `inference/backend/rag` is deprecated.
"""

import asyncio
import aiohttp
import os
import re
import sys
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Module-level reranking constants (built once, reused across calls)
_RERANK_STOP = {
    "the","a","an","and","or","of","to","in","is","it","on","for","with","as","by",
    "that","this","are","be","at","from","about","into","over","under","than","then",
    "but","if","so","not",
}

def _tok(text: str) -> list:
    """Tokenize text, removing stop words."""
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-']+", (text or "").lower()) if t not in _RERANK_STOP]

# Load environment variables
load_dotenv()

# Prefer Pinecone-backed search; fall back gracefully if unavailable
try:
    from peatlearn.rag.vector_search import PineconeVectorSearch as RayPeatVectorSearch, SearchResult
    from config.settings import settings
except Exception:
    RayPeatVectorSearch = None
    SearchResult = None
    settings = None

@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: List[Any] = None
    confidence: float = 0.0
    query: str = ""

class RayPeatRAG:
    """RAG system for answering questions about Ray Peat's work using vector search + LLM."""
    
    def __init__(self, search_engine=None):
        """Initialize the RAG system."""
        # Initialize Pinecone-backed search by default
        self.search_engine = search_engine or (RayPeatVectorSearch() if RayPeatVectorSearch else None)
        self.llm_model = "gemini-2.5-flash-lite"
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.search_engine:
            print("Warning: Vector search engine not available. Using fallback mode.")
        if not self.api_key:
            print("Warning: Google API key not found. Using fallback mode.")
        
    def get_rag_response(self, query: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Get response using proper RAG with vector search
        
        Args:
            query: User's question
            user_profile: User's learning profile for adaptation
            
        Returns:
            Detailed response with sources from Ray Peat's work
        """
        if not self.search_engine or not self.api_key:
            return self._fallback_response(query)

        try:
            return self._get_rag_response_sync(query, user_profile)
        except RuntimeError as e:
            err = str(e)
            if "RATE_LIMITED_DAILY" in err:
                return "The Gemini API daily quota has been reached. Quotas reset at midnight Pacific time. Please try again later, or upgrade to a paid Gemini API plan at aistudio.google.com."
            if "RATE_LIMITED" in err:
                return "The Gemini API is rate-limited right now (too many requests per minute). Please wait a minute and try again."
            import traceback; traceback.print_exc()
            return self._fallback_response(query)
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._fallback_response(query)
    
    def _get_rag_response_sync(self, query: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Fully synchronous RAG path — safe to call from Streamlit or any context."""
        gemini_headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        # --- Step 1: Generate query embedding via Gemini REST (sync) ---
        emb_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        emb_resp = requests.post(
            emb_url,
            json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": query}]}},
            headers=gemini_headers,
            timeout=30,
        )
        if emb_resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {emb_resp.status_code}: {emb_resp.text[:200]}")
        embedding = emb_resp.json().get("embedding", {}).get("values")
        if not embedding:
            raise RuntimeError("Empty embedding returned")

        # --- Step 2: Query Pinecone — fetch more candidates for reranking ---
        pinecone_resp = self.search_engine.index.query(
            vector=embedding,
            top_k=20,
            include_metadata=True,
        )
        matches = pinecone_resp.get("matches", [])

        # Convert to dicts, filter low-similarity matches
        candidates = []
        for m in matches:
            if m.get("score", 0) < 0.3:
                continue
            meta = m.get("metadata", {})
            candidates.append({
                "id": m.get("id", ""),
                "source_file": meta.get("source_file", ""),
                "context": meta.get("context", ""),
                "ray_peat_response": meta.get("ray_peat_response", ""),
                "score": m.get("score", 0),
            })

        if not candidates:
            return "I couldn't find relevant information on that topic in Ray Peat's work. Try rephrasing or ask about metabolism, thyroid, hormones, or nutrition."

        # --- Step 2b: Rerank: 70% vector similarity + 30% keyword overlap, dedupe by source ---
        q_vocab = set(_tok(query)) or set(re.findall(r"[a-zA-Z]+", query.lower()))
        for c in candidates:
            toks = _tok(f"{c['context']} {c['ray_peat_response']}")
            overlap = len(q_vocab.intersection(toks)) / max(1, len(q_vocab)) if toks else 0.0
            c["rerank_score"] = 0.7 * c["score"] + 0.3 * overlap

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Keep best 8, one per source file for diversity
        seen, sources = set(), []
        for c in candidates:
            key = c["source_file"].strip()
            if key in seen:
                continue
            sources.append(c)
            seen.add(key)
            if len(sources) >= 8:
                break
        # Backfill if fewer than 8 unique sources
        if len(sources) < 8:
            for c in candidates:
                if c not in sources:
                    sources.append(c)
                    if len(sources) >= 8:
                        break

        if not sources:
            return "I couldn't find relevant information on that topic in Ray Peat's work."

        # --- Step 3: Build prompt ---
        context_parts = []
        for i, s in enumerate(sources, 1):
            context_parts.append(
                f"Source {i} (from {s['source_file']}):\n"
                f"Context: {s['context']}\n"
                f"Ray Peat's response: {s['ray_peat_response']}"
            )
        context = "\n---\n".join(context_parts)
        prompt = self._create_adaptive_prompt(query, context, user_profile)

        # --- Step 4: Call Gemini LLM via REST (avoids SDK retry wrapping) ---
        _all_models = [self.llm_model, "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
        seen_models: set = set()
        models_to_try = [m for m in _all_models if not (m in seen_models or seen_models.add(m))]
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096, "topP": 0.8, "topK": 40},
        }
        answer = ""
        rate_limited = False
        daily_limit_hit = False
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            resp = requests.post(url, json=payload, headers=gemini_headers, timeout=60)
            if resp.status_code == 429:
                rate_limited = True
                # Distinguish RPM vs daily quota to decide whether to wait or skip
                err_body = resp.text.lower()
                if "daily" in err_body or "per day" in err_body or "quota_exceeded" in err_body:
                    daily_limit_hit = True
                    continue  # Daily quota per-model — try next
                # RPM limit: wait 30 s then retry this model once
                time.sleep(30)
                resp = requests.post(url, json=payload, headers=gemini_headers, timeout=60)
                if resp.status_code == 429:
                    daily_limit_hit = True
                    continue
                if resp.status_code in (404, 400):
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:200]}")
            elif resp.status_code in (404, 400):
                continue  # Model not found or bad request — try next in cascade
            elif resp.status_code != 200:
                raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:200]}")
            try:
                answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                answer = ""
            if answer:
                break

        if not answer and rate_limited:
            if daily_limit_hit:
                raise RuntimeError("RATE_LIMITED_DAILY")
            raise RuntimeError("RATE_LIMITED")

        # --- Step 5: Append sources footer ---
        source_info = "\n\n📚 Sources:\n" + "".join(
            f"{i}. {s['source_file']} (relevance: {s['score']:.2f})\n"
            for i, s in enumerate(sources, 1)
        )
        return answer + source_info

    async def _answer_question_async(
        self, 
        question: str, 
        user_profile: Optional[Dict[str, Any]] = None,
        max_sources: int = 5,
        min_similarity: float = 0.3
    ) -> RAGResponse:
        """
        Answer a question using RAG approach with vector search.
        """
        
        # Step 1: Retrieve relevant passages using vector search
        search_results = await self.search_engine.search(
            query=question,
            top_k=max_sources,
            min_similarity=min_similarity
        )
        
        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant information about that topic in Ray Peat's work. Please try rephrasing your question or asking about topics like metabolism, thyroid function, hormones, or nutrition.",
                sources=[],
                confidence=0.0,
                query=question
            )
        
        # Step 2: Generate answer using retrieved context
        answer = await self._generate_answer(question, search_results, user_profile)
        
        # Step 3: Calculate confidence based on similarity scores
        avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
        confidence = min(avg_similarity * 1.2, 1.0)  # Scale similarity to confidence
        
        return RAGResponse(
            answer=answer,
            sources=search_results,
            confidence=confidence,
            query=question
        )
    
    async def _generate_answer(self, question: str, sources: List[Any], user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Generate an answer using the LLM with retrieved context."""
        
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"Source {i} (from {source.source_file}):\n"
                f"Context: {source.context}\n"
                f"Ray Peat's response: {source.ray_peat_response}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create adaptive prompt based on user profile
        prompt = self._create_adaptive_prompt(question, context, user_profile)

        try:
            answer = await self._call_gemini_llm(prompt)
            if answer:
                # Add source information to the answer (UI parses this header)
                source_info = f"\n\n📚 Sources:\n"
                for i, source in enumerate(sources, 1):
                    source_info += f"{i}. {source.source_file} (relevance: {source.similarity_score:.2f})\n"
                
                return answer + source_info
            else:
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def _create_adaptive_prompt(self, question: str, context: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Create a prompt adapted to the user's learning profile"""
        
        # Classify question depth from signals
        q_lower = question.lower()
        detail_signals = ["explain", "detail", "elaborate", "in depth", "thoroughly", "how does", "why does", "mechanism", "relationship between", "tell me more", "comprehensive"]
        wants_detail = any(s in q_lower for s in detail_signals)

        if wants_detail:
            length_instruction = "Aim for 200–300 words across 2–4 tight paragraphs."
            followup_instruction = """End with ONE practical follow-up question a real person would naturally ask next — something like "Want to know what's actually slowing yours down?" or "Curious about the thyroid connection?" NOT a quiz question or academic prompt."""
        else:
            length_instruction = "Aim for 60–100 words across 2–3 short paragraphs. Be punchy. Every sentence should earn its place."
            followup_instruction = """End with ONE very short follow-up question that feels like a natural conversation next step — e.g. "Curious how to boost yours?" or "Want to know what's blocking it?" Keep it under 12 words. Sound like a person, not a professor."""

        base_prompt = f"""You are Ray Peat AI — a knowledgeable guide to Ray Peat's bioenergetic philosophy. Explain his ideas clearly and confidently, always attributing them to him ("Peat argued...", "In his view...", "He was direct about this..."). You're an informed friend, not a polemicist.

Use ONLY the provided sources to answer. Do not invent anything.

Question: {question}

SOURCES:
{context}

Rules:
- {length_instruction}
- Write in plain prose. No bullet points. No headers unless the question genuinely covers multiple distinct topics.
- Vary your opening — never use the same phrasing twice. Don't rely on clichés like "Forget what they tell you." Start from a different angle each time: a fact, a contrast, a direct answer, Peat's own words.
- Attribute Peat's views to him clearly ("Peat saw this as...", "He argued...", "His take was..."). This distinguishes his perspective from mainstream medicine without being aggressive toward it.
- Cite sources inline as [S1], [S2] etc. Weave citations naturally into sentences — don't cluster them at the start of a paragraph.
- Lead with the most interesting point. Counterintuitive ideas are worth surfacing early.
- Use Peat's exact words only when they're genuinely striking. Avoid academic filler phrases.
- {followup_instruction}
- If the sources don't cover the question, say so in one sentence."""

        return base_prompt + "\n\nAnswer:"
    
    async def _call_gemini_llm(self, prompt: str) -> Optional[str]:
        """Call Gemini LLM with continuation to reduce truncation."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        async def call_once(user_prompt: str) -> tuple[str, str]:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)
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
                return text, finish
            except Exception:
                pass

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096, "topP": 0.8, "topK": 40},
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            text = ""
                            finish = ""
                            try:
                                candidates = result.get("candidates", [])
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
                            return "", ""
            except Exception:
                return "", ""

        accumulated = ""
        loops = 0
        max_chars = 12000
        first, finish = await call_once(prompt)
        accumulated += first or ""
        while loops < 4 and len(accumulated) < max_chars:
            seems_cut = not accumulated.strip().endswith(('.', '"', "'", '}', ']', ')'))
            if finish and finish.upper() != 'MAX_TOKENS' and not seems_cut:
                break
            loops += 1
            tail = accumulated[-600:]
            cont = f"Continue the previous answer. Continue seamlessly without repeating.\nContext tail: {tail}"
            more, finish = await call_once(cont)
            if not more:
                break
            accumulated += ("\n" if not accumulated.endswith("\n") else "") + more
        return accumulated or None
    
    def _fallback_response(self, query: str) -> str:
        """Provide a fallback response when full RAG is unavailable."""
        return (
            f'Sorry, I ran into a technical issue processing your question: "{query}"\n\n'
            "The knowledge base (Pinecone) and API key are configured, but the language model "
            "failed to generate a response. This usually means:\n\n"
            "- All Gemini model tiers are rate-limited (try again in a few minutes)\n"
            "- A temporary API outage\n\n"
            "Ask me again in a moment and it should work."
        )
