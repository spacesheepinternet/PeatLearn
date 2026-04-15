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
        self.llm_model = "gemini-2.5-flash"
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.search_engine:
            print("Warning: Vector search engine not available. Using fallback mode.")
        if not self.api_key:
            print("Warning: Google API key not found. Using fallback mode.")
        
    def get_rag_response(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_sources: Optional[int] = None,
    ) -> str:
        """
        Get response using proper RAG with vector search.

        Args:
            query: User's question
            user_profile: User's learning profile for adaptation
            chat_history: Recent conversation turns (list of {role, content} dicts)
            max_sources: Max number of source chunks to include. If None (the
                default), the RAG auto-scales based on query complexity — hard
                multi-concept or ambiguous queries get 12 sources, everything
                else gets 8. Callers can override with an explicit int (3–15).

        Returns:
            Detailed response with sources from Ray Peat's work
        """
        if not self.search_engine or not self.api_key:
            return self._fallback_response(query)

        if max_sources is None:
            max_sources = self._estimate_max_sources(query)

        try:
            return self._get_rag_response_sync(query, user_profile, chat_history, max_sources)
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

    @staticmethod
    def _estimate_max_sources(query: str) -> int:
        """Heuristic: scale max_sources up for hard/synthesis queries.

        Returns 12 for queries that look hard (multi-concept synthesis, nuance/
        exception framing, or very short/ambiguous single-word queries) and 8
        for everything else. Used when callers don't pass an explicit max_sources.

        Rationale: benchmark showed completeness (8.07) was the lowest rubric
        dimension — hard questions were retrieval-starved at 8 sources.
        """
        q = (query or "").lower().strip()
        if not q:
            return 8

        n_words = len(q.split())

        # Ambiguous short/single-word queries — widen retrieval to disambiguate
        if n_words <= 3:
            return 12

        # Multi-concept synthesis markers — asking the model to connect things
        synthesis = (
            "relationship", "link ", "connect the", "connect ",
            "tie together", "tie these", "tie ", "fit together",
            "how do these", "how does it all",
        )
        # Nuance / limits / exception framing — sparse in corpus, needs wide net
        nuance = (
            "when does", "are there cases", "cases where", "limits",
            "uncertainties", "might actually", "exceptions to",
            "counterexample", "are there any",
        )

        # 2+ "and" usually means listing 3+ concepts
        if q.count(" and ") >= 2:
            return 12
        if any(m in q for m in synthesis):
            return 12
        if any(m in q for m in nuance):
            return 12

        return 8

    def _get_rag_response_sync(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_sources: int = 8,
    ) -> str:
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

        # --- Step 1b: HyDE — re-embed a hypothetical Peat-style answer for better retrieval ---
        hyde_text = query  # fallback default
        try:
            hyde_prompt = (
                "You are Ray Peat. Answer the question below in 3-4 sentences in your exact voice. "
                "Draw on your specific vocabulary: bioenergetics, oxidative metabolism, cellular respiration, "
                "thyroid, T3, T4, PUFAs, progesterone, cortisol, ATP, mitochondria, CO2, glucose oxidation, "
                "pro-metabolic, anti-metabolic, estrogen dominance, serotonin, lactic acid, "
                "unsaturated fatty acids, ray peat diet. Be mechanistic and specific — name the biochemical "
                "pathway or hormone involved. Do not hedge or use filler phrases. "
                f"Question: {query}"
            )
            hyde_resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
                json={
                    "contents": [{"role": "user", "parts": [{"text": hyde_prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
                },
                headers=gemini_headers,
                timeout=15,
            )
            if hyde_resp.status_code == 200:
                hyde_text = hyde_resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip() or query
        except Exception:
            pass

        hyde_emb = requests.post(
            emb_url,
            json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": hyde_text}]}},
            headers=gemini_headers,
            timeout=30,
        )
        if hyde_emb.status_code == 200:
            hyde_vals = hyde_emb.json().get("embedding", {}).get("values")
            if hyde_vals:
                embedding = hyde_vals  # overwrite — Pinecone query uses this transparently

        # --- Step 2: Query Pinecone with two-pass diversity strategy ---
        # Some files have 50+ near-identical chunks that drown out diverse results.
        # Pass 1: fetch results, cap per file. Pass 2: if too few unique files,
        # re-query excluding dominant files to surface diverse sources.
        _MAX_CANDIDATES_PER_FILE = 4
        _MIN_UNIQUE_FILES = max(6, max_sources)

        def _collect_candidates(matches_list, existing_candidates=None, existing_counts=None):
            cands = list(existing_candidates or [])
            counts = dict(existing_counts or {})
            seen_ids = {c["id"] for c in cands}
            for m in matches_list:
                if m.get("score", 0) < 0.15:
                    continue
                mid = m.get("id", "")
                if mid in seen_ids:
                    continue
                meta = m.get("metadata", {})
                sf = meta.get("source_file", "")
                counts[sf] = counts.get(sf, 0) + 1
                if counts[sf] > _MAX_CANDIDATES_PER_FILE:
                    continue
                seen_ids.add(mid)
                cands.append({
                    "id": mid,
                    "source_file": sf,
                    "context": meta.get("context", ""),
                    "ray_peat_response": meta.get("ray_peat_response", ""),
                    "score": m.get("score", 0),
                })
            return cands, counts

        pinecone_resp = self.search_engine.index.query(
            vector=embedding, top_k=80, include_metadata=True,
        )
        candidates, file_counts = _collect_candidates(pinecone_resp.get("matches", []))

        # Pass 2: if not enough unique files, re-query excluding dominant files
        unique_files = len(file_counts)
        if unique_files < _MIN_UNIQUE_FILES:
            dominant = [f for f, n in file_counts.items() if n >= _MAX_CANDIDATES_PER_FILE]
            if dominant:
                # Pinecone metadata filter: exclude dominant files
                exclude_filter = {"source_file": {"$nin": dominant}}
                try:
                    pass2_resp = self.search_engine.index.query(
                        vector=embedding, top_k=40, include_metadata=True,
                        filter=exclude_filter,
                    )
                    candidates, file_counts = _collect_candidates(
                        pass2_resp.get("matches", []), candidates, file_counts
                    )
                except Exception:
                    pass  # if filter not supported, keep what we have

        if not candidates:
            return "I couldn't find relevant information on that topic in Ray Peat's work. Try rephrasing or ask about metabolism, thyroid, hormones, or nutrition."

        # --- Step 2b: Rerank using cross-encoder (falls back to keyword overlap) ---
        from peatlearn.rag.reranker import rerank as _rerank
        candidates = _rerank(query, candidates)

        # MMR-style: penalise repeated source files AND content types for diversity
        source_counts: dict = {}
        type_counts: dict = {}
        sources = []
        remaining = list(candidates)
        while remaining and len(sources) < max_sources:
            best_idx, best_score = 0, float("-inf")
            for idx, c in enumerate(remaining):
                key = c["source_file"].strip()
                # Content type = top-level directory (e.g. "01_Audio_Transcripts")
                ctype = key.split("/")[0] if "/" in key else key.split("\\")[0] if "\\" in key else ""
                n_file = source_counts.get(key, 0)
                n_type = type_counts.get(ctype, 0)
                raw = c["rerank_score"]
                # Penalise repeated files (heavy) and repeated content types (lighter)
                adjusted = raw - 0.3 * n_file - 0.05 * n_type
                if adjusted > best_score:
                    best_score, best_idx = adjusted, idx
            winner = remaining.pop(best_idx)
            sources.append(winner)
            key = winner["source_file"].strip()
            ctype = key.split("/")[0] if "/" in key else key.split("\\")[0] if "\\" in key else ""
            source_counts[key] = source_counts.get(key, 0) + 1
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            # Hard cap: never take more than 2 chunks from the same source
            if source_counts[key] >= 2:
                remaining = [c for c in remaining if c["source_file"].strip() != key]

        if not sources:
            return "I couldn't find relevant information on that topic in Ray Peat's work."

        # --- Step 3: Build prompt ---
        context_parts = []
        for i, s in enumerate(sources, 1):
            context_parts.append(
                f"[S{i}] {s['source_file']} (relevance {s['score']:.2f})\n"
                f"Topic context: {s['context']}\n"
                f"Peat's words: {s['ray_peat_response']}"
            )
        context = "\n---\n".join(context_parts)
        prompt = self._create_adaptive_prompt(
            query, context, user_profile, chat_history, n_sources=len(sources)
        )

        # --- Step 4: Call Gemini LLM via REST (avoids SDK retry wrapping) ---
        _all_models = [self.llm_model, "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
        seen_models: set = set()
        models_to_try = [m for m in _all_models if not (m in seen_models or seen_models.add(m))]
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 4096, "topP": 0.85, "topK": 40, "candidateCount": 1},
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
                if resp.status_code in (404, 400, 503, 529):
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:200]}")
            elif resp.status_code in (404, 400):
                continue  # Model not found or bad request — try next in cascade
            elif resp.status_code in (503, 529):
                continue  # Model overloaded/unavailable — try next in cascade
            elif resp.status_code != 200:
                raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:200]}")
            try:
                answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                answer = ""
            if answer:
                break

        # --- Step 4b: Groq fallback when all Gemini models are rate-limited or unavailable ---
        if not answer:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    groq_resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.25,
                            "max_tokens": 4096,
                        },
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        timeout=60,
                    )
                    if groq_resp.status_code == 200:
                        answer = groq_resp.json()["choices"][0]["message"]["content"]
                except Exception:
                    pass

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
        prompt = self._create_adaptive_prompt(
            question, context, user_profile, n_sources=len(sources)
        )

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
    
    def _create_adaptive_prompt(
        self,
        question: str,
        context: str,
        user_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        n_sources: int = 8,
    ) -> str:
        """Create a prompt tiered by question complexity and retrieved source count.

        Three tiers:
        - Deep synthesis (n_sources >= 12 and not single-word): 260-360 words,
          unpack sub-questions, use the wider source set.
        - Ambiguous short (query is 1-3 words): 160-220 words, pick ONE angle
          and go deep rather than laundry-listing.
        - Default: current two-tier 120-180 / 180-260 logic.
        """

        q_lower = question.lower()
        n_words = len(q_lower.split())

        detail_signals = [
            "explain", "detail", "elaborate", "in depth", "thoroughly",
            "how does", "why does", "mechanism", "relationship between",
            "tell me more", "comprehensive", "deep dive",
        ]
        wants_detail = any(s in q_lower for s in detail_signals)

        is_ambiguous_short = n_words <= 3
        is_deep_synthesis = n_sources >= 12 and not is_ambiguous_short

        extra_rules = ""

        if is_deep_synthesis:
            length_instruction = (
                "Write up to 360 words across 3–5 tight paragraphs — but go shorter "
                "when the sources don't justify more. Brevity with precision beats a "
                "padded answer. Every sentence should add new information."
            )
            extra_rules = (
                "- Answer the literal question FIRST, in the opening sentence or two. "
                "Only then add depth, mechanism, or related angles. Never dodge the framing.\n"
                "- If the honest answer is \"Peat didn't really engage with this framing\" "
                "or \"he was consistent against it with no real exceptions\", say so "
                "directly and keep the answer shorter. Do not invent nuance the sources don't show.\n"
                "- When the sources genuinely hedge or show tension, surface it explicitly — "
                "say \"Peat sometimes emphasized X, in other contexts Y\" rather than smoothing it over."
            )
            followup_instruction = (
                'End with ONE probing follow-up that opens a new mechanistic thread — '
                'something a researcher would ask next. Keep it under 22 words.'
            )
        elif is_ambiguous_short:
            length_instruction = (
                "Write 160–220 words picking ONE specific angle. The question is a bare "
                "term, so the user wants depth, not a tour of every aspect."
            )
            extra_rules = (
                "- Pick the most distinctively Peat-ian angle the sources offer — a specific "
                "mechanism, hormone, or food he emphasized — and make that the spine. "
                "Mention other angles only briefly to frame the chosen one.\n"
                "- Resist laundry-list framing. Don't itemise every related topic."
            )
            followup_instruction = (
                'End with ONE short follow-up (≤15 words) that drills further into the angle you took.'
            )
        elif wants_detail:
            length_instruction = "Write 180–260 words across 2–4 tight paragraphs. Go deep on mechanism where sources allow."
            followup_instruction = (
                'End with ONE specific follow-up question that would genuinely deepen understanding — '
                'something a curious person would actually wonder next. Keep it under 20 words.'
            )
        else:
            length_instruction = "Write 120–180 words. Be precise and punchy. One clear idea per sentence."
            followup_instruction = (
                'End with ONE short, curious follow-up question (≤15 words) that feels like a natural '
                'next step in the conversation — not a quiz question.'
            )

        # Build optional conversation history block
        history_block = ""
        if chat_history:
            recent = [m for m in chat_history if m.get("role") in ("user", "assistant")][-6:]
            if recent:
                lines = []
                for m in recent:
                    role_label = "User" if m["role"] == "user" else "Assistant"
                    # Truncate long assistant messages to keep prompt focused
                    content = (m.get("content") or "")[:400]
                    lines.append(f"{role_label}: {content}")
                history_block = "\n\nCONVERSATION SO FAR (for context — do not repeat):\n" + "\n".join(lines)

        base_prompt = f"""You are Ray Peat AI — a knowledgeable, warm guide to Ray Peat's bioenergetic philosophy. You speak like an informed friend who has read everything Peat ever wrote: direct, curious, never preachy.

Use ONLY the provided SOURCES to answer. Never add external knowledge or invent anything.{history_block}

CURRENT QUESTION: {question}

SOURCES:
{context}

Rules:
- {length_instruction}
- Never open with "Certainly", "Great question", "Of course", or any filler. Start directly: a fact, a contrast, a direct answer, or Peat's own words.
- Write in plain prose. No bullet points. No headers unless the question genuinely covers multiple distinct topics.
- Attribute every claim to Peat explicitly: "Peat argued...", "He was direct about this...", "In his view...", "His take was..."
- Cite sources inline as [S1], [S2] etc. — matching the source numbers in the SOURCES block. Weave citations naturally into sentences, never clustered.
- If sources conflict on a point, acknowledge the tension explicitly rather than picking one silently.
- Lead with the most interesting or counterintuitive point. Surface it early.
- Use Peat's exact words only when they're genuinely striking. Avoid academic filler.
- When sources contain practical advice (foods, supplements, techniques, or lifestyle changes Peat recommended), include it concretely — don't stop at theory.
{extra_rules}
- {followup_instruction}
- If the sources don't cover the question, say so in one sentence and suggest a related angle the user might find useful."""

        return base_prompt + "\n\nAnswer:"
    
    async def _call_gemini_llm(self, prompt: str) -> Optional[str]:
        """Call Gemini LLM with continuation to reduce truncation."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        async def call_once(user_prompt: str) -> tuple[str, str]:
            try:
                from google import genai as _genai  # type: ignore
                from google.genai import types as _gtypes
                _client = _genai.Client(api_key=self.api_key)
                model_name = self.llm_model if self.llm_model.startswith("gemini") else f"models/{self.llm_model}"
                resp = await asyncio.to_thread(
                    _client.models.generate_content,
                    model=model_name,
                    contents=user_prompt,
                    config=_gtypes.GenerateContentConfig(
                        temperature=0.25,
                        max_output_tokens=4096,
                        top_p=0.85,
                        top_k=40,
                    ),
                )
                text = resp.text or ""
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
                "generationConfig": {"temperature": 0.25, "maxOutputTokens": 4096, "topP": 0.85, "topK": 40, "candidateCount": 1},
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
