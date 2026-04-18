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

        # --- Step 0: Temporal guard — auto-ABSTAIN on post-2022 topics ---
        from peatlearn.rag.temporal_guard import check_temporal as _check_temporal
        temporal_reason = _check_temporal(query)
        if temporal_reason:
            confidence_footer = f"\n\n\U0001f512 Confidence: ABSTAIN | Temporal guard: {temporal_reason}"
            return (
                "I can't answer this question because it references a topic "
                "that emerged after Ray Peat's death in October 2022. Peat's "
                "corpus does not cover this subject, and any answer would be "
                "speculation rather than grounded in his actual work.\n\n"
                f"Reason: {temporal_reason}"
                + confidence_footer
            )

        # --- Step 0.5: Query vocabulary normalization ---
        # Map colloquial terms ("carbs", "seed oils", "gut health") to Peat's
        # corpus vocabulary so embedding, HyDE, and cross-encoder all operate
        # on terms that actually exist in the corpus.
        from peatlearn.rag.query_normalizer import normalize_query as _normalize
        search_query = _normalize(query)

        # --- Step 1: Generate query embedding via Gemini REST (sync) ---
        emb_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        emb_resp = requests.post(
            emb_url,
            json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": search_query}]}},
            headers=gemini_headers,
            timeout=30,
        )
        if emb_resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {emb_resp.status_code}: {emb_resp.text[:200]}")
        embedding = emb_resp.json().get("embedding", {}).get("values")
        if not embedding:
            raise RuntimeError("Empty embedding returned")
        raw_query_embedding = list(embedding)  # preserve for HyDE divergence fallback

        # --- Step 1b: Dual HyDE — academic + email style, sequential to avoid RPM bursts ---
        # Academic HyDE: mechanistic Peat vocabulary → surfaces written articles + transcripts.
        # Email HyDE:    direct Q&A style → surfaces email corpus (2.6k+ vectors of direct
        #                Peat quotes invisible to academic-style embeddings).
        # Both calls are sequential (not concurrent) to prevent rate-limit silent failures.
        # Each call validates output: must be ≥ 20 words and meaningfully different from query;
        # a single retry fires on 429 after a short wait.
        _hyde_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

        def _robust_hyde(prompt: str) -> str | None:
            """Call Flash-Lite for a HyDE answer. Returns text or None on failure."""
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
            }
            for attempt in range(2):
                try:
                    r = requests.post(_hyde_url, json=payload, headers=gemini_headers, timeout=20)
                    if r.status_code == 429:
                        time.sleep(8)
                        continue
                    if r.status_code != 200:
                        return None
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    # Validate: must be ≥ 20 words and not a trivial repeat of the query
                    if text and len(text.split()) >= 20 and text.lower() != query.lower():
                        return text
                except Exception:
                    pass
            return None

        # Academic HyDE (always fires — replaces original single HyDE)
        academic_prompt = (
            "You are Ray Peat. Answer the question below in 3-4 sentences in your exact voice. "
            "Draw on your specific vocabulary: bioenergetics, oxidative metabolism, cellular respiration, "
            "thyroid, T3, T4, PUFAs, progesterone, cortisol, ATP, mitochondria, CO2, glucose oxidation, "
            "pro-metabolic, anti-metabolic, estrogen dominance, serotonin, lactic acid, "
            "unsaturated fatty acids, ray peat diet. Be mechanistic and specific — name the biochemical "
            "pathway or hormone involved. Do not hedge or use filler phrases. "
            f"Question: {search_query}"
        )
        _academic_hyde_result = _robust_hyde(academic_prompt)
        hyde_text = _academic_hyde_result or query

        academic_hyde_embedding = None  # saved for confidence scoring (Phase 2)
        hyde_emb = requests.post(
            emb_url,
            json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": hyde_text}]}},
            headers=gemini_headers,
            timeout=30,
        )
        if hyde_emb.status_code == 200:
            hyde_vals = hyde_emb.json().get("embedding", {}).get("values")
            if hyde_vals:
                embedding = hyde_vals  # primary embedding used for Pinecone
                if _academic_hyde_result:
                    academic_hyde_embedding = hyde_vals

        # Email HyDE (fires second, after academic — sequential to avoid RPM bursts)
        email_prompt = (
            "You are Ray Peat replying to a direct email question. "
            "Give a short, specific, opinionated answer — 2-3 sentences. "
            "Name specific foods, supplements, or practical recommendations. "
            "Be direct: state what you do or don't recommend, and briefly say why. "
            "No hedging, no academic framing. Write as you would in a personal email reply. "
            f"Question: {search_query}"
        )
        hyde_email_text = _robust_hyde(email_prompt)

        # Embed email HyDE — only if it produced a valid, distinct answer
        email_embedding = None
        if hyde_email_text:
            try:
                email_emb_resp = requests.post(
                    emb_url,
                    json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": hyde_email_text}]}},
                    headers=gemini_headers,
                    timeout=30,
                )
                if email_emb_resp.status_code == 200:
                    ev = email_emb_resp.json().get("embedding", {}).get("values")
                    if ev:
                        email_embedding = ev
            except Exception:
                pass

        # --- Step 1c: HyDE divergence guard ---
        # If the two HyDE embeddings disagree strongly, both are unreliable —
        # a bad HyDE is worse than no HyDE because it pulls retrieval off-topic.
        # Discard both and retrieve from the raw query embedding only.
        if academic_hyde_embedding and email_embedding:
            from peatlearn.rag.confidence import _cosine_similarity
            hyde_cos = _cosine_similarity(academic_hyde_embedding, email_embedding)
            if hyde_cos < 0.55:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    f"HyDE divergence detected (cosine={hyde_cos:.3f}) — "
                    f"discarding both HyDE embeddings, falling back to raw query"
                )
                embedding = raw_query_embedding
                academic_hyde_embedding = None
                email_embedding = None

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

        # Email HyDE pass — fires only for specific food/supplement topics where the email
        # corpus holds direct Peat quotes not well-represented in written articles or audio.
        # Trigger: query contains a known food/supplement term. This is precise — it fires
        # for "milk", "gelatin", "cruciferous", "coconut oil" etc. but NOT for broad topics
        # like "stress", "aging", or hormone-mechanism questions where audio/written articles
        # are the right primary source and email injection causes noise.
        _EMAIL_FOOD_TERMS = {
            "milk", "dairy", "cheese", "cream", "kefir",
            "gelatin", "collagen", "glycine", "broth",
            "crucifer", "broccoli", "kale", "cabbage", "cauliflower", "goitrogen",
            "coconut", "orange juice", "oj", "juice",
            "oyster", "liver", "sardine", "fish",
            "potato", "starch", "carrot", "fruit",
            "coffee", "caffeine",
            "aspirin", "supplement", "vitamin", "mineral", "magnesium", "calcium",
            "progesterone cream", "dhea", "pregnenolone",
            "salt", "sodium", "sugar", "fructose", "sucrose",
        }
        _q_lower = query.lower()
        _query_has_food_term = any(t in _q_lower for t in _EMAIL_FOOD_TERMS)
        _email_pass_needed = email_embedding is not None and _query_has_food_term
        if _email_pass_needed:
            try:
                email_pass_resp = self.search_engine.index.query(
                    vector=email_embedding, top_k=40, include_metadata=True,
                )
                candidates, file_counts = _collect_candidates(
                    email_pass_resp.get("matches", []), candidates, file_counts
                )
            except Exception:
                pass

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

        # --- Step 2c: Confidence scoring — decide if retrieval supports answering ---
        from peatlearn.rag.confidence import score_retrieval as _score_retrieval
        confidence = _score_retrieval(
            candidates,
            academic_hyde_embedding=academic_hyde_embedding,
            email_hyde_embedding=email_embedding,
        )

        # --- Step 2d: Entity grounding check ---
        # If the query mentions a specific entity (e.g. "berberine", "psilocybin")
        # that appears in ZERO retrieved sources, the LLM would fabricate Peat's
        # views on it. Downgrade to ABSTAIN when the majority of key entities
        # are missing and confidence isn't already HIGH.
        from peatlearn.rag.confidence import check_entity_grounding as _check_grounding
        _missing, _should_downgrade = _check_grounding(search_query, candidates)
        if _should_downgrade and confidence.tier != "HIGH":
            confidence.tier = "ABSTAIN"
            confidence.reasons.append(
                f"Key entity not found in any source: {', '.join(_missing)} "
                f"— Peat likely never discussed this specifically"
            )

        if confidence.tier == "ABSTAIN":
            # Short-circuit: refuse to answer, list top weak sources for transparency
            weak_sources = candidates[:3]
            source_lines = "\n".join(
                f"  - {s['source_file']} (rerank score: {s.get('rerank_score', 0):.2f})"
                for s in weak_sources
            )
            reasons_text = "; ".join(confidence.reasons)
            confidence_footer = f"\n\n\U0001f512 Confidence: {confidence.tier} | {reasons_text}"
            return (
                "I don't have sufficient information in Ray Peat's corpus to "
                "answer this question reliably. The retrieved sources are only "
                "weakly related to your query.\n\n"
                f"Reason: {reasons_text}\n\n"
                f"Top (weak) sources found:\n{source_lines}\n\n"
                "Try rephrasing your question, or ask about a topic that Ray "
                "Peat directly addressed in his writings, interviews, or email "
                "correspondence."
                + confidence_footer
            )

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
        # Truncate oversized chunks to prevent one source from monopolising
        # the prompt window — 1200 chars is ~300 tokens, plenty for grounding.
        _CHUNK_CAP = 1200
        context_parts = []
        for i, s in enumerate(sources, 1):
            peat_text = s['ray_peat_response']
            if len(peat_text) > _CHUNK_CAP:
                peat_text = peat_text[:_CHUNK_CAP] + "..."
            context_parts.append(
                f"[S{i}] {s['source_file']} (relevance {s['score']:.2f})\n"
                f"Topic context: {s['context'][:400]}\n"
                f"Peat's words: {peat_text}"
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
        # llama-3.3-70b is more prone to instruction-drift than Gemini, so we
        # prepend an extra grounding instruction specific to this fallback path.
        if not answer:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                groq_prompt = (
                    "CRITICAL: You will be penalised for any claim not directly "
                    "supported by verbatim text in the SOURCES below. If you "
                    "cannot find a supporting quote for a claim, do NOT include "
                    "that claim. If the SOURCES are insufficient to answer the "
                    "question, output exactly: INSUFFICIENT_SOURCES\n\n"
                    + prompt
                )
                try:
                    groq_resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": groq_prompt}],
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

        # --- Step 5: Grounding verifier — strip unsupported claims ---
        from peatlearn.rag.verifier import verify_claims as _verify_claims
        verification = _verify_claims(answer, sources, api_key=self.api_key)
        if verification.unsupported:
            answer = verification.revised_answer
            import logging as _lg
            _lg.getLogger(__name__).warning(
                f"Verifier stripped {len(verification.unsupported)} unsupported claim(s)"
            )

        # --- Step 6: Append sources + confidence footer ---
        source_info = "\n\n📚 Sources:\n" + "".join(
            f"{i}. {s['source_file']} (relevance: {s['score']:.2f})\n"
            for i, s in enumerate(sources, 1)
        )
        reasons_text = "; ".join(confidence.reasons)
        confidence_footer = f"\n\U0001f512 Confidence: {confidence.tier} | {reasons_text}"
        return answer + source_info + confidence_footer

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

        # Build optional conversation history block — USER TURNS ONLY.
        # Prior assistant text is excluded to prevent hallucination propagation:
        # if turn 1 hallucinated, injecting that text into turn 2's context
        # causes the LLM to treat the hallucination as established fact.
        history_block = ""
        if chat_history:
            recent_user = [m for m in chat_history if m.get("role") == "user"][-4:]
            if recent_user:
                lines = [f"User: {(m.get('content') or '')[:200]}" for m in recent_user]
                history_block = "\n\nPRIOR USER QUESTIONS (for topic continuity — do not repeat answers):\n" + "\n".join(lines)

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
- If the question embeds a false or inverted premise (e.g. "Peat recommended X" when the sources show he opposed X, or "Peat endorsed Y" when Y contradicts his core views), reject the premise explicitly in the FIRST sentence — e.g. "That premise doesn't reflect Peat's documented position" — then state what he actually believed. Never answer as if a false premise were true, even partially.
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
