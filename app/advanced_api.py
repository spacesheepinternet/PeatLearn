#!/usr/bin/env python3
"""
Advanced RAG System with Personalization, RL, and Knowledge Graphs

Integrates all sophisticated ML/AI components:
- Neural Collaborative Filtering for recommendations
- Deep Reinforcement Learning for content sequencing
- Knowledge Graph Neural Networks
- Fine-tuned domain embeddings
- Multi-task learning for quiz generation
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import time
import numpy as np
import random
import sqlite3
import json
import uuid

from config.settings import settings
from peatlearn.rag.rag_system import PineconeRAG as RayPeatRAG, RAGResponse
from peatlearn.personalization.quiz_logger import log_quiz_outcome
from peatlearn.personalization.utils import generate_mcq_from_passage
from peatlearn.adaptive.quiz_generator import call_llm_api as llm_call

# Import advanced ML components
try:
    from peatlearn.personalization.neural import (
        AdvancedPersonalizationEngine,
        UserInteraction,
        LearningState,
        personalization_engine
    )
    from peatlearn.personalization.rl_agent import (
        AdaptiveLearningAgent,
        LearningEnvironmentState,
        adaptive_agent
    )
    from peatlearn.personalization.knowledge_graph import (
        AdvancedKnowledgeGraph,
        ray_peat_knowledge_graph
    )

    ADVANCED_ML_AVAILABLE = True

except ImportError as e:
    print(f"Advanced ML components not available: {e}")
    print("Install optional deps: pip install torch-geometric")
    ADVANCED_ML_AVAILABLE = False

# Pydantic models for API
class UserProfile(BaseModel):
    user_id: str
    name: str
    email: Optional[str] = None
    learning_style: Optional[str] = "unknown"
    preferences: Dict[str, Any] = {}

class InteractionData(BaseModel):
    user_id: str
    content_id: str
    interaction_type: str
    performance_score: float
    time_spent: float
    difficulty_level: float
    topic_tags: List[str]
    context: Dict[str, Any] = {}

class QuizRequest(BaseModel):
    user_id: str
    topic: Optional[str] = None
    difficulty_preference: Optional[float] = None
    num_questions: int = 5

class QuizSessionStart(BaseModel):
    user_id: str
    topics: Optional[List[str]] = None
    num_questions: int = 5
    target_difficulty: Optional[float] = None

class QuizAnswer(BaseModel):
    session_id: str
    item_id: str
    chosen_index: int
    time_ms: Optional[int] = None
    user_id: Optional[str] = None

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    exclude_seen: bool = True
    topic_filter: Optional[List[str]] = None

class KnowledgeGraphQuery(BaseModel):
    query: str
    max_expansions: int = 5
    include_related_concepts: bool = True

# Initialize FastAPI app
app = FastAPI(
    title="PeatLearn - Advanced AI Learning Platform",
    description="AI-powered personalized learning system for Ray Peat's bioenergetic knowledge",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (Pinecone-based)
rag_system = RayPeatRAG()

# Lightweight MF recommender loader
MF_MODEL_PATH = Path("data/models/recs/mf_model.npz")
MF_AVAILABLE = False
MF_U = None  # type: ignore
MF_V = None  # type: ignore
MF_USER_TO_IDX = {}
MF_ITEM_TO_IDX = {}

# Simple in-memory cache for recommendations to reduce repeated vector searches
RECS_CACHE: dict[str, dict] = {}
RECS_TTL_SECONDS = 60

# Quiz DB schema and helpers
DB_PATH = Path("data/user_interactions/interactions.db")

def _ensure_quiz_schema() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # quiz_items: item bank
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_items (
            item_id TEXT PRIMARY KEY,
            topic TEXT,
            stem TEXT,
            options TEXT,
            correct_index INTEGER,
            explanation TEXT,
            passage_excerpt TEXT,
            source_file TEXT,
            difficulty_b REAL,
            discrimination_a REAL,
            guessing_c REAL,
            type TEXT,
            validated INTEGER DEFAULT 0
        )
        """
    )
    # sessions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            started_at TEXT,
            finished_at TEXT,
            topics TEXT,
            num_questions INTEGER,
            target_difficulty REAL,
            policy TEXT,
            status TEXT
        )
        """
    )
    # queue of items for a session
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_queue (
            session_id TEXT,
            ord INTEGER,
            item_id TEXT,
            shown INTEGER DEFAULT 0,
            shown_at TEXT,
            PRIMARY KEY (session_id, ord)
        )
        """
    )
    # events per item
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_events (
            session_id TEXT,
            item_id TEXT,
            ord INTEGER,
            shown_at TEXT,
            answered_at TEXT,
            chosen_index INTEGER,
            correct INTEGER,
            response_time_ms INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

def _ensure_adaptive_schema() -> None:
    """Create adaptive learning tables if they do not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Per-user per-topic ability (theta)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_abilities (
            user_id TEXT,
            topic TEXT,
            ability REAL,
            updated_at TEXT,
            PRIMARY KEY (user_id, topic)
        )
        """
    )
    # Optional item-level running statistics
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS item_stats (
            item_id TEXT PRIMARY KEY,
            attempts INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0,
            last_updated TEXT
        )
        """
    )
    # Ability history over time
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ability_history (
            user_id TEXT,
            topic TEXT,
            ability REAL,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def _get_user_abilities(user_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT topic, ability FROM user_abilities WHERE user_id=?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return {t: float(a) for (t, a) in rows}

def _update_user_ability(user_id: str, topic: str, ability: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_abilities (user_id, topic, ability, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(user_id, topic) DO UPDATE SET ability=excluded.ability, updated_at=excluded.updated_at",
        (user_id, topic, float(ability), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def _bump_item_stats(item_id: str, correct: bool) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO item_stats (item_id, attempts, correct, last_updated) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(item_id) DO UPDATE SET attempts=item_stats.attempts+1, correct=item_stats.correct+?, last_updated=?",
        (item_id, 1, 1 if correct else 0, datetime.now().isoformat(), 1 if correct else 0, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def _elo_expected(ability: float, difficulty_b: float) -> float:
    # Logistic with slope 1.7 approximates 2-PL IRT
    return 1.0 / (1.0 + np.exp(-1.7 * (ability - difficulty_b)))

def _update_ability_and_difficulty(user_id: str, topic: str, item_id: str, correct: bool) -> None:
    # Read current ability and item difficulty
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ability FROM user_abilities WHERE user_id=? AND topic=?", (user_id, topic))
    row = cur.fetchone()
    theta = float(row[0]) if row else 0.5
    cur.execute("SELECT difficulty_b FROM quiz_items WHERE item_id=?", (item_id,))
    row2 = cur.fetchone()
    b = float(row2[0]) if row2 and row2[0] is not None else 0.5
    conn.close()

    expected = _elo_expected(theta, b)
    K_theta = 0.15
    K_b = 0.07
    # Update rules
    theta_new = float(np.clip(theta + K_theta * ((1.0 if correct else 0.0) - expected), -3.0, 3.0))
    b_new = float(np.clip(b + K_b * (expected - (1.0 if correct else 0.0)), -3.0, 3.0))

    # Persist updates
    _update_user_ability(user_id, topic, theta_new)
    # Append to ability history
    try:
        connh = sqlite3.connect(DB_PATH)
        curh = connh.cursor()
        curh.execute(
            "INSERT INTO user_ability_history (user_id, topic, ability, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, topic, float(theta_new), datetime.now().isoformat()),
        )
        connh.commit()
        connh.close()
    except Exception:
        pass
    conn2 = sqlite3.connect(DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute("UPDATE quiz_items SET difficulty_b=? WHERE item_id=?", (b_new, item_id))
    conn2.commit()
    conn2.close()
    _bump_item_stats(item_id, correct)
def _row_to_item(row: tuple) -> dict:
    return {
        "item_id": row[0],
        "topic": row[1],
        "stem": row[2],
        "options": json.loads(row[3]) if row[3] else [],
        "correct_index": row[4],
        "explanation": row[5] or "",
        "passage_excerpt": row[6] or "",
        "source_file": row[7] or "",
        "difficulty_b": float(row[8]) if row[8] is not None else 0.5,
        "discrimination_a": float(row[9]) if row[9] is not None else 1.0,
        "guessing_c": float(row[10]) if row[10] is not None else 0.0,
        "type": row[11] or "mcq",
        "validated": int(row[12]) if len(row) > 12 and row[12] is not None else 0,
    }

def _parse_llm_mcq_json(payload: str) -> dict | None:
    try:
        data = json.loads(payload)
        # Accept either wrapped or flat structure
        q = data.get("question") if isinstance(data, dict) else None
        if q and isinstance(q, dict) and all(k in q for k in ("stem", "options", "correct_index", "explanation")):
            return q
        if isinstance(data, dict) and all(k in data for k in ("stem", "options", "correct_index", "explanation")):
            return data
    except Exception:
        return None
    return None

def _llm_generate_mcq_from_passage(topic: str, passage_text: str, target_difficulty: float) -> dict | None:
    """Use LLM to generate a high-quality, grounded MCQ from a passage."""
    prompt = (
        "You are an expert tutor grounded in Ray Peat's corpus. Create ONE multiple-choice question grounded STRICTLY in the given passage.\n"
        "- Calibrate difficulty around the target difficulty (0=easy, 1=hard): {difficulty:.2f}.\n"
        "- The stem must reference or paraphrase the passage precisely.\n"
        "- Include 1-2 short direct quotes from the passage in the explanation to justify the correct answer.\n"
        "- Options must be plausible and mutually exclusive; exactly one correct.\n"
        "Return a JSON object with keys: stem, options (array of 4), correct_index (0-3), explanation.\n"
        "Passage:\n" + passage_text[:1600]
    ).format(difficulty=target_difficulty)
    raw = llm_call(prompt)
    parsed = _parse_llm_mcq_json(raw)
    if not parsed:
        # Retry once with stricter instruction and shorter passage
        prompt_retry = (
            "Return ONLY JSON with keys: stem, options (4), correct_index (0-3), explanation. No prose.\n"
            + prompt
        )
        raw2 = llm_call(prompt_retry)
        parsed = _parse_llm_mcq_json(raw2)
    return parsed

async def _seed_items_for_topics(
    topics: list[str],
    per_topic: int = 8,
    target_anchor: float = 0.5,
    max_time_seconds: float = 6.0,
    llm_max_items: int = 3,
) -> int:
    """Seed the item bank with items from vector search with a tight time/LLM budget.

    - Tries LLM-grounded MCQs for a few items (llm_max_items)
    - Falls back to template generation if over budget or LLM fails
    - Ensures seeding returns quickly to avoid HTTP timeouts
    """
    if not topics:
        return 0
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    inserted = 0
    start_time = time.time()
    llm_used = 0
    for topic in topics:
        try:
            results = await rag_system.search_engine.search(query=topic, top_k=per_topic, min_similarity=0.25)
        except Exception:
            results = []
        count = 0
        for r in results:
            if count >= per_topic:
                break
            item_id = f"it_{uuid.uuid4().hex[:12]}"
            # Try to build a better grounded MCQ via LLM
            mcq = None
            time_exceeded = (time.time() - start_time) > max_time_seconds
            if (not time_exceeded) and (llm_used < llm_max_items):
                try:
                    mcq = _llm_generate_mcq_from_passage(topic, r.context or "", target_anchor)
                    if mcq:
                        llm_used += 1
                except Exception:
                    mcq = None
            if mcq is not None:
                stem = mcq.get("stem") or "Based on the passage, which is correct?"
                options = mcq.get("options") or []
                if not isinstance(options, list) or len(options) < 4:
                    # Fallback to four options by padding
                    options = (options or [])[:4]
                    while len(options) < 4:
                        options.append("(distractor)")
                correct_index = int(mcq.get("correct_index") or 0)
                explanation = mcq.get("explanation") or ""
            else:
                # Fallback to faster template grounded on this passage
                try:
                    templ = generate_mcq_from_passage(topic=topic, passage_text=(r.context or ""))
                    stem = templ.get("question_text") or "Based on the passage, which is correct?"
                    options = templ.get("options") or ["A", "B", "C", "D"]
                    correct_index = int(templ.get("correct_answer") or 0)
                    explanation = templ.get("rationale") or ""
                except Exception:
                    key_phrase = topic.split()[0].capitalize() if topic else "Topic"
                    stem = f"Which statement is supported by the passage about {key_phrase}?"
                    options = [
                        f"{key_phrase} supports metabolic health in the bioenergetic view.",
                        f"{key_phrase} should generally be avoided according to the excerpt.",
                        f"{key_phrase} has no meaningful metabolic impact per the passage.",
                        f"{key_phrase} only matters in rare cases as implied in the excerpt.",
                    ]
                    correct_index = 0
                    explanation = "This option most closely aligns with the cited passage."
            try:
                cur.execute(
                    "INSERT INTO quiz_items (item_id, topic, stem, options, correct_index, explanation, passage_excerpt, source_file, difficulty_b, discrimination_a, guessing_c, type, validated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item_id,
                        topic,
                        stem,
                        json.dumps(options),
                        correct_index,
                        explanation,
                        (r.context or "")[:800],
                        r.source_file or "",
                        float(target_anchor),
                        1.0,
                        0.0,
                        "mcq",
                        0,
                    ),
                )
                inserted += 1
                count += 1
            except Exception:
                continue
    conn.commit()
    conn.close()
    return inserted

def _select_items(user_id: str, topics: list[str] | None, num_questions: int, target_difficulty: Optional[float] = None) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        if topics:
            placeholders = ",".join(["?"] * len(topics))
            cur.execute(
                f"SELECT item_id, topic, stem, options, correct_index, explanation, passage_excerpt, source_file, difficulty_b, discrimination_a, guessing_c, type, validated FROM quiz_items WHERE topic IN ({placeholders})",
                topics,
            )
        else:
            cur.execute(
                "SELECT item_id, topic, stem, options, correct_index, explanation, passage_excerpt, source_file, difficulty_b, discrimination_a, guessing_c, type, validated FROM quiz_items"
            )
        rows = cur.fetchall()
    finally:
        conn.close()
    items = [_row_to_item(r) for r in rows]
    if not items:
        return []
    # Determine per-topic target from user's current ability when available
    abilities = _get_user_abilities(user_id)
    # Rank by |b - theta_topic| or fallback to |b - target_difficulty| or |b - 0.5|
    for it in items:
        b = it.get("difficulty_b") or 0.5
        theta_topic = abilities.get((it.get("topic") or "").lower())
        if theta_topic is None:
            # Try raw topic without lower
            theta_topic = abilities.get(it.get("topic") or "")
        anchor = theta_topic if theta_topic is not None else (target_difficulty if target_difficulty is not None else 0.5)
        it["rank"] = abs(b - anchor)
    items.sort(key=lambda x: x["rank"]) 
    selected: list[dict] = []
    seen_sources: set[str] = set()
    for it in items:
        src = (it.get("source_file") or "").strip()
        if src in seen_sources:
            continue
        selected.append(it)
        seen_sources.add(src)
        if len(selected) >= num_questions:
            break
    # If not enough unique sources, fill remaining ignoring source diversity
    if len(selected) < num_questions:
        leftover = [it for it in items if it not in selected]
        selected.extend(leftover[: (num_questions - len(selected))])
    return selected[:num_questions]

def load_mf_model() -> None:
    global MF_AVAILABLE, MF_U, MF_V, MF_USER_TO_IDX, MF_ITEM_TO_IDX
    if MF_MODEL_PATH.exists():
        try:
            data = np.load(MF_MODEL_PATH, allow_pickle=True)
            MF_U = data['U']
            MF_V = data['V']
            MF_USER_TO_IDX = dict(data['user_to_idx'])  # type: ignore
            MF_ITEM_TO_IDX = dict(data['item_to_idx'])  # type: ignore
            MF_AVAILABLE = True
            print(f"✅ MF model loaded: users={len(MF_USER_TO_IDX)}, items={len(MF_ITEM_TO_IDX)}")
        except Exception as e:
            print(f"⚠️ MF model load failed: {e}")
            MF_AVAILABLE = False
    else:
        MF_AVAILABLE = False

@app.on_event("startup")
async def startup_event():
    """Initialize all advanced ML components on startup."""
    if ADVANCED_ML_AVAILABLE:
        print("🚀 Initializing Advanced ML Components...")
        
        # Initialize personalization engine
        await personalization_engine.initialize_models()
        
        # Initialize RL agent
        print("🤖 RL Agent ready")
        
        # Initialize knowledge graph
        await ray_peat_knowledge_graph.initialize_models()
        
        print("✅ All advanced ML components initialized!")
    else:
        print("⚠️ Running in basic mode without advanced ML features")
    # Load MF recommender regardless, if available
    load_mf_model()
    # Ensure quiz schema exists
    _ensure_quiz_schema()
    _ensure_adaptive_schema()

# Basic endpoints (existing)
@app.get("/")
async def root():
    return {
        "message": "PeatLearn Advanced AI Learning Platform",
        "version": "2.0.0",
        "features": {
            "basic_rag": True,
            "advanced_ml": ADVANCED_ML_AVAILABLE,
            "personalization": ADVANCED_ML_AVAILABLE,
            "reinforcement_learning": ADVANCED_ML_AVAILABLE,
            "knowledge_graph": ADVANCED_ML_AVAILABLE
        }
    }

@app.get("/api/ask")
async def ask_question(
    q: str = Query(..., description="Question to ask Ray Peat's knowledge"),
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    max_sources: int = Query(5, description="Maximum number of sources to use"),
    min_similarity: float = Query(0.3, description="Minimum similarity threshold")
):
    """Ask a question with optional personalization."""
    
    try:
        # Basic RAG response
        response = await rag_system.answer_question(q, max_sources, min_similarity)
        
        # Add personalization if available and user provided
        if ADVANCED_ML_AVAILABLE and user_id:
            
            # Expand query using knowledge graph
            expanded_query_data = await ray_peat_knowledge_graph.expand_query_with_concepts(q)
            
            if expanded_query_data['num_expansions'] > 0:
                # Re-run RAG with expanded query
                expanded_response = await rag_system.answer_question(
                    expanded_query_data['expanded_query'], 
                    max_sources, 
                    min_similarity
                )
                
                # Blend responses (simple strategy)
                if expanded_response.confidence > response.confidence:
                    response = expanded_response
                    response.query = f"{q} (expanded with: {', '.join(expanded_query_data['expansion_terms'])})"
            
            # Log interaction for learning
            interaction = UserInteraction(
                user_id=user_id,
                content_id=f"question_{hash(q)}",
                interaction_type="question",
                timestamp=datetime.now(),
                performance_score=response.confidence,
                time_spent=0.0,  # Will be updated by frontend
                difficulty_level=0.5,  # Default
                topic_tags=[],  # Could be extracted from question
                context={"question": q, "sources_used": len(response.sources)}
            )
            
            await personalization_engine.update_user_state(user_id, interaction)
        
        return {
            "question": q,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [
                {
                    "source_file": source.source_file,
                    "similarity_score": source.similarity_score,
                    "context": source.context[:200] + "..." if len(source.context) > 200 else source.context
                }
                for source in response.sources
            ],
            "personalized": user_id is not None and ADVANCED_ML_AVAILABLE
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Advanced ML endpoints
if ADVANCED_ML_AVAILABLE:
    @app.post("/api/users/profile")
    async def create_or_update_user_profile(profile: UserProfile):
        """Create or update user profile."""
        return {"message": f"Profile updated for user {profile.user_id}", "profile": profile}

    @app.post("/api/interactions")
    async def log_interaction(interaction: InteractionData):
        """Log a user interaction for learning."""

        user_interaction = UserInteraction(
            user_id=interaction.user_id,
            content_id=interaction.content_id,
            interaction_type=interaction.interaction_type,
            timestamp=datetime.now(),
            performance_score=interaction.performance_score,
            time_spent=interaction.time_spent,
            difficulty_level=interaction.difficulty_level,
            topic_tags=interaction.topic_tags,
            context=interaction.context,
        )

        await personalization_engine.update_user_state(interaction.user_id, user_interaction)

        learning_state = LearningEnvironmentState(
            user_id=interaction.user_id,
            current_topic_mastery={},
            recent_performance=[interaction.performance_score],
            time_in_session=interaction.time_spent,
            difficulty_progression=[interaction.difficulty_level],
            engagement_level=min(1.0, interaction.performance_score + 0.2),
            fatigue_level=max(0.0, interaction.time_spent / 3600 - 0.5),
            topics_covered_today=len(interaction.topic_tags),
            consecutive_correct=0,
            consecutive_incorrect=0,
            preferred_learning_style="unknown",
            context_features=np.zeros(20),
        )

        reward = adaptive_agent.calculate_reward(
            learning_state,
            0,
            learning_state,
            {
                "performance_score": interaction.performance_score,
                "difficulty_rating": interaction.difficulty_level,
                "actual_time": interaction.time_spent,
                "expected_time": 300,
            },
        )

        await adaptive_agent.update_bandit(0, reward)

        return {"message": "Interaction logged successfully", "reward": reward}

    @app.post("/api/recommendations")
    async def get_recommendations(request: RecommendationRequest):
        """Get personalized content recommendations using MF if available, blended with content relevance."""
        cache_key = f"adv:{request.user_id}:{','.join(request.topic_filter or [])}:{request.num_recommendations}:{request.exclude_seen}"
        now = time.time()
        cached = RECS_CACHE.get(cache_key)
        if cached and now - cached["ts"] < RECS_TTL_SECONDS:
            return cached["data"]

        topics = request.topic_filter or [
            "thyroid function",
            "estrogen progesterone",
            "metabolism energy",
            "serotonin light",
            "carbon dioxide",
            "PUFA oils",
            "aspirin inflammation",
        ]
        candidates: Dict[str, float] = {}
        for topic in topics:
            results = await rag_system.search_engine.search(query=topic, top_k=30, min_similarity=0.2)
            for r in results:
                cid = r.source_file or r.id
                candidates[cid] = max(float(r.similarity_score), candidates.get(cid, 0.0))

        recs: List[Dict[str, Any]] = []
        user_idx = MF_USER_TO_IDX.get(request.user_id) if MF_AVAILABLE else None
        for cid, base_sim in candidates.items():
            mf_score = 0.0
            if MF_AVAILABLE and user_idx is not None:
                item_idx = MF_ITEM_TO_IDX.get(cid)
                if item_idx is not None:
                    try:
                        mf_score = float(np.dot(MF_U[user_idx], MF_V[item_idx]))  # type: ignore
                    except Exception:
                        mf_score = 0.0
            score = 0.7 * mf_score + 0.3 * base_sim if MF_AVAILABLE and user_idx is not None else base_sim
            recs.append(
                {
                    "content_id": cid,
                    "predicted_score": score,
                    "recommendation_reason": "MF+content blend" if MF_AVAILABLE and user_idx is not None else "content relevance",
                }
            )

        recs.sort(key=lambda x: x["predicted_score"], reverse=True)
        recs = recs[: request.num_recommendations]

        resp = {"user_id": request.user_id, "recommendations": recs, "mode": "mf_blend" if MF_AVAILABLE and user_idx is not None else "content_based"}
        RECS_CACHE[cache_key] = {"ts": now, "data": resp}
        return resp

    from src.adaptive_learning.quiz_generator import quiz_generator

    @app.post("/api/quiz/generate")
    async def generate_personalized_quiz(request: QuizRequest):
        """Generate a personalized quiz using the new Gemini-powered quiz_generator."""
        user_profile = {
            "user_id": request.user_id,
            "overall_state": "learning",
            "topic_mastery": {},
            "learning_style": "explorer",
        }
        recent_interactions: List[Dict[str, Any]] = []

        quiz_data = quiz_generator.generate_quiz(
            user_profile=user_profile,
            topic=request.topic,
            num_questions=request.num_questions,
            recent_interactions=recent_interactions,
        )
        if not quiz_data or not quiz_data.get("questions"):
            raise HTTPException(status_code=500, detail="Failed to generate quiz from the LLM.")
        return quiz_data

    @app.post("/api/knowledge-graph/query")
    async def query_knowledge_graph(request: KnowledgeGraphQuery):
        """Query the knowledge graph for concept expansion and relationships."""

        expanded_query = await ray_peat_knowledge_graph.expand_query_with_concepts(request.query, request.max_expansions)

        result = {
            "original_query": request.query,
            "expanded_query": expanded_query["expanded_query"],
            "expansion_terms": expanded_query["expansion_terms"],
            "expansion_sources": expanded_query["expansion_sources"],
            "num_expansions": expanded_query["num_expansions"],
        }

        if request.include_related_concepts:
            pass

        return result

    @app.get("/api/analytics/user/{user_id}")
    async def get_user_analytics(user_id: str):
        """Get comprehensive analytics for a user."""

        # Initialize user if they don't exist
        if user_id not in personalization_engine.learning_states:
            from inference.backend.personalization.neural_personalization import LearningState
            import numpy as np
            from datetime import datetime
            
            personalization_engine.learning_states[user_id] = LearningState(
                user_id=user_id,
                topic_mastery={},
                learning_velocity=0.5,
                preferred_difficulty=0.5,
                learning_style_vector=np.random.normal(0, 0.1, 128),
                attention_span=30.0,
                last_active=datetime.now()
            )

        user_analytics = personalization_engine.get_user_analytics(user_id)

        return {"user_analytics": user_analytics, "timestamp": datetime.now().isoformat()}

    @app.get("/api/analytics/system")
    async def get_system_analytics():
        """Get system-wide analytics."""

        agent_analytics = adaptive_agent.get_agent_analytics()
        graph_analytics = ray_peat_knowledge_graph.get_graph_analytics()

        return {
            "reinforcement_learning": agent_analytics,
            "knowledge_graph": graph_analytics,
            "total_users": len(personalization_engine.learning_states),
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
        }

    @app.post("/api/rl/train")
    async def train_rl_agent():
        """Manually trigger RL agent training."""

        training_results = await adaptive_agent.train_dqn()

        return {
            "message": "RL agent training completed",
            "training_loss": training_results["loss"],
            "training_steps": adaptive_agent.steps_done,
            "epsilon": adaptive_agent.epsilon,
        }

else:
    # Basic, usable endpoints when advanced ML is not available
    from peatlearn.personalization.utils import (
        estimate_difficulty_score,
        generate_mcq_from_passage,
    )

    @app.post("/api/recommendations")
    async def get_recommendations_basic(request: RecommendationRequest):
        """Return content-based recommendations using RAG search only.

        Strategy: if topic_filter provided, search each topic and aggregate top results.
        Otherwise, return a diverse set by sampling common topics.
        """
        # Check cache first
        cache_key = f"basic:{request.user_id}:{','.join(request.topic_filter or [])}:{request.num_recommendations}:{request.exclude_seen}"
        now = time.time()
        cached = RECS_CACHE.get(cache_key)
        if cached and now - cached['ts'] < RECS_TTL_SECONDS:
            return cached['data']
        try:
            topics = request.topic_filter or [
                "thyroid function", "estrogen progesterone", "metabolism energy",
                "sugar and cellular energy", "aspirin inflammation",
            ]

            seen_sources = set()
            recs: List[Dict[str, Any]] = []

            for topic in topics:
                results = await rag_system.search_engine.search(query=topic, top_k=10, min_similarity=0.2)
                for r in results:
                    key = (r.source_file, r.id)
                    if key in seen_sources:
                        continue
                    seen_sources.add(key)
                    recs.append({
                        "content_id": r.source_file or r.id,
                        "predicted_score": float(r.similarity_score),
                        "title": r.source_file,
                        "snippet": r.context[:200] + ("..." if len(r.context) > 200 else ""),
                        "recommendation_reason": f"Related to topic: {topic}",
                    })

            # Sort by score and truncate
            recs.sort(key=lambda x: x["predicted_score"], reverse=True)
            recs = recs[: request.num_recommendations]

            resp = {
                "user_id": request.user_id,
                "recommendations": recs,
                "mode": "basic_content_based",
            }
            RECS_CACHE[cache_key] = {"ts": now, "data": resp}
            return resp
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Basic recommendations failed: {e}")

    @app.post("/api/quiz/generate")
    async def generate_quiz_basic(request: QuizRequest):
        """Generate a retrieval-based quiz without advanced ML.

        Uses RAG search to fetch passages and templates MCQs from them.
        """
        try:
            topic = request.topic or "thyroid metabolism"
            results = await rag_system.search_engine.search(query=topic, top_k=max(3, request.num_questions), min_similarity=0.3)
            if not results:
                # Fallback to a generic topic
                results = await rag_system.search_engine.search(query="ray peat metabolism", top_k=max(3, request.num_questions), min_similarity=0.1)

            questions: List[Dict[str, Any]] = []
            for i, r in enumerate(results[: request.num_questions]):
                q = generate_mcq_from_passage(topic=topic, passage_text=r.context)
                q.update({
                    "question_id": f"q_{i}",
                    "ray_peat_context": r.context[:220] + ("..." if len(r.context) > 220 else ""),
                    "source_file": r.source_file,
                })
                questions.append(q)

            return {
                "user_id": request.user_id,
                "quiz_id": f"quiz_{request.user_id}_{datetime.now().isoformat()}",
                "questions": questions,
                "quiz_metadata": {
                    "topic": topic,
                    "generation_method": "retrieval_templates",
                },
                "rag_integration": "enabled",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Basic quiz generation failed: {e}")

    @app.post("/api/knowledge-graph/query")
    async def query_knowledge_graph_basic(request: KnowledgeGraphQuery):
        """Provide co-occurrence based query expansion when advanced KG is unavailable."""
        try:
            from utils.concept_graph import expand_query_terms
            result = expand_query_terms(request.query, max_expansions=request.max_expansions)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Basic KG expansion failed: {e}")

# Quiz submission (available in both modes)
class QuizSubmission(BaseModel):
    user_id: str
    quiz_id: str
    answers: Dict[str, int]
    questions: List[Dict[str, Any]]
    topic: Optional[str] = None


@app.post("/api/quiz/submit")
async def submit_quiz(payload: QuizSubmission):
    """Accept quiz answers, log outcomes, and return score summary."""
    try:
        user_id = payload.user_id
        quiz_id = payload.quiz_id
        answers = payload.answers or {}
        questions = payload.questions or []
        topic = payload.topic or ""
        total = len(questions)
        correct_ct = 0
        for q in questions:
            qid = q.get("question_id")
            chosen = answers.get(qid)
            correct_idx = q.get("correct_answer")
            correct = chosen == correct_idx
            if correct:
                correct_ct += 1
            log_quiz_outcome(
                user_id,
                quiz_id,
                qid,
                topic,
                correct,
                time_taken=0.0,
                context={"question_text": q.get("question_text", "")[:160], "source_file": q.get("source_file", "")},
            )
        score_pct = (correct_ct / total * 100.0) if total else 0.0
        if ADVANCED_ML_AVAILABLE:
            topics_from_questions = list({q.get("topic") for q in questions if q.get("topic")})
            topic_tags = topics_from_questions if topics_from_questions else ([topic] if topic else [])
            interaction = UserInteraction(
                user_id=user_id,
                content_id=f"quiz_{quiz_id}",
                interaction_type="quiz",
                timestamp=datetime.now(),
                performance_score=score_pct / 100.0,
                time_spent=0.0,
                difficulty_level=0.5,
                topic_tags=topic_tags,
                context={"quiz_id": quiz_id, "total": total, "correct": correct_ct},
            )
            await personalization_engine.update_user_state(user_id, interaction)
        return {"score_percentage": score_pct, "correct": correct_ct, "total": total}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Quiz submission error: {e}")


# Session-based quiz endpoints (available in both modes)
@app.post("/api/quiz/session/start")
async def quiz_session_start(payload: QuizSessionStart):
    """Start a session from the item bank, seeding items if needed."""
    topics = payload.topics or []
    target_diff = payload.target_difficulty if payload.target_difficulty is not None else None
    # Use user's ability for target anchor if available
    abilities = _get_user_abilities(payload.user_id)
    topic_key = (topics[0] if topics else "").lower()
    anchor = abilities.get(topic_key) if topic_key else None
    if anchor is None:
        anchor = target_diff if target_diff is not None else 0.5
    await _seed_items_for_topics(topics or ["serotonin", "thyroid", "progesterone"], per_topic=20, target_anchor=float(anchor))
    items = _select_items(payload.user_id, topics if topics else None, payload.num_questions, target_diff if target_diff is not None else anchor)
    if not items:
        raise HTTPException(status_code=400, detail="No quiz items available. Please seed the item bank.")
    session_id = f"qs_{uuid.uuid4().hex[:12]}"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO quiz_sessions (session_id, user_id, started_at, topics, num_questions, target_difficulty, policy, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            payload.user_id,
            datetime.now().isoformat(),
            json.dumps(topics),
            payload.num_questions,
            (float(target_diff) if target_diff is not None else None),
            "diverse_source_rank",
            "active",
        ),
    )
    for i, it in enumerate(items):
        cur.execute(
            "INSERT INTO quiz_queue (session_id, ord, item_id, shown) VALUES (?, ?, ?, ?)",
            (session_id, i, it["item_id"], 0),
        )
    conn.commit()
    conn.close()
    return {"session_id": session_id, "total": len(items)}


@app.get("/api/quiz/next")
async def quiz_next(session_id: str, user_id: Optional[str] = None):
    """Return the next queued item for this session."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT ord, item_id FROM quiz_queue WHERE session_id=? AND shown=0 ORDER BY ord ASC LIMIT 1",
        (session_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"done": True}
    ord_idx, item_id = row
    cur.execute(
        "SELECT item_id, topic, stem, options, correct_index, explanation, passage_excerpt, source_file, difficulty_b, discrimination_a, guessing_c, type, validated FROM quiz_items WHERE item_id=?",
        (item_id,),
    )
    item_row = cur.fetchone()
    shown_at = datetime.now().isoformat()
    cur.execute(
        "UPDATE quiz_queue SET shown=1, shown_at=? WHERE session_id=? AND ord=?",
        (shown_at, session_id, ord_idx),
    )
    conn.commit()
    conn.close()
    if not item_row:
        raise HTTPException(status_code=404, detail="Item not found")
    item = _row_to_item(item_row)
    # Compute debug anchors if user_id provided
    ability_topic = None
    target_anchor = None
    if user_id:
        try:
            abilities = _get_user_abilities(user_id)
            ability_topic = abilities.get((item.get("topic") or "").lower()) or abilities.get(item.get("topic") or "")
            target_anchor = ability_topic if ability_topic is not None else None
        except Exception:
            ability_topic = None
            target_anchor = None
    return {
        "session_id": session_id,
        "order": ord_idx,
        "item_id": item["item_id"],
        "topic": item["topic"],
        "type": item["type"],
        "stem": item["stem"],
        "options": item["options"],
        "passage_excerpt": item["passage_excerpt"],
        "source_file": item["source_file"],
        "difficulty_b": item.get("difficulty_b", 0.5),
        "target_anchor": target_anchor,
        "ability_topic": ability_topic,
        "time_limit": 90,
    }


@app.post("/api/quiz/answer")
async def quiz_answer(payload: QuizAnswer):
    """Record the user's answer and return correctness and explanation."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT ord, shown_at FROM quiz_queue WHERE session_id=? AND item_id=?",
        (payload.session_id, payload.item_id),
    )
    qrow = cur.fetchone()
    if not qrow:
        conn.close()
        raise HTTPException(status_code=404, detail="Queue entry not found")
    ord_idx, shown_at = qrow
    cur.execute(
        "SELECT correct_index, topic, stem, options FROM quiz_items WHERE item_id=?",
        (payload.item_id,),
    )
    irow = cur.fetchone()
    if not irow:
        conn.close()
        raise HTTPException(status_code=404, detail="Item not found")
    correct_index, topic, stem, options_json = irow[0], irow[1], irow[2], irow[3]
    correct = int(payload.chosen_index == correct_index)
    answered_at = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO quiz_events (session_id, item_id, ord, shown_at, answered_at, chosen_index, correct, response_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            payload.session_id,
            payload.item_id,
            ord_idx,
            shown_at,
            answered_at,
            int(payload.chosen_index),
            int(correct),
            int(payload.time_ms or 0),
        ),
    )
    conn.commit()
    conn.close()
    # Update adaptive parameters and optionally personalization
    try:
        caller_user_id = payload.user_id or "anonymous"
        _update_ability_and_difficulty(caller_user_id, (topic or "").lower(), payload.item_id, bool(correct))
        if ADVANCED_ML_AVAILABLE:
            interaction = UserInteraction(
                user_id=caller_user_id,
                content_id=f"item_{payload.item_id}",
                interaction_type="quiz",
                timestamp=datetime.now(),
                performance_score=1.0 if correct else 0.0,
                time_spent=(payload.time_ms or 0) / 1000.0,
                difficulty_level=0.5,
                topic_tags=[topic] if topic else [],
                context={"session_id": payload.session_id, "ord": ord_idx},
            )
            await personalization_engine.update_user_state(interaction.user_id, interaction)
    except Exception:
        pass
    return {"correct": bool(correct), "correct_index": int(correct_index)}


@app.post("/api/quiz/finish")
async def quiz_finish(session_id: str):
    """Finalize the session and return a summary."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM quiz_events WHERE session_id=? AND correct=1", (session_id,))
    correct_ct = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM quiz_queue WHERE session_id=?", (session_id,))
    total = cur.fetchone()[0]
    cur.execute("UPDATE quiz_sessions SET finished_at=?, status='finished' WHERE session_id=?", (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()
    pct = (correct_ct / total * 100.0) if total else 0.0
    return {"correct": correct_ct, "total": total, "score_percentage": pct}

# Health check
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "advanced_ml": ADVANCED_ML_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
