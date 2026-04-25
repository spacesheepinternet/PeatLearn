"""
Gemini embedding singleton for RAG queries.

Uses Google's gemini-embedding-001 API to generate 3072-dim embeddings.
Falls back to SHA-256 hash for offline/test use (same pattern as before).

Thread-safe: multiple concurrent requests share the same aiohttp session
or use the sync wrapper which creates a one-shot session.
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Optional

import aiohttp
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = settings.EMBEDDING_DIMENSIONS  # 3072
_MODEL = settings.EMBEDDING_MODEL  # gemini-embedding-001
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Rate limiting
_last_request_time = 0.0
_MIN_DELAY = 0.05  # 50ms between requests (20 req/s safe margin)


def _get_api_key() -> Optional[str]:
    """Get Gemini API key from settings or environment."""
    return settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")


async def get_embedding_async(text: str, session: Optional[aiohttp.ClientSession] = None) -> list[float]:
    """Generate a 3072-dim embedding via Gemini API (async).

    Args:
        text: The text to embed.
        session: Optional aiohttp session for connection reuse.

    Returns:
        List of floats (3072-dim embedding vector).

    Raises:
        RuntimeError: In production if API is unavailable.
    """
    global _last_request_time

    api_key = _get_api_key()
    if not api_key:
        return _hash_fallback(text)

    url = f"{_BASE_URL}/models/{_MODEL}:embedContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "model": f"models/{_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
    }

    # Simple rate limiting
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_DELAY:
        await asyncio.sleep(_MIN_DELAY - elapsed)
    _last_request_time = time.time()

    # Retry with exponential backoff
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        retry_delay = 1.0
        for attempt in range(5):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        values = data.get("embedding", {}).get("values", [])
                        if values:
                            return values
                        logger.warning("Gemini returned empty embedding")
                        return _hash_fallback(text)
                    elif resp.status == 429:
                        if attempt < 4:
                            logger.warning(f"Gemini rate limit (429), retry in {retry_delay}s (attempt {attempt+1}/5)")
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 30)
                            continue
                        error = await resp.text()
                        logger.error(f"Gemini rate limit after 5 retries: {error}")
                        return _hash_fallback(text)
                    else:
                        error = await resp.text()
                        logger.error(f"Gemini API error {resp.status}: {error}")
                        return _hash_fallback(text)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < 4:
                    logger.warning(f"Gemini request failed (attempt {attempt+1}/5): {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue
                logger.error(f"Gemini request failed after 5 retries: {e}")
                return _hash_fallback(text)
    finally:
        if own_session:
            await session.close()

    return _hash_fallback(text)


def get_embedding(text: str) -> list[float]:
    """Sync wrapper — generates a 3072-dim embedding via Gemini API.

    Safe to call from sync code. Creates an event loop if needed.
    For batch operations, prefer get_embedding_async with a shared session.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context — can't use asyncio.run
        # Use a new thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, get_embedding_async(text))
            return future.result(timeout=30)
    else:
        return asyncio.run(get_embedding_async(text))


def _hash_fallback(text: str) -> list[float]:
    """SHA-256 hash fallback for offline/test use (not semantically meaningful)."""
    if os.getenv("PEATLEARN_ENV") == "production":
        raise RuntimeError(
            "Gemini embedding API unavailable; refusing to serve hash-fallback "
            "in production. This would return meaningless search results."
        )
    logger.warning("Using SHA-256 hash fallback — results will NOT be semantically meaningful.")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(
        (digest * ((_EMBEDDING_DIM // len(digest)) + 1))[:_EMBEDDING_DIM],
        dtype=np.uint8,
    ).astype(np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def get_embedding_dim() -> int:
    """Return the embedding dimension (3072)."""
    return _EMBEDDING_DIM
