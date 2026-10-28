"""Redis query/response caching service for RAG pipeline.

Supports:
1. Tier-1 Exact Match: normalized query SHA-256 lookup in < 1ms (0 API calls).
2. Tier-2 Semantic Match: Cosine similarity comparison on query embeddings (via numpy).
3. Fail-open error handling: if Redis is unreachable, safely bypasses cache.
4. Invalidation: clear collection cache when new documents are ingested.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

import numpy as np
import redis

from src.config import settings
from src.models.schemas import ChatResponse
from src.utils.logger import logger

_redis_client: redis.Redis | None = None
_client_lock = threading.Lock()


def get_redis_client() -> redis.Redis | None:
    """Return the shared Redis client, creating it if not already connected (thread-safe singleton).

    Fails open (returns None) if Redis server is unreachable.
    """
    global _redis_client
    if not settings.ENABLE_QUERY_CACHE:
        return None

    if _redis_client is None:
        with _client_lock:
            if _redis_client is None:
                try:
                    client = redis.Redis.from_url(
                        settings.REDIS_URL,
                        decode_responses=True,
                        socket_connect_timeout=1.5,
                        socket_timeout=1.5,
                    )
                    client.ping()
                    _redis_client = client
                    logger.bind(url=settings.REDIS_URL).info("Connected to Redis cache server")
                except Exception as exc:
                    logger.bind(url=settings.REDIS_URL, error=str(exc)).warning(
                        "Redis server unreachable; query caching disabled (fail-open)"
                    )
                    return None
    return _redis_client


def reset_redis_client() -> None:
    """Reset the singleton Redis client instance (useful for unit tests)."""
    global _redis_client
    with _client_lock:
        if _redis_client is not None:
            try:
                _redis_client.close()
            except Exception:
                pass
            _redis_client = None


def normalize_query(query: str) -> str:
    """Normalize user query by trimming, converting to lowercase, and collapsing whitespace."""
    return " ".join((query or "").strip().lower().split())


def build_exact_cache_key(query: str, collection_name: str, top_k: int) -> str:
    """Generate Redis key for Tier-1 Exact Match cache."""
    normalized = normalize_query(query)
    q_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"rag:cache:exact:{collection_name}:{top_k}:{q_hash}"


def build_semantic_index_key(collection_name: str, top_k: int) -> str:
    """Generate Redis key for Tier-2 Semantic Match hash index."""
    return f"rag:cache:semantic:{collection_name}:{top_k}"


def get_cached_response(
    query: str,
    collection_name: str | None = None,
    top_k: int | None = None,
    query_embedding: list[float] | None = None,
) -> ChatResponse | None:
    """Retrieve cached ChatResponse for a query via Exact Match or Semantic Match.

    Args:
        query: User search query.
        collection_name: Target collection name.
        top_k: Number of retrieved chunks configured for the request.
        query_embedding: Optional precomputed query embedding vector for semantic search.

    Returns:
        ChatResponse | None: Cached response if hit, otherwise None.
    """
    if not settings.ENABLE_QUERY_CACHE:
        return None

    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return None

    client = get_redis_client()
    if client is None:
        return None

    target_col = collection_name or settings.CHROMA_COLLECTION_NAME
    k = top_k or settings.DEFAULT_TOP_K

    # 1. Tier-1: Exact Match Lookup (O(1), < 1ms)
    exact_key = build_exact_cache_key(cleaned_query, target_col, k)
    try:
        cached_json = client.get(exact_key)
        if cached_json:
            data = json.loads(cached_json)
            logger.bind(
                query=cleaned_query,
                collection=target_col,
                top_k=k,
                match="exact",
            ).info("Redis Query Cache HIT (Exact Match)")
            return ChatResponse.model_validate(data)
    except Exception as exc:
        logger.bind(query=cleaned_query, error=str(exc)).warning(
            "Failed reading from Redis exact cache; continuing with pipeline"
        )
        return None

    # 2. Tier-2: Semantic Match Lookup (Cosine Similarity on query embeddings)
    if settings.ENABLE_SEMANTIC_CACHE:
        sem_key = build_semantic_index_key(target_col, k)
        try:
            raw_entries = client.hgetall(sem_key)
            if raw_entries:
                from src.core.retriever import embed_query

                q_vec = np.array(
                    query_embedding if query_embedding is not None else embed_query(cleaned_query),
                    dtype=np.float32,
                )
                q_norm = float(np.linalg.norm(q_vec))

                if q_norm > 0:
                    best_sim = -1.0
                    best_response_data: dict[str, Any] | None = None

                    for entry_str in raw_entries.values():
                        try:
                            entry = json.loads(entry_str)
                            c_vec = np.array(entry["embedding"], dtype=np.float32)
                            c_norm = float(np.linalg.norm(c_vec))
                            if c_norm > 0:
                                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_response_data = entry["response"]
                        except Exception:
                            continue

                    if best_sim >= settings.SEMANTIC_CACHE_THRESHOLD and best_response_data:
                        logger.bind(
                            query=cleaned_query,
                            collection=target_col,
                            similarity=round(best_sim, 4),
                            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
                            match="semantic",
                        ).info("Redis Query Cache HIT (Semantic Match)")
                        return ChatResponse.model_validate(best_response_data)
        except Exception as exc:
            logger.bind(query=cleaned_query, error=str(exc)).warning(
                "Failed querying Redis semantic cache; continuing with pipeline"
            )

    return None


def set_cached_response(
    query: str,
    response: ChatResponse,
    collection_name: str | None = None,
    top_k: int | None = None,
    query_embedding: list[float] | None = None,
) -> bool:
    """Store ChatResponse in Redis for exact and optional semantic matching.

    Args:
        query: User search query text.
        response: Generated ChatResponse to store.
        collection_name: Target collection name.
        top_k: Top-K chunks setting.
        query_embedding: Optional precomputed query vector for semantic caching.

    Returns:
        bool: True if stored successfully, False otherwise.
    """
    if not settings.ENABLE_QUERY_CACHE:
        return False

    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return False

    client = get_redis_client()
    if client is None:
        return False

    target_col = collection_name or settings.CHROMA_COLLECTION_NAME
    k = top_k or settings.DEFAULT_TOP_K
    ttl = settings.QUERY_CACHE_TTL_SECONDS
    response_dict = response.model_dump()
    response_json = json.dumps(response_dict, ensure_ascii=False)

    try:
        # 1. Store Tier-1 Exact Match key
        exact_key = build_exact_cache_key(cleaned_query, target_col, k)
        client.set(exact_key, response_json, ex=ttl)

        # 2. Store Tier-2 Semantic Index (if enabled)
        if settings.ENABLE_SEMANTIC_CACHE:
            from src.core.retriever import embed_query

            q_vec = (
                query_embedding
                if query_embedding is not None
                else embed_query(cleaned_query)
            )
            sem_key = build_semantic_index_key(target_col, k)
            norm_q = normalize_query(cleaned_query)
            q_hash = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()

            entry_payload = json.dumps(
                {
                    "query": cleaned_query,
                    "embedding": q_vec,
                    "response": response_dict,
                    "timestamp": time.time(),
                },
                ensure_ascii=False,
            )
            client.hset(sem_key, q_hash, entry_payload)
            client.expire(sem_key, ttl)

        logger.bind(
            query=cleaned_query,
            collection=target_col,
            top_k=k,
            ttl=ttl,
        ).debug("Successfully saved RAG response to Redis cache")
        return True
    except Exception as exc:
        logger.bind(query=cleaned_query, error=str(exc)).warning(
            "Failed saving response to Redis query cache"
        )
        return False


def invalidate_collection_cache(collection_name: str | None = None) -> int:
    """Delete all cached queries and semantic indexes for a given collection.

    Args:
        collection_name: Target collection name to clear.

    Returns:
        int: Number of invalidated Redis keys.
    """
    client = get_redis_client()
    if client is None:
        return 0

    target_col = collection_name or settings.CHROMA_COLLECTION_NAME
    deleted_count = 0

    try:
        # Scan and delete exact keys and semantic hash keys matching collection
        patterns = [
            f"rag:cache:exact:{target_col}:*",
            f"rag:cache:semantic:{target_col}:*",
        ]
        matched_keys: list[str] = []
        for pattern in patterns:
            for key in client.scan_iter(match=pattern, count=100):
                matched_keys.append(key)

        if matched_keys:
            deleted_count = client.delete(*matched_keys)

        logger.bind(collection=target_col, deleted_keys=deleted_count).info(
            "Invalidated Redis collection query cache"
        )
    except Exception as exc:
        logger.bind(collection=target_col, error=str(exc)).warning(
            "Failed invalidating Redis collection cache"
        )

    return deleted_count

