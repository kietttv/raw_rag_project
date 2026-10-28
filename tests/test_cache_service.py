"""Unit tests for src.services.cache_service module."""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.config import settings
from src.models.schemas import ChatResponse, SourceDocument
from src.services import cache_service


def test_normalize_query() -> None:
    """normalize_query trims, lowercases, and collapses irregular spaces."""
    assert cache_service.normalize_query("   Hỏi   về   RAG?  \n ") == "hỏi về rag?"
    assert cache_service.normalize_query("") == ""
    assert cache_service.normalize_query("FASTAPI") == "fastapi"


def test_build_exact_cache_key() -> None:
    """build_exact_cache_key generates consistent SHA-256 keys."""
    k1 = cache_service.build_exact_cache_key("RAG là gì?", "docs", 5)
    k2 = cache_service.build_exact_cache_key("  rag   LÀ GÌ?  ", "docs", 5)
    assert k1 == k2
    assert "rag:cache:exact:docs:5:" in k1


def test_exact_match_cache_hit_and_miss() -> None:
    """get_cached_response returns ChatResponse on exact hit and None on miss."""
    mock_redis = MagicMock()
    mock_response = ChatResponse(
        answer="RAG là kỹ thuật kết hợp tìm kiếm và sinh văn bản.",
        sources=[SourceDocument(content="RAG chunk", score=0.9, metadata={"chunk_id": "c1"})],
    )
    cached_json = mock_response.model_dump_json()

    with patch("src.services.cache_service.get_redis_client", return_value=mock_redis):
        # 1. Cache Miss
        mock_redis.get.return_value = None
        res_miss = cache_service.get_cached_response("RAG là gì?", collection_name="docs", top_k=5)
        assert res_miss is None

        # 2. Cache Hit
        mock_redis.get.return_value = cached_json
        res_hit = cache_service.get_cached_response("RAG là gì?", collection_name="docs", top_k=5)
        assert res_hit is not None
        assert res_hit.answer == mock_response.answer
        assert len(res_hit.sources) == 1
        assert res_hit.sources[0].content == "RAG chunk"


def test_set_cached_response() -> None:
    """set_cached_response serializes ChatResponse and sets Redis key with TTL."""
    mock_redis = MagicMock()
    mock_response = ChatResponse(
        answer="Test answer",
        sources=[SourceDocument(content="Test source")],
    )

    with patch("src.services.cache_service.get_redis_client", return_value=mock_redis):
        ok = cache_service.set_cached_response(
            query="Câu hỏi kiểm tra",
            response=mock_response,
            collection_name="col1",
            top_k=3,
        )
        assert ok is True
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert "rag:cache:exact:col1:3:" in args[0]
        assert "Test answer" in args[1]
        assert kwargs["ex"] == settings.QUERY_CACHE_TTL_SECONDS


def test_semantic_cache_hit_and_miss() -> None:
    """Semantic cache hits when cosine similarity >= threshold, and misses when below."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None  # Exact miss

    # Stored query: vector [1.0, 0.0]
    cached_payload = {
        "query": "RAG là gì?",
        "embedding": [1.0, 0.0],
        "response": {"answer": "Cached semantic answer", "sources": []},
    }
    mock_redis.hgetall.return_value = {"hash_1": json.dumps(cached_payload)}

    with patch("src.services.cache_service.get_redis_client", return_value=mock_redis), \
         patch.object(settings, "ENABLE_SEMANTIC_CACHE", True), \
         patch.object(settings, "SEMANTIC_CACHE_THRESHOLD", 0.90):

        # Test A: Very similar query vector [0.99, 0.05] (Cosine sim ~ 0.998 >= 0.90) -> HIT
        hit = cache_service.get_cached_response(
            query="Định nghĩa RAG?",
            collection_name="docs",
            top_k=5,
            query_embedding=[0.99, 0.05],
        )
        assert hit is not None
        assert hit.answer == "Cached semantic answer"

        # Test B: Dissimilar query vector [0.0, 1.0] (Cosine sim 0.0 < 0.90) -> MISS
        miss = cache_service.get_cached_response(
            query="Thời tiết hôm nay?",
            collection_name="docs",
            top_k=5,
            query_embedding=[0.0, 1.0],
        )
        assert miss is None


def test_cache_service_fail_open_on_redis_error() -> None:
    """get_cached_response gracefully returns None when Redis fails or is offline."""
    mock_redis = MagicMock()
    mock_redis.get.side_effect = RuntimeError("Redis connection lost")

    with patch("src.services.cache_service.get_redis_client", return_value=mock_redis):
        # Should NOT raise, must return None
        res = cache_service.get_cached_response("query", collection_name="col", top_k=5)
        assert res is None


def test_invalidate_collection_cache() -> None:
    """invalidate_collection_cache deletes all matching exact and semantic keys."""
    mock_redis = MagicMock()
    mock_redis.scan_iter.side_effect = [
        iter(["rag:cache:exact:docs:5:hash1", "rag:cache:exact:docs:5:hash2"]),
        iter(["rag:cache:semantic:docs:5"]),
    ]
    mock_redis.delete.return_value = 3

    with patch("src.services.cache_service.get_redis_client", return_value=mock_redis):
        count = cache_service.invalidate_collection_cache("docs")
        assert count == 3
        mock_redis.delete.assert_called_once_with(
            "rag:cache:exact:docs:5:hash1",
            "rag:cache:exact:docs:5:hash2",
            "rag:cache:semantic:docs:5",
        )


def test_chat_endpoint_serves_from_cache_and_bypasses_retrieval(client) -> None:
    """chat_endpoint serves directly from Redis cache without invoking retrieve or LLM."""
    cached_doc = SourceDocument(content="Cached text content", score=0.99, metadata={"chunk_id": "c_cached"})
    cached_response = ChatResponse(
        answer="Câu trả lời lấy tức thì từ Redis cache.",
        sources=[cached_doc],
    )

    with patch("src.services.cache_service.get_cached_response", return_value=cached_response) as mock_get_cache, \
         patch("src.api.v1.endpoints.chat.retrieve") as mock_ret, \
         patch("src.api.v1.endpoints.chat.generate_answer") as mock_gen:

        # 1. Non-streaming call
        resp = client.post("/api/v1/chat", json={"query": "RAG là gì?", "stream": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Câu trả lời lấy tức thì từ Redis cache."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["content"] == "Cached text content"

        # Neither retrieve nor LLM should be called
        mock_ret.assert_not_called()
        mock_gen.assert_not_called()

        # 2. Streaming call
        stream_resp = client.post("/api/v1/chat", json={"query": "RAG là gì?", "stream": True})
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        assert "Câu trả lời lấy tức thì từ Redis cache." in stream_resp.text
        assert "cached" in stream_resp.text

