"""Unit and integration tests for API v1 chat and streaming endpoints."""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.models.schemas import ChatMessage, ChatResponse, SourceDocument


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_chat_non_stream_success(client):
    """Test standard non-streaming chat request returning JSON ChatResponse."""
    mock_sources = [
        SourceDocument(
            content="FastAPI is a modern web framework.",
            score=0.88,
            metadata={"file_name": "fastapi.txt", "chunk_id": "c1"},
        )
    ]
    mock_response = ChatResponse(
        answer="FastAPI là một web framework hiện đại.",
        sources=mock_sources,
    )

    with patch("src.api.v1.endpoints.chat.retrieve", return_value=mock_sources) as mock_ret, \
         patch("src.api.v1.endpoints.chat.generate_answer", return_value=mock_response) as mock_gen:

        payload = {
            "query": "FastAPI là gì?",
            "top_k": 3,
            "collection_name": "my_docs",
            "min_score": 0.5,
            "stream": False,
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["answer"] == "FastAPI là một web framework hiện đại."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["content"] == "FastAPI is a modern web framework."

        mock_ret.assert_called_once_with(
            query="FastAPI là gì?",
            top_k=3,
            min_score=0.5,
            collection_name="my_docs",
        )
        mock_gen.assert_called_once()


def test_chat_stream_success(client):
    """Test streaming chat request returning Server-Sent Events."""
    mock_sources = [
        SourceDocument(
            content="Chroma is an AI-native vector database.",
            score=0.95,
            metadata={"file_name": "chroma.txt"},
        )
    ]
    tokens = ["Chroma ", "là ", "vector ", "database."]

    with patch("src.api.v1.endpoints.chat.retrieve", return_value=mock_sources), \
         patch("src.api.v1.endpoints.chat.stream_answer", return_value=iter(tokens)):

        payload = {
            "query": "Chroma là gì?",
            "stream": True,
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE stream
        raw_text = response.text
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        events = []
        for i in range(0, len(lines), 2):
            if lines[i].startswith("event:") and lines[i + 1].startswith("data:"):
                ev_name = lines[i].replace("event:", "").strip()
                ev_data = json.loads(lines[i + 1].replace("data:", "").strip())
                events.append((ev_name, ev_data))

        assert len(events) >= 3
        # First event is sources
        assert events[0][0] == "sources"
        assert len(events[0][1]) == 1
        assert events[0][1][0]["content"] == "Chroma is an AI-native vector database."

        # Token events
        token_events = [ev for ev in events if ev[0] == "token"]
        assert len(token_events) == 4
        reconstructed = "".join([t[1]["text"] for t in token_events])
        assert reconstructed == "Chroma là vector database."

        # Final event is done
        assert events[-1][0] == "done"
        assert events[-1][1]["answer_length"] == len("Chroma là vector database.")


def test_chat_stream_error_handling(client):
    """Test SSE error event emission when stream generator fails."""
    mock_sources = [
        SourceDocument(content="Test doc", metadata={"file_name": "test.txt"})
    ]

    def failing_stream(*args, **kwargs):
        yield "Initial token "
        raise RuntimeError("API quota exceeded mid-stream")

    with patch("src.api.v1.endpoints.chat.retrieve", return_value=mock_sources), \
         patch("src.api.v1.endpoints.chat.stream_answer", side_effect=failing_stream):

        payload = {"query": "Test query", "stream": True}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200

        raw_text = response.text
        assert "event: error" in raw_text
        assert "LLM generation failed during streaming" in raw_text


def test_chat_empty_query_validation(client):
    """Test 422 Unprocessable Entity when query is empty."""
    payload = {"query": "", "stream": False}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422


def test_chat_retrieval_failure_503(client):
    """Test 503 Service Unavailable when vector database or retrieval fails."""
    with patch("src.api.v1.endpoints.chat.retrieve", side_effect=RuntimeError("DB connection failed")):
        payload = {"query": "Test question"}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 503
        assert "unreachable" in response.json()["detail"].lower()


def test_chat_generator_failure_502(client):
    """Test 502 Bad Gateway when LLM generation fails in non-streaming mode."""
    with patch("src.api.v1.endpoints.chat.retrieve", return_value=[]), \
         patch("src.api.v1.endpoints.chat.generate_answer", side_effect=RuntimeError("Gemini 500")):
        payload = {"query": "Test question"}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 502
        assert "failed" in response.json()["detail"].lower()


def test_chat_history_trimming(client):
    """Test that history exceeding MAX_HISTORY_TURNS is trimmed appropriately."""
    history_turns = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Turn {i}")
        for i in range(20)
    ]

    mock_response = ChatResponse(answer="Answer", sources=[])

    with patch("src.api.v1.endpoints.chat.retrieve", return_value=[]), \
         patch("src.api.v1.endpoints.chat.generate_answer", return_value=mock_response) as mock_gen:

        payload = {
            "query": "Current question",
            "history": [msg.model_dump() for msg in history_turns],
            "stream": False,
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200

        mock_gen.assert_called_once()
        passed_history = mock_gen.call_args.kwargs.get("history") or mock_gen.call_args[1].get("history")
        assert len(passed_history) == settings.MAX_HISTORY_TURNS * 2
        # Should be the most recent turns
        assert passed_history[-1].content == "Turn 19"

