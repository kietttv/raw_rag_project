"""Unit and integration tests for Tracing and User Feedback (src/services/tracing.py)."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.models.schemas import ChatResponse, SourceDocument
from src.services import tracing
from src.services.tracing import TraceContext, flush_tracing, get_langfuse_client, record_user_feedback, start_trace


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_tracing_no_op_when_disabled():
    """Test that tracing methods operate safely in no-op mode when ENABLE_TRACING is False."""
    with patch.object(settings, "ENABLE_TRACING", False):
        tracing.reset_tracing_client()
        assert get_langfuse_client() is None

        ctx = start_trace("Câu hỏi kiểm thử no-op", trace_id="trace-test-1")
        assert ctx.trace_id == "trace-test-1"
        assert ctx.langfuse_trace is None

        # Recording spans and generations shouldn't raise any exceptions
        ctx.record_span(name="test_span", input_data={"k": 5}, duration_ms=12.5)
        ctx.record_generation(
            name="test_generation",
            model="gemini-3.5-flash",
            prompt="Hello",
            completion="Hi",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            duration_ms=45.0,
        )
        ctx.end(output={"status": "done"})

        # Feedback in no-op mode returns True
        assert record_user_feedback("trace-test-1", score=1, comment="Good") is True


def test_tracing_active_with_mocked_langfuse():
    """Test that active tracing records traces, spans, generations with token usage, and feedback."""
    mock_span = MagicMock()
    mock_gen = MagicMock()
    mock_trace = MagicMock()
    mock_trace.span.return_value = mock_span
    mock_trace.generation.return_value = mock_gen

    mock_client = MagicMock()
    mock_client.trace.return_value = mock_trace

    ctx = TraceContext(
        trace_id="tid-999",
        name="rag_chat_query",
        input_data={"query": "RAG là gì?"},
        metadata={"top_k": 3},
        client=mock_client,
    )

    mock_client.trace.assert_called_once()
    assert ctx.langfuse_trace == mock_trace

    # Record Retriever Span
    ctx.record_span(name="retriever", input_data={"top_k": 3}, output_data={"count": 3}, duration_ms=25.0)
    mock_trace.span.assert_called_once()
    mock_span.end.assert_called_once()

    # Record Generation with Token Usage
    ctx.record_generation(
        name="gemini_generation",
        model="gemini-3.5-flash",
        prompt="RAG là gì?",
        completion="RAG là viết tắt của...",
        prompt_tokens=120,
        completion_tokens=45,
        total_tokens=165,
        duration_ms=450.0,
    )
    mock_trace.generation.assert_called_once()
    _, gen_kwargs = mock_trace.generation.call_args
    assert gen_kwargs["usage"] == {"input": 120, "output": 45, "total": 165}
    mock_gen.end.assert_called_once()

    # End Trace
    ctx.end(output={"answer_length": 50})
    mock_trace.update.assert_called_once()

    # Test record feedback
    with patch("src.services.tracing.get_langfuse_client", return_value=mock_client):
        res = record_user_feedback(trace_id="tid-999", score=1, comment="Rất tốt")
        assert res is True
        mock_client.score.assert_called_once_with(
            trace_id="tid-999",
            name="user_feedback",
            value=1.0,
            comment="Rất tốt",
        )
        mock_client.flush.assert_called_once()


def test_flush_tracing_handles_exception_safely():
    """Test flush_tracing does not crash if client.flush raises an error."""
    mock_client = MagicMock()
    mock_client.flush.side_effect = RuntimeError("Network timeout")

    with patch("src.services.tracing.get_langfuse_client", return_value=mock_client):
        # Should not raise exception
        flush_tracing()


def test_api_submit_feedback_thumbs_up(client):
    """Test POST /api/v1/chat/feedback with positive score (1)."""
    payload = {
        "trace_id": "trace-uuid-12345",
        "score": 1,
        "comment": "Câu trả lời đúng trọng tâm tài liệu",
    }
    response = client.post("/api/v1/chat/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["trace_id"] == "trace-uuid-12345"
    assert data["score"] == 1


def test_api_submit_feedback_thumbs_down(client):
    """Test POST /api/v1/chat/feedback with negative score (-1)."""
    payload = {
        "trace_id": "trace-uuid-67890",
        "score": -1,
        "comment": "Thiếu thông tin chi tiết",
    }
    response = client.post("/api/v1/chat/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["score"] == -1


def test_api_submit_feedback_invalid_score(client):
    """Test POST /api/v1/chat/feedback rejects out-of-range scores."""
    payload = {
        "trace_id": "trace-abc",
        "score": 5,  # Valid range is -1 to 1
    }
    response = client.post("/api/v1/chat/feedback", json=payload)
    assert response.status_code == 422


def test_api_chat_returns_trace_id_in_response(client):
    """Test that POST /api/v1/chat attaches trace_id in body and X-Trace-ID header."""
    mock_sources = [
        SourceDocument(content="ChromaDB docs", metadata={"file_name": "chroma.txt"})
    ]
    mock_resp = ChatResponse(
        answer="ChromaDB là vector database.",
        sources=mock_sources,
    )

    with patch("src.api.v1.endpoints.chat.retrieve", return_value=mock_sources), \
         patch("src.api.v1.endpoints.chat.generate_answer", return_value=mock_resp):

        payload = {"query": "ChromaDB là gì?", "stream": False}
        headers = {"X-Request-ID": "test-req-custom-999"}
        response = client.post("/api/v1/chat", json=payload, headers=headers)
        assert response.status_code == 200

        # Verify X-Trace-ID header matches request-id or trace-id
        assert response.headers["X-Trace-ID"] == "test-req-custom-999"

        # Verify trace_id in response JSON
        data = response.json()
        assert data["trace_id"] == "test-req-custom-999"
        assert data["answer"] == "ChromaDB là vector database."

