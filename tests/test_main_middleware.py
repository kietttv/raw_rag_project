"""Tests for Request Timing Middleware and Centralized Exception Handlers in src.main."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_timing_headers_present(client):
    """Test that X-Process-Time-Ms and X-Request-ID headers are present in response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    assert "X-Request-ID" in response.headers

    # Verify duration is numeric and non-negative
    duration = float(response.headers["X-Process-Time-Ms"])
    assert duration >= 0.0


def test_custom_request_id_preserved(client):
    """Test that client-supplied X-Request-ID header is propagated to response."""
    custom_id = "custom-trace-uuid-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


def test_http_exception_centralized_format(client):
    """Test that HTTPException returns structured JSON with detail and request_id."""
    response = client.get("/api/v1/ingest/non_existent_task_id_99999")
    assert response.status_code == 404

    data = response.json()
    assert "detail" in data
    assert data["status_code"] == 404
    assert data["error_type"] == "http_error"
    assert "request_id" in data
    assert response.headers.get("X-Request-ID") == data["request_id"]
    assert "X-Process-Time-Ms" in response.headers


def test_validation_error_centralized_format(client):
    """Test that RequestValidationError returns structured 422 JSON."""
    # POST to /api/v1/chat with missing query
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422

    data = response.json()
    assert "detail" in data
    assert isinstance(data["detail"], list)  # Pydantic validation errors list
    assert data["status_code"] == 422
    assert data["error_type"] == "validation_error"
    assert "request_id" in data


def test_unhandled_exception_centralized_format(client):
    """Test that an unexpected exception is caught and returns 500 internal_error."""
    @app.get("/test-unhandled-error")
    def error_route():
        raise RuntimeError("Unexpected database crash")

    response = client.get("/test-unhandled-error")
    assert response.status_code == 500

    data = response.json()
    assert data["detail"] == "Internal server error"
    assert data["status_code"] == 500
    assert data["error_type"] == "internal_error"
    assert "request_id" in data
