"""Unit and integration tests for API v1 document ingestion endpoints."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.core.ingestion import IngestResult
from src.main import app
from src.models.schemas import TaskStatus
from src.services.task_store import task_store


@pytest.fixture(autouse=True)
def clean_task_store():
    """Ensure clean task store before and after each test."""
    task_store.clear()
    yield
    task_store.clear()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "core-rag-api"}


def test_ingest_txt_file_success(client):
    """Test successful async ingestion of a valid .txt file."""
    fake_result = IngestResult(
        files_processed=1,
        chunks_created=4,
        failed_files=[],
        collection_name="documents",
    )

    with patch("src.api.v1.endpoints.ingest.ingest_path", return_value=fake_result) as mock_ingest:
        file_content = b"Hello, this is a test document content for RAG."
        files = {"file": ("sample.txt", io.BytesIO(file_content), "text/plain")}

        response = client.post("/api/v1/ingest", files=files)
        assert response.status_code == 202

        data = response.json()
        assert "task_id" in data
        assert data["status"] in (TaskStatus.PENDING.value, TaskStatus.COMPLETED.value)
        assert data["file_name"] == "sample.txt"

        task_id = data["task_id"]

        # In TestClient, BackgroundTasks run synchronously before response completes,
        # so polling task status should reflect COMPLETED.
        status_res = client.get(f"/api/v1/ingest/{task_id}")
        assert status_res.status_code == 200
        task_detail = status_res.json()
        assert task_detail["task_id"] == task_id
        assert task_detail["status"] == TaskStatus.COMPLETED.value
        assert task_detail["chunks_created"] == 4
        assert task_detail["files_processed"] == 1
        assert task_detail["failed_files"] == []
        assert task_detail["error"] is None

        mock_ingest.assert_called_once()


def test_ingest_with_custom_collection(client):
    """Test ingestion with custom collection_name form parameter."""
    fake_result = IngestResult(
        files_processed=1,
        chunks_created=2,
        failed_files=[],
        collection_name="custom_kb",
    )

    with patch("src.api.v1.endpoints.ingest.ingest_path", return_value=fake_result) as mock_ingest:
        file_content = b"Content for custom collection."
        files = {"file": ("manual.txt", io.BytesIO(file_content), "text/plain")}
        data = {"collection_name": "custom_kb"}

        response = client.post("/api/v1/ingest", files=files, data=data)
        assert response.status_code == 202
        task_id = response.json()["task_id"]

        status_res = client.get(f"/api/v1/ingest/{task_id}")
        assert status_res.status_code == 200
        assert status_res.json()["collection_name"] == "custom_kb"

        args, kwargs = mock_ingest.call_args
        assert kwargs.get("collection_name") == "custom_kb"


def test_ingest_unsupported_file_extension(client):
    """Test uploading unsupported file format (.docx) returns 415."""
    files = {"file": ("test.docx", io.BytesIO(b"binary docx content"), "application/octet-stream")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_ingest_empty_file(client):
    """Test uploading an empty 0-byte file returns 400."""
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ingest_file_exceeds_max_size(client, monkeypatch):
    """Test uploading file exceeding MAX_UPLOAD_MB returns 413."""
    from src.config import settings

    # Temporarily set max size to very small limit
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0.00001)  # ~10 bytes

    big_content = b"A" * 200
    files = {"file": ("large.txt", io.BytesIO(big_content), "text/plain")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 413
    assert "exceeds the maximum allowed limit" in response.json()["detail"]


def test_ingest_background_task_failure(client):
    """Test when ingestion pipeline encounters an error during processing."""
    with patch(
        "src.api.v1.endpoints.ingest.ingest_path",
        side_effect=RuntimeError("ChromaDB connection timed out"),
    ):
        files = {"file": ("sample.txt", io.BytesIO(b"valid content"), "text/plain")}
        response = client.post("/api/v1/ingest", files=files)
        assert response.status_code == 202
        task_id = response.json()["task_id"]

        status_res = client.get(f"/api/v1/ingest/{task_id}")
        assert status_res.status_code == 200
        detail = status_res.json()
        assert detail["status"] == TaskStatus.FAILED.value
        assert "ChromaDB connection timed out" in detail["error"]


def test_get_nonexistent_task(client):
    """Test querying non-existent task_id returns 404."""
    response = client.get("/api/v1/ingest/non_existent_id_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

