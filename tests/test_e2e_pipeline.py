"""Integration test for full RAG pipeline flow (ingest -> retrieve -> generate)."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.core.generator import generate_rag_response
from src.core.ingestion import ingest_path
from src.services.vector_db import get_chroma_client, get_collection


def test_e2e_pipeline_orchestration(tmp_path: Path) -> None:
    """Test full workflow: Ingest -> Retrieve -> Generate with mocked Gemini responses."""
    # Check if Chroma server is online; if not, skip
    try:
        client = get_chroma_client()
        client.heartbeat()
    except Exception:
        pytest.skip("ChromaDB server is not running on localhost:8000")

    # Create temporary sample file
    test_file = tmp_path / "rag_guide.txt"
    test_file.write_text(
        "Kỹ thuật RAG kết hợp giữa mô hình ngôn ngữ lớn và hệ thống truy xuất thông tin ngoài.",
        encoding="utf-8",
    )

    test_collection = "pytest_rag_orchestration_col"
    dummy_vector = [0.08] * 768

    mock_generate_answer = "RAG là sự kết hợp giữa mô hình ngôn ngữ lớn và hệ thống truy xuất thông tin ngoài."

    with patch("src.core.ingestion._embed_single_batch", return_value=[dummy_vector]), \
         patch("src.core.retriever.embed_query", return_value=dummy_vector), \
         patch("src.core.generator._generate_with_retry", return_value=mock_generate_answer):

        # 1. Ingest
        ingest_res = ingest_path(test_file, collection_name=test_collection)
        assert ingest_res.files_processed == 1
        assert ingest_res.chunks_created >= 1

        # 2. Retrieve & Generate
        response = generate_rag_response(
            query="RAG kết hợp những gì?",
            top_k=2,
            collection_name=test_collection,
        )

        # 3. Assertions
        assert response.answer == mock_generate_answer
        assert len(response.sources) >= 1
        assert response.sources[0].metadata["file_name"] == "rag_guide.txt"

