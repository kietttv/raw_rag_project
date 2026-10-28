"""Unit tests for src.core.retriever module."""

from unittest.mock import MagicMock, patch
import pytest

from src.config import settings
from src.core import retriever
from src.models.schemas import SourceDocument


def test_retrieve_empty_query_raises_value_error() -> None:
    """Empty or whitespace-only query raises ValueError."""
    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("    \n\t  ")


def test_retrieve_invalid_top_k_raises_value_error() -> None:
    """top_k <= 0 or top_k > MAX_TOP_K raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be between 1 and"):
        retriever.retrieve("valid query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be between 1 and"):
        retriever.retrieve("valid query", top_k=-5)

    with pytest.raises(ValueError, match="top_k must be between 1 and"):
        retriever.retrieve("valid query", top_k=settings.MAX_TOP_K + 1)


def test_embed_query_calls_with_retrieval_query_task_type() -> None:
    """embed_query must pass task_type='RETRIEVAL_QUERY' to embed_texts."""
    dummy_vec = [0.1] * 768
    with patch("src.core.retriever.embed_texts", return_value=[dummy_vec]) as mock_embed:
        result = retriever.embed_query("câu hỏi mẫu")
        assert result == dummy_vec
        mock_embed.assert_called_once_with(["câu hỏi mẫu"], task_type="RETRIEVAL_QUERY")


def test_embed_query_empty_return_raises_runtime_error() -> None:
    """embed_query raises RuntimeError if no embeddings returned."""
    with patch("src.core.retriever.embed_texts", return_value=[]):
        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            retriever.embed_query("test query")


def test_retrieve_success() -> None:
    """retrieve correctly maps ChromaDB hits to list[SourceDocument]."""
    mock_hits = [
        {
            "id": "chunk-123",
            "content": "RAG architecture explanation",
            "metadata": {"source": "rag_doc.pdf", "page": 1},
            "distance": 0.15,
            "score": 0.85,
        },
        {
            "id": "chunk-456",
            "content": "LLM fine-tuning vs RAG",
            "metadata": {"source": "rag_doc.pdf", "page": 2},
            "distance": 0.28,
            "score": 0.72,
        },
    ]

    with patch("src.core.retriever.embed_query", return_value=[0.1] * 768), \
         patch("src.services.vector_db.query_vectors", return_value=mock_hits) as mock_qv:
        
        docs = retriever.retrieve("RAG là gì?", top_k=2, collection_name="custom_col")

        assert len(docs) == 2
        assert isinstance(docs[0], SourceDocument)
        assert docs[0].content == "RAG architecture explanation"
        assert docs[0].score == 0.85
        assert docs[0].metadata["chunk_id"] == "chunk-123"
        assert docs[0].metadata["source"] == "rag_doc.pdf"
        assert docs[0].metadata["page"] == 1

        assert docs[1].score == 0.72
        assert docs[1].metadata["chunk_id"] == "chunk-456"

        mock_qv.assert_called_once_with(
            query_embedding=[0.1] * 768,
            top_k=2,
            where=None,
            collection_name="custom_col",
        )


def test_retrieve_uses_default_top_k() -> None:
    """retrieve falls back to settings.DEFAULT_TOP_K when top_k is None."""
    with patch("src.core.retriever.embed_query", return_value=[0.1] * 768), \
         patch("src.services.vector_db.query_vectors", return_value=[]) as mock_qv:
        
        retriever.retrieve("RAG query", top_k=None)
        mock_qv.assert_called_once()
        _, kwargs = mock_qv.call_args
        assert kwargs["top_k"] == settings.DEFAULT_TOP_K


def test_retrieve_min_score_filtering() -> None:
    """min_score filters out items with score strictly lower than threshold."""
    mock_hits = [
        {"id": "c1", "content": "Good match", "score": 0.90, "metadata": {}},
        {"id": "c2", "content": "Moderate match", "score": 0.75, "metadata": {}},
        {"id": "c3", "content": "Poor match", "score": 0.40, "metadata": {}},
    ]

    with patch("src.core.retriever.embed_query", return_value=[0.1] * 768), \
         patch("src.services.vector_db.query_vectors", return_value=mock_hits):
        
        docs = retriever.retrieve("sample query", min_score=0.75)
        assert len(docs) == 2
        assert [d.score for d in docs] == [0.90, 0.75]


def test_retrieve_empty_results() -> None:
    """retrieve safely returns an empty list if vector DB has no hits."""
    with patch("src.core.retriever.embed_query", return_value=[0.1] * 768), \
         patch("src.services.vector_db.query_vectors", return_value=[]):
        
        docs = retriever.retrieve("something obscure")
        assert docs == []


def test_main_cli_execution(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI prints formatted retrieval report."""
    mock_doc = SourceDocument(
        content="Test content chunk for CLI preview",
        score=0.88,
        metadata={"source": "test.txt", "chunk_id": "c_abc"},
    )
    with patch("sys.argv", ["retriever.py", "RAG query", "--top-k", "3"]), \
         patch("src.core.retriever.retrieve", return_value=[mock_doc]):
        
        retriever.main()
        captured = capsys.readouterr()
        assert "RETRIEVAL SEARCH RESULTS" in captured.out
        assert "Query:        RAG query" in captured.out
        assert "Found:        1 chunks" in captured.out
        assert "[#1] Score: 0.8800" in captured.out
        assert "Chunk ID: c_abc" in captured.out

