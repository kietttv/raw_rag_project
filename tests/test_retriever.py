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


def test_tokenize_text_preserves_codes_and_vietnamese() -> None:
    """tokenize_text preserves Vietnamese accents and splits compound codes into components."""
    tokens = retriever.tokenize_text("Nhân viên mã số NV-001 đạt chứng chỉ ISO_9001!")
    assert "nhân" in tokens
    assert "viên" in tokens
    assert "nv-001" in tokens
    assert "nv" in tokens
    assert "001" in tokens
    assert "iso_9001" in tokens
    assert "iso" in tokens
    assert "9001" in tokens

    assert retriever.tokenize_text("") == []


def test_reciprocal_rank_fusion_score_calculation() -> None:
    """RRF correctly calculates scores: 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)."""
    dense_hits = [
        {"id": "doc1", "content": "Doc 1 text", "score": 0.85, "metadata": {"file_name": "f1.pdf"}},
        {"id": "doc2", "content": "Doc 2 text", "score": 0.70, "metadata": {"file_name": "f2.pdf"}},
    ]
    bm25_hits = [
        {"id": "doc2", "content": "Doc 2 text", "score": 4.5, "metadata": {"file_name": "f2.pdf"}},
        {"id": "doc3", "content": "Doc 3 text", "score": 3.2, "metadata": {"file_name": "f3.pdf"}},
    ]

    # rrf_k = 60
    # doc1: dense rank 1 -> 1 / (60 + 1) = 1/61 = ~0.016393
    # doc2: dense rank 2, bm25 rank 1 -> 1 / (60 + 2) + 1 / (60 + 1) = 1/62 + 1/61 = ~0.032522
    # doc3: bm25 rank 2 -> 1 / (60 + 2) = 1/62 = ~0.016129
    fused = retriever.reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=60)

    assert len(fused) == 3
    # doc2 appeared in both -> ranked #1
    assert fused[0]["id"] == "doc2"
    assert fused[0]["metadata"]["dense_score"] == 0.70
    assert fused[0]["metadata"]["bm25_score"] == 4.5
    assert pytest.approx(fused[0]["metadata"]["rrf_score"], rel=1e-3) == (1 / 62 + 1 / 61)

    # doc1 is ranked #2
    assert fused[1]["id"] == "doc1"
    assert pytest.approx(fused[1]["metadata"]["rrf_score"], rel=1e-3) == (1 / 61)

    # doc3 is ranked #3
    assert fused[2]["id"] == "doc3"
    assert pytest.approx(fused[2]["metadata"]["rrf_score"], rel=1e-3) == (1 / 62)


def test_bm25_search_and_caching() -> None:
    """bm25_search finds exact keyword matches and leverages in-memory caching."""
    retriever.reset_bm25_cache()

    sample_corpus = [
        {"id": "doc_code", "content": "Hợp đồng kinh tế số HĐ-2024-999 ký ngày 15/01", "metadata": {}},
        {"id": "doc_general", "content": "Chính sách nhân sự và chế độ phúc lợi", "metadata": {}},
    ]

    with patch("src.services.vector_db.get_all_documents", return_value=sample_corpus) as mock_get_docs:
        # First search: builds cache
        hits1 = retriever.bm25_search("HĐ-2024-999", top_k=2, collection_name="cache_test_col")
        assert len(hits1) == 1
        assert hits1[0]["id"] == "doc_code"
        assert hits1[0]["score"] > 0
        assert mock_get_docs.call_count == 1

        # Second search: uses cached BM25 index
        hits2 = retriever.bm25_search("HĐ-2024-999", top_k=2, collection_name="cache_test_col")
        assert len(hits2) == 1
        assert hits2[0]["id"] == "doc_code"
        assert mock_get_docs.call_count == 2  # checks doc_ids then reuses bm25

        # Cache reset
        retriever.reset_bm25_cache()


def test_bm25_search_handles_exception_gracefully() -> None:
    """bm25_search returns empty list without raising if vector_db.get_all_documents fails."""
    with patch("src.services.vector_db.get_all_documents", side_effect=RuntimeError("Connection lost")):
        hits = retriever.bm25_search("mã số test", top_k=5)
        assert hits == []


def test_retrieve_hybrid_boosts_exact_code() -> None:
    """retrieve() combines dense and BM25 via RRF, prioritizing exact code matches."""
    retriever.reset_bm25_cache()

    dense_hits = [
        {"id": "c_general", "content": "Quy chế chung của công ty", "score": 0.88, "metadata": {"chunk_id": "c_general"}},
        {"id": "c_code", "content": "Mã nhân viên NV-999 là trưởng phòng", "score": 0.65, "metadata": {"chunk_id": "c_code"}},
    ]
    bm25_hits = [
        {"id": "c_code", "content": "Mã nhân viên NV-999 là trưởng phòng", "score": 5.2, "metadata": {"chunk_id": "c_code"}},
    ]

    with patch("src.core.retriever.embed_query", return_value=[0.1] * 768), \
         patch("src.services.vector_db.query_vectors", return_value=dense_hits), \
         patch("src.core.retriever.bm25_search", return_value=bm25_hits):

        docs = retriever.retrieve("NV-999", top_k=2, use_hybrid=True)

        assert len(docs) == 2
        # c_code was ranked #1 in BM25 and #2 in Dense -> RRF score is higher than c_general (#1 Dense only)
        # c_code RRF: 1/62 + 1/61 = 0.0325
        # c_general RRF: 1/61 = 0.0164
        assert docs[0].metadata["chunk_id"] == "c_code"
        assert docs[0].metadata["bm25_score"] == 5.2
        assert docs[0].metadata["dense_score"] == 0.65
        assert docs[1].metadata["chunk_id"] == "c_general"


