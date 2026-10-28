"""Unit tests for src.core.reranker module."""

from unittest.mock import MagicMock, patch
import pytest

from src.config import settings
from src.core import reranker
from src.models.schemas import SourceDocument


def test_normalize_rerank_score() -> None:
    """normalize_rerank_score applies sigmoid to convert CrossEncoder logits to [0, 1] probabilities."""
    # logit = 0.0 -> 0.5
    assert reranker.normalize_rerank_score(0.0) == 0.5000
    assert reranker.normalize_rerank_score(4.0) == pytest.approx(0.982, abs=1e-3)
    assert reranker.normalize_rerank_score(-4.0) == pytest.approx(0.018, abs=1e-3)
    # Extreme values do not overflow
    assert reranker.normalize_rerank_score(100.0) == 1.0000
    assert reranker.normalize_rerank_score(-100.0) == 0.0000


def test_rerank_empty_query_raises_value_error() -> None:
    """Empty or whitespace-only query raises ValueError."""
    with pytest.raises(ValueError, match="Query must not be empty"):
        reranker.rerank("", [SourceDocument(content="text")])

    with pytest.raises(ValueError, match="Query must not be empty"):
        reranker.rerank("   \n\t  ", [SourceDocument(content="text")])


def test_rerank_invalid_top_k_raises_value_error() -> None:
    """top_k <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        reranker.rerank("query", [SourceDocument(content="text")], top_k=0)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        reranker.rerank("query", [SourceDocument(content="text")], top_k=-2)


def test_rerank_empty_documents_returns_empty_list() -> None:
    """rerank returns empty list immediately if input documents list is empty."""
    res = reranker.rerank("valid query", [])
    assert res == []


def test_rerank_success_reorders_and_slices_top_k() -> None:
    """rerank accurately re-scores candidates and retains only Top-K best chunks."""
    docs = [
        SourceDocument(content="Doc 1 - mediocre", score=0.80, metadata={"chunk_id": "c1"}),
        SourceDocument(content="Doc 2 - highest relevance", score=0.50, metadata={"chunk_id": "c2"}),
        SourceDocument(content="Doc 3 - poor", score=0.75, metadata={"chunk_id": "c3"}),
        SourceDocument(content="Doc 4 - second best", score=0.60, metadata={"chunk_id": "c4"}),
        SourceDocument(content="Doc 5 - worst", score=0.40, metadata={"chunk_id": "c5"}),
    ]

    mock_model = MagicMock()
    # Mocked CrossEncoder logits: c2 (highest: 3.5), c4 (second: 2.0), c1 (1.0), c3 (-1.0), c5 (-3.0)
    mock_model.predict.return_value = [1.0, 3.5, -1.0, 2.0, -3.0]

    with patch("src.core.reranker.get_reranker", return_value=mock_model):
        results = reranker.rerank(
            query="Where is the highest relevance document?",
            documents=docs,
            top_k=3,
        )

        assert len(results) == 3
        # Rank 1: c2
        assert results[0].metadata["chunk_id"] == "c2"
        assert results[0].metadata["initial_score"] == 0.50
        assert results[0].score == pytest.approx(0.9707, abs=1e-3)
        assert results[0].metadata["rerank_score"] == results[0].score

        # Rank 2: c4
        assert results[1].metadata["chunk_id"] == "c4"
        assert results[1].metadata["initial_score"] == 0.60
        assert results[1].score == pytest.approx(0.8808, abs=1e-3)

        # Rank 3: c1
        assert results[2].metadata["chunk_id"] == "c1"
        assert results[2].metadata["initial_score"] == 0.80
        assert results[2].score == pytest.approx(0.7311, abs=1e-3)

        # Assert correct pairs were passed to CrossEncoder
        mock_model.predict.assert_called_once()
        passed_pairs = mock_model.predict.call_args[0][0]
        assert len(passed_pairs) == 5
        assert passed_pairs[0] == ["Where is the highest relevance document?", "Doc 1 - mediocre"]


def test_get_reranker_singleton_and_reset() -> None:
    """get_reranker instantiates CrossEncoder once and reset_reranker clears it."""
    reranker.reset_reranker()
    mock_ce = MagicMock()

    with patch("sentence_transformers.CrossEncoder", return_value=mock_ce) as mock_init:
        m1 = reranker.get_reranker()
        m2 = reranker.get_reranker()
        assert m1 is m2
        assert mock_init.call_count == 1

        reranker.reset_reranker()
        m3 = reranker.get_reranker()
        assert m3 is not None
        assert mock_init.call_count == 2

    reranker.reset_reranker()


def test_retrieve_and_rerank_pipeline() -> None:
    """retrieve_and_rerank orchestrates candidate retrieval followed by reranking."""
    mock_candidates = [
        SourceDocument(content="Chunk A", score=0.9, metadata={"chunk_id": "cA"}),
        SourceDocument(content="Chunk B", score=0.8, metadata={"chunk_id": "cB"}),
        SourceDocument(content="Chunk C", score=0.7, metadata={"chunk_id": "cC"}),
    ]

    mock_reranked = [
        SourceDocument(content="Chunk C", score=0.99, metadata={"chunk_id": "cC"}),
        SourceDocument(content="Chunk A", score=0.95, metadata={"chunk_id": "cA"}),
    ]

    with patch("src.core.retriever.retrieve", return_value=mock_candidates) as mock_ret, \
         patch("src.core.reranker.rerank", return_value=mock_reranked) as mock_rr:

        res = reranker.retrieve_and_rerank(
            query="câu hỏi kiểm tra",
            top_k=2,
            candidate_k=15,
            collection_name="test_col",
        )

        assert res == mock_reranked
        mock_ret.assert_called_once_with(
            query="câu hỏi kiểm tra",
            top_k=15,
            where=None,
            min_score=None,
            collection_name="test_col",
            use_hybrid=True,
            use_reranker=False,
        )
        mock_rr.assert_called_once_with(
            query="câu hỏi kiểm tra",
            documents=mock_candidates,
            top_k=2,
        )


def test_reranker_cli_execution(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI prints formatted reranking report."""
    mock_doc = SourceDocument(
        content="Preview snippet text for reranker CLI",
        score=0.9850,
        metadata={"file_name": "test_guide.pdf", "chunk_id": "c_rerank", "initial_score": 0.0321},
    )

    with patch("sys.argv", ["reranker.py", "Thủ tục đăng ký", "--top-k", "3", "--candidate-k", "10"]), \
         patch("src.core.reranker.retrieve_and_rerank", return_value=[mock_doc]):

        reranker.main()
        captured = capsys.readouterr()

        assert "CROSS-ENCODER RERANKING REPORT" in captured.out
        assert "Query:               Thủ tục đăng ký" in captured.out
        assert "Candidates fetched:  10" in captured.out
        assert "Top-K kept:          1" in captured.out
        assert "Rerank Score: 0.9850 (Initial: 0.0321)" in captured.out
        assert "Source: test_guide.pdf" in captured.out
        assert "Chunk ID: c_rerank" in captured.out

