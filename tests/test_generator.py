"""Unit tests for src.core.generator module."""

from unittest.mock import MagicMock, patch
import pytest

from src.core import generator
from src.models.schemas import ChatMessage, ChatResponse, SourceDocument


def test_format_context_empty() -> None:
    """format_context returns placeholder when sources list is empty."""
    res = generator.format_context([])
    assert "Không có tài liệu" in res


def test_format_context_with_sources() -> None:
    """format_context correctly numbers and prefixes documents."""
    docs = [
        SourceDocument(
            content="Content of first chunk.",
            score=0.92,
            metadata={"file_name": "guide.pdf", "page": 4},
        ),
        SourceDocument(
            content="Content of second chunk.",
            score=0.81,
            metadata={"source": "faq.txt"},
        ),
    ]

    context = generator.format_context(docs)
    assert "[1] Nguồn: guide.pdf, Trang 4" in context
    assert "Content of first chunk." in context
    assert "[2] Nguồn: faq.txt" in context
    assert "Content of second chunk." in context


def test_build_prompt() -> None:
    """build_prompt injects context and question into the prompt template."""
    docs = [
        SourceDocument(
            content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"file_name": "rag.txt"},
        )
    ]
    prompt = generator.build_prompt("What is RAG?", docs)

    assert "NGỮ CẢNH (Context):" in prompt
    assert "RAG stands for Retrieval-Augmented Generation." in prompt
    assert "CÂU HỎI (Question):" in prompt
    assert "What is RAG?" in prompt


def test_generate_answer_empty_query_raises_value_error() -> None:
    """Empty or whitespace-only query raises ValueError."""
    with pytest.raises(ValueError, match="Query must not be empty"):
        generator.generate_answer("", [])

    with pytest.raises(ValueError, match="Query must not be empty"):
        generator.generate_answer("   \n\t  ", [])


def test_generate_answer_success() -> None:
    """generate_answer calls Gemini client and returns ChatResponse with sources."""
    docs = [
        SourceDocument(
            content="Python 3.11 is faster than Python 3.10.",
            score=0.89,
            metadata={"file_name": "python.txt", "chunk_id": "c1"},
        )
    ]

    mock_response = MagicMock()
    mock_response.text = "Python 3.11 có tốc độ nhanh hơn Python 3.10."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.core.generator.get_genai_client", return_value=mock_client):
        res = generator.generate_answer("So sánh Python 3.11 và 3.10?", docs)

        assert isinstance(res, ChatResponse)
        assert res.answer == "Python 3.11 có tốc độ nhanh hơn Python 3.10."
        assert len(res.sources) == 1
        assert res.sources[0].metadata["chunk_id"] == "c1"
        assert res.sources[0].score == 0.89

        mock_client.models.generate_content.assert_called_once()
        _, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["model"] == generator.settings.GENERATION_MODEL_NAME
        assert kwargs["config"].temperature == generator.settings.GENERATION_TEMPERATURE


def test_generate_rag_response_e2e() -> None:
    """generate_rag_response links retriever output to generation."""
    mock_docs = [
        SourceDocument(
            content="Retrieval Augmented Generation combines retrieval and LLM.",
            score=0.91,
            metadata={"file_name": "rag.pdf", "page": 1},
        )
    ]

    with patch("src.core.generator.retrieve", return_value=mock_docs) as mock_ret, \
         patch("src.core.generator._generate_with_retry", return_value="RAG là kỹ thuật kết hợp...") as mock_gen:
        
        res = generator.generate_rag_response("RAG là gì?", top_k=2)

        mock_ret.assert_called_once_with(
            query="RAG là gì?",
            top_k=2,
            min_score=None,
            collection_name=None,
        )
        assert res.answer == "RAG là kỹ thuật kết hợp..."
        assert res.sources == mock_docs


def test_generator_cli_execution(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI formats and prints full RAG answer and cited sources."""
    mock_res = ChatResponse(
        answer="Hà Nội là thủ đô của Việt Nam.",
        sources=[
            SourceDocument(
                content="Hà Nội là thủ đô và thành phố trực thuộc trung ương...",
                score=0.96,
                metadata={"file_name": "vietnam.txt", "chunk_id": "c_hanoi"},
            )
        ],
    )

    with patch("sys.argv", ["generator.py", "Thủ đô của Việt Nam là gì?", "--top-k", "2"]), \
         patch("src.core.generator.generate_rag_response", return_value=mock_res):
        
        generator.main()
        captured = capsys.readouterr()

        assert "RAG ANSWER GENERATION REPORT" in captured.out
        assert "QUESTION: Thủ đô của Việt Nam là gì?" in captured.out
        assert "Hà Nội là thủ đô của Việt Nam." in captured.out
        assert "SOURCES CITED (1 references):" in captured.out
        assert "Score: 0.9600 | Source: vietnam.txt" in captured.out
        assert "Chunk ID: c_hanoi" in captured.out


def test_build_prompt_with_history() -> None:
    """build_prompt correctly embeds conversation history."""
    docs = [
        SourceDocument(
            content="ChromaDB is an open-source vector store.",
            metadata={"file_name": "chroma.txt"},
        )
    ]
    history = [
        ChatMessage(role="user", content="Tôi là ai?"),
        ChatMessage(role="assistant", content="Bạn là người dùng hệ thống."),
    ]
    prompt = generator.build_prompt("ChromaDB là gì?", docs, history=history)

    assert "LỊCH SỬ HỘI THOẠI" in prompt
    assert "Người dùng: Tôi là ai?" in prompt
    assert "Trợ lý: Bạn là người dùng hệ thống." in prompt
    assert "ChromaDB is an open-source vector store." in prompt
    assert "ChromaDB là gì?" in prompt


def test_stream_answer_yields_tokens() -> None:
    """stream_answer yields individual token chunks from stream."""
    docs = [
        SourceDocument(
            content="Sample text",
            metadata={"file_name": "sample.txt"},
        )
    ]
    chunk1 = MagicMock()
    chunk1.text = "Xin "
    chunk2 = MagicMock()
    chunk2.text = "chào!"
    chunk3 = MagicMock()
    chunk3.text = ""  # Should be skipped

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [chunk1, chunk2, chunk3]

    with patch("src.core.generator.get_genai_client", return_value=mock_client):
        tokens = list(generator.stream_answer("Chào bạn", docs))
        assert tokens == ["Xin ", "chào!"]


def test_get_or_create_context_cache_creates_and_reuses() -> None:
    """get_or_create_context_cache creates a Gemini cache once and reuses it on subsequent calls."""
    generator.reset_context_cache()

    mock_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.name = "cachedContents/sample-cache-123"
    mock_client.caches.create.return_value = mock_cache

    static_text = "Tài liệu quy chế bảo mật thông tin nội bộ của công ty..."

    # Call 1: creates cache
    c1 = generator.get_or_create_context_cache(mock_client, static_context=static_text)
    assert c1 == "cachedContents/sample-cache-123"
    assert mock_client.caches.create.call_count == 1

    # Call 2: reuses existing cache from RAM without calling API
    c2 = generator.get_or_create_context_cache(mock_client, static_context=static_text)
    assert c2 == "cachedContents/sample-cache-123"
    assert mock_client.caches.create.call_count == 1

    generator.reset_context_cache()


def test_get_or_create_context_cache_fallback_on_api_error() -> None:
    """get_or_create_context_cache falls back to None and skips repeat failures if API rejects cache."""
    generator.reset_context_cache()

    mock_client = MagicMock()
    mock_client.caches.create.side_effect = RuntimeError("400 Cached content is too small")

    static_text = "Short text"

    # Call 1: fails and catches exception
    c1 = generator.get_or_create_context_cache(mock_client, static_context=static_text)
    assert c1 is None
    assert mock_client.caches.create.call_count == 1

    # Call 2: remembered as failed hash, skips calling API again
    c2 = generator.get_or_create_context_cache(mock_client, static_context=static_text)
    assert c2 is None
    assert mock_client.caches.create.call_count == 1

    generator.reset_context_cache()


def test_generate_answer_with_context_cache() -> None:
    """generate_answer sends cached_content in GenerateContentConfig when cache is available."""
    docs = [
        SourceDocument(content="Retrieved dynamic chunk", metadata={"chunk_id": "c1"})
    ]

    mock_response = MagicMock()
    mock_response.text = "Answer generated with context cache."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.core.generator.get_genai_client", return_value=mock_client), \
         patch("src.core.generator.get_or_create_context_cache", return_value="cachedContents/cached-abc"):

        res = generator.generate_answer(
            query="Câu hỏi cần trả lời",
            sources=docs,
            use_context_cache=True,
            static_context="Tài liệu tĩnh",
        )

        assert res.answer == "Answer generated with context cache."
        mock_client.models.generate_content.assert_called_once()
        _, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["config"].cached_content == "cachedContents/cached-abc"
        # When cache is active, prompt only contains dynamic retrieved context, not full instructions
        assert "NGỮ CẢNH TRUY XUẤT (Retrieved Context):" in kwargs["contents"]


def test_stream_answer_with_context_cache() -> None:
    """stream_answer sends cached_content in GenerateContentConfig during streaming."""
    docs = [SourceDocument(content="Chunk stream test")]

    chunk = MagicMock()
    chunk.text = "Token streamed."

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [chunk]

    with patch("src.core.generator.get_genai_client", return_value=mock_client), \
         patch("src.core.generator.get_or_create_context_cache", return_value="cachedContents/stream-cache-xyz"):

        tokens = list(generator.stream_answer(
            query="Câu hỏi stream",
            sources=docs,
            use_context_cache=True,
            static_context="Tài liệu mẫu",
        ))

        assert tokens == ["Token streamed."]
        mock_client.models.generate_content_stream.assert_called_once()
        _, kwargs = mock_client.models.generate_content_stream.call_args
        assert kwargs["config"].cached_content == "cachedContents/stream-cache-xyz"


def test_load_static_context_file(tmp_path) -> None:
    """load_static_context successfully reads contents from existing text file."""
    static_file = tmp_path / "system_policy.txt"
    static_file.write_text("Chính sách hệ thống toàn cầu năm 2026.", encoding="utf-8")

    loaded = generator.load_static_context(static_file)
    assert "Chính sách hệ thống toàn cầu năm 2026." in loaded

    # Non-existing path returns empty string
    assert generator.load_static_context("non_existent_path.txt") == ""


def test_load_prompt_file_and_fallback() -> None:
    """_load_prompt_file reads prompt from file and falls back to default if file is missing."""
    # Existing prompt file
    system_inst = generator._load_prompt_file("system_instruction.txt", "fallback text")
    assert "Bạn là trợ lý AI trả lời câu hỏi" in system_inst

    # Missing file uses fallback
    fallback_result = generator._load_prompt_file("non_existent_prompt.txt", "my custom fallback")
    assert fallback_result == "my custom fallback"



