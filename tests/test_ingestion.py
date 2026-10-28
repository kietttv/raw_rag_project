"""Unit tests for src.core.ingestion pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from src.core import ingestion


def test_load_file_txt(tmp_path: Path) -> None:
    """Test loading a plain UTF-8 text file."""
    txt_file = tmp_path / "hello.txt"
    txt_file.write_text("Hello world RAG pipeline.\nSecond line content.", encoding="utf-8")

    docs = ingestion.load_file(txt_file)
    assert len(docs) == 1
    assert "Hello world RAG pipeline." in docs[0].page_content
    assert docs[0].metadata["file_name"] == "hello.txt"
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["source"] == str(txt_file.resolve())


def test_load_file_empty_txt(tmp_path: Path) -> None:
    """Empty text file returns empty list."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\t  ", encoding="utf-8")

    docs = ingestion.load_file(empty_file)
    assert docs == []


def test_load_file_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported extension raises ValueError."""
    unsupported = tmp_path / "data.docx"
    unsupported.write_text("content")

    with pytest.raises(ValueError, match="Unsupported file format"):
        ingestion.load_file(unsupported)


def test_load_file_non_existent(tmp_path: Path) -> None:
    """Non-existent file raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        ingestion.load_file(missing)


def test_load_file_pdf_mocked(tmp_path: Path) -> None:
    """Mock pypdf.PdfReader to test PDF loading logic with page metadata."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1: Introduction to AI and RAG."

    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "   "  # Empty page to be skipped

    mock_page3 = MagicMock()
    mock_page3.extract_text.return_value = "Page 3: Evaluation Metrics."

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1, mock_page2, mock_page3]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        docs = ingestion.load_file(pdf_path)

    assert len(docs) == 2
    assert docs[0].page_content == "Page 1: Introduction to AI and RAG."
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 3
    assert docs[0].metadata["file_type"] == "pdf"

    assert docs[1].page_content == "Page 3: Evaluation Metrics."
    assert docs[1].metadata["page"] == 3


def test_load_documents_directory(tmp_path: Path) -> None:
    """Test recursive loading of supported files in a folder."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "file1.txt").write_text("Content 1", encoding="utf-8")
    (tmp_path / "sub" / "file2.txt").write_text("Content 2", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("id,val\n1,2", encoding="utf-8")

    docs, failed = ingestion.load_documents(tmp_path)
    assert len(failed) == 0
    assert len(docs) == 2
    contents = {d.page_content for d in docs}
    assert contents == {"Content 1", "Content 2"}


def test_split_documents() -> None:
    """Test document splitting with chunk_size and chunk_overlap."""
    long_text = "Thành phần 1. " * 30 + "\n\n" + "Thành phần 2. " * 30
    doc = Document(
        page_content=long_text,
        metadata={"source": "test.txt", "file_name": "test.txt", "file_type": "txt"},
    )

    chunks = ingestion.split_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source"] == "test.txt"
        assert len(chunk.page_content) <= 120  # splitter character tolerance


def test_generate_chunk_id() -> None:
    """Test chunk ID is deterministic and hashes identically for identical contents."""
    doc1 = Document(
        page_content="Same content",
        metadata={"source": "/path/to/file.txt", "page": 1, "chunk_index": 0},
    )
    doc2 = Document(
        page_content="Same content",
        metadata={"source": "/path/to/file.txt", "page": 1, "chunk_index": 0},
    )
    doc3 = Document(
        page_content="Different content",
        metadata={"source": "/path/to/file.txt", "page": 1, "chunk_index": 0},
    )

    id1 = ingestion.generate_chunk_id(doc1)
    id2 = ingestion.generate_chunk_id(doc2)
    id3 = ingestion.generate_chunk_id(doc3)

    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 64  # SHA-256 hex string


def test_embed_texts_batching() -> None:
    """Test embed_texts batches calls according to EMBEDDING_BATCH_SIZE."""
    texts = [f"Text item {i}" for i in range(5)]
    mock_embedding = [0.1] * 768

    mock_item = MagicMock()
    mock_item.values = mock_embedding
    mock_response = MagicMock()
    mock_response.embeddings = [mock_item]

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_response

    # Force batch size of 2 for testing
    with patch("src.core.ingestion.get_genai_client", return_value=mock_client), \
         patch("src.core.ingestion.settings.EMBEDDING_BATCH_SIZE", 2), \
         patch("src.core.ingestion.settings.EMBEDDING_DIMENSION", 768), \
         patch("src.core.ingestion._embed_single_batch", return_value=[mock_embedding, mock_embedding]) as mock_batch:
        
        # 5 items / 2 batch size = 3 batches
        results = ingestion.embed_texts(texts)
        assert mock_batch.call_count == 3


def test_embed_dimension_mismatch() -> None:
    """Test that embedding with incorrect dimension raises ValueError."""
    mock_item = MagicMock()
    mock_item.values = [0.1] * 512  # Mismatched dimension (expected 768)
    mock_response = MagicMock()
    mock_response.embeddings = [mock_item]

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_response

    with patch("src.core.ingestion.settings.EMBEDDING_DIMENSION", 768):
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            ingestion._embed_single_batch(mock_client, ["test"], "RETRIEVAL_DOCUMENT")


def test_ingest_path_pipeline(tmp_path: Path) -> None:
    """Test full ingest_path pipeline with mocked embedding and upsert."""
    sample_file = tmp_path / "test.txt"
    sample_file.write_text("Sample paragraph for RAG core pipeline testing.", encoding="utf-8")

    dummy_vector = [0.05] * 768

    with patch("src.core.ingestion.embed_texts", return_value=[dummy_vector]) as mock_embed, \
         patch("src.services.vector_db.upsert_vectors", return_value=1) as mock_upsert:
        
        result = ingestion.ingest_path(sample_file, collection_name="test_col")

        assert result.files_processed == 1
        assert result.chunks_created == 1
        assert result.failed_files == []
        assert result.collection_name == "test_col"

        mock_embed.assert_called_once()
        mock_upsert.assert_called_once()
        args, kwargs = mock_upsert.call_args
        assert len(kwargs["ids"]) == 1
        assert kwargs["embeddings"] == [dummy_vector]
        assert kwargs["collection_name"] == "test_col"

