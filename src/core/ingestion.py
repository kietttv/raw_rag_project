"""Document ingestion pipeline for Core RAG MVP.

Handles loading (PDF, TXT), chunking (RecursiveCharacterTextSplitter),
embedding generation via Gemini (gemini-embedding-001 with output_dimensionality),
and upserting into ChromaDB via vector_db service.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.services import vector_db
from src.utils.logger import logger

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Global singleton client for Gemini API
_genai_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    """Return a cached Google GenAI client initialized with settings.GEMINI_API_KEY."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            api_key=settings.GEMINI_API_KEY.get_secret_value()
        )
    return _genai_client


def reset_genai_client() -> None:
    """Reset the singleton Gemini client (used in testing)."""
    global _genai_client
    _genai_client = None


@dataclass
class IngestResult:
    """Summary of ingestion operation."""

    files_processed: int = 0
    chunks_created: int = 0
    failed_files: list[str] = field(default_factory=list)
    collection_name: str = ""


# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------


def _load_txt(path: Path) -> list[Document]:
    """Read a plain text file trying UTF-8, then fallback encodings."""
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    text = ""
    used_enc = ""
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue

    if not used_enc:
        raise ValueError(f"Failed to decode text file with encodings {encodings}: {path}")

    cleaned = text.strip()
    if not cleaned:
        logger.bind(path=str(path)).warning("Text file is empty; skipping")
        return []

    return [
        Document(
            page_content=cleaned,
            metadata={
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": "txt",
            },
        )
    ]


def _load_pdf(path: Path) -> list[Document]:
    """Read PDF pages using pypdf and extract non-empty text per page."""
    docs: list[Document] = []
    reader = pypdf.PdfReader(str(path))
    total_pages = len(reader.pages)

    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.bind(path=str(path), page=idx).warning(
                f"Error extracting text from PDF page: {e}"
            )
            page_text = ""

        cleaned = page_text.strip()
        if not cleaned:
            logger.bind(path=str(path), page=idx).debug(
                "PDF page has no extractable text (blank or scanned image); skipping"
            )
            continue

        docs.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source": str(path.resolve()),
                    "file_name": path.name,
                    "file_type": "pdf",
                    "page": idx,
                    "total_pages": total_pages,
                },
            )
        )

    if not docs:
        logger.bind(path=str(path)).warning(
            "PDF file contains no extractable text across all pages"
        )

    return docs


def load_file(path: Path | str) -> list[Document]:
    """Load a single file (.pdf or .txt) into a list of Document objects.

    Args:
        path: Path to the target file.

    Returns:
        list[Document]: Extracted documents with metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file format is not supported.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".txt":
        return _load_txt(file_path)
    if ext == ".pdf":
        return _load_pdf(file_path)

    return []


def load_documents(path: Path | str) -> tuple[list[Document], list[str]]:
    """Load all supported documents from a file or directory recursively.

    Args:
        path: File or directory path.

    Returns:
        tuple[list[Document], list[str]]: (all_extracted_docs, failed_file_paths)
    """
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    candidate_files: list[Path] = []
    if target.is_file():
        candidate_files = [target]
    else:
        for p in sorted(target.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                candidate_files.append(p)

    all_docs: list[Document] = []
    failed_files: list[str] = []

    for file_path in candidate_files:
        try:
            docs = load_file(file_path)
            all_docs.extend(docs)
            logger.bind(path=str(file_path), count=len(docs)).debug("Loaded document file")
        except Exception as e:
            logger.bind(path=str(file_path)).error(f"Failed to load file: {e}")
            failed_files.append(str(file_path))

    return all_docs, failed_files


# ---------------------------------------------------------------------------
# 2. Text Chunking
# ---------------------------------------------------------------------------


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: Raw documents to split.
        chunk_size: Size of chunk (defaults to settings.CHUNK_SIZE).
        chunk_overlap: Overlap between consecutive chunks (defaults to settings.CHUNK_OVERLAP).

    Returns:
        list[Document]: List of chunked Document instances with chunk_index in metadata.
    """
    if not documents:
        return []

    c_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Assign sequential chunk_index per original source
    counts_per_source: dict[str, int] = {}
    for chunk in chunks:
        source_key = f"{chunk.metadata.get('source', '')}:{chunk.metadata.get('page', 0)}"
        idx = counts_per_source.get(source_key, 0)
        chunk.metadata["chunk_index"] = idx
        counts_per_source[source_key] = idx + 1

    logger.bind(
        input_docs=len(documents),
        output_chunks=len(chunks),
        chunk_size=c_size,
        chunk_overlap=c_overlap,
    ).info("Split documents into chunks")

    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding Generation
# ---------------------------------------------------------------------------


def _is_retryable_exception(exc: BaseException) -> bool:
    """Return True if exception looks like a transient rate-limit or network issue."""
    err_str = str(exc).lower()
    return (
        "429" in err_str
        or "resource_exhausted" in err_str
        or "quota" in err_str
        or "503" in err_str
        or "unavailable" in err_str
        or "timeout" in err_str
        or "connection" in err_str
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable_exception),
    reraise=True,
)
def _embed_single_batch(
    client: genai.Client,
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    """Embed a single batch of texts via Google GenAI SDK with retry logic."""
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=settings.EMBEDDING_DIMENSION,
    )

    response = client.models.embed_content(
        model=settings.EMBEDDING_MODEL_NAME,
        contents=texts,
        config=config,
    )

    embeddings: list[list[float]] = []
    for item in response.embeddings:
        vec = list(item.values)
        if len(vec) != settings.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIMENSION}, got {len(vec)}"
            )
        embeddings.append(vec)

    return embeddings


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Generate embeddings for a list of texts in batches.

    Args:
        texts: Raw text strings to embed.
        task_type: Embedding task type (e.g. 'RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY').

    Returns:
        list[list[float]]: Corresponding embedding vectors of length settings.EMBEDDING_DIMENSION.
    """
    if not texts:
        return []

    client = get_genai_client()
    batch_size = settings.EMBEDDING_BATCH_SIZE
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.bind(
            batch_start=i,
            batch_count=len(batch),
            total=len(texts),
            model=settings.EMBEDDING_MODEL_NAME,
        ).debug("Embedding text batch")

        batch_embeddings = _embed_single_batch(client, batch, task_type=task_type)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ---------------------------------------------------------------------------
# 4. Storage & Orchestration
# ---------------------------------------------------------------------------


def generate_chunk_id(chunk: Document) -> str:
    """Generate a deterministic SHA-256 ID for a chunk to ensure idempotent upsert."""
    meta = chunk.metadata
    source = meta.get("source", "")
    page = meta.get("page", "")
    chunk_index = meta.get("chunk_index", 0)
    raw = f"{source}:{page}:{chunk_index}:{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ingest_path(
    path: Path | str,
    collection_name: str | None = None,
) -> IngestResult:
    """End-to-end ingestion pipeline: load -> split -> embed -> upsert.

    Args:
        path: Path to a file or directory containing .pdf or .txt files.
        collection_name: Optional target collection name in ChromaDB.

    Returns:
        IngestResult: Summary containing files processed, chunks created, and failed files.
    """
    target = Path(path).resolve()
    target_collection = collection_name or settings.CHROMA_COLLECTION_NAME

    logger.bind(path=str(target), collection=target_collection).info(
        "Starting document ingestion"
    )

    # 1. Load documents
    docs, failed_files = load_documents(target)
    if not docs:
        logger.bind(path=str(target)).warning("No valid documents found to ingest")
        return IngestResult(
            files_processed=0,
            chunks_created=0,
            failed_files=failed_files,
            collection_name=target_collection,
        )

    # 2. Split into chunks
    chunks = split_documents(docs)
    if not chunks:
        logger.bind(path=str(target)).warning("Document splitting produced 0 chunks")
        return IngestResult(
            files_processed=len({d.metadata.get("source") for d in docs}),
            chunks_created=0,
            failed_files=failed_files,
            collection_name=target_collection,
        )

    # 3. Generate IDs and extract text
    ids = [generate_chunk_id(c) for c in chunks]
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # 4. Embed chunks
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    # 5. Upsert to Vector DB
    upserted_count = vector_db.upsert_vectors(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        collection_name=target_collection,
    )

    # Invalidate Redis query cache and BM25 index for updated collection
    try:
        from src.services import cache_service
        cache_service.invalidate_collection_cache(target_collection)
        from src.core.retriever import reset_bm25_cache
        reset_bm25_cache()
    except Exception as cache_err:
        logger.bind(collection=target_collection, error=str(cache_err)).warning(
            "Failed invalidating caches after document ingestion"
        )

    distinct_files = len({c.metadata.get("source") for c in chunks})
    result = IngestResult(
        files_processed=distinct_files,
        chunks_created=upserted_count,
        failed_files=failed_files,
        collection_name=target_collection,
    )

    logger.bind(
        files=result.files_processed,
        chunks=result.chunks_created,
        failed=len(result.failed_files),
        collection=result.collection_name,
    ).info("Document ingestion completed successfully")

    return result


# ---------------------------------------------------------------------------
# 5. CLI Interface
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for running ingestion on a file or folder."""
    parser = argparse.ArgumentParser(
        description="Ingest documents (.pdf, .txt) into ChromaDB using Gemini embeddings."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to file or directory to ingest.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (defaults to CHROMA_COLLECTION_NAME in .env).",
    )

    args = parser.parse_args()

    try:
        result = ingest_path(args.path, collection_name=args.collection)
        print("\n" + "=" * 50)
        print("DOCUMENT INGESTION REPORT")
        print("=" * 50)
        print(f"Target path:       {args.path}")
        print(f"Collection:        {result.collection_name}")
        print(f"Files processed:   {result.files_processed}")
        print(f"Chunks created:    {result.chunks_created}")
        if result.failed_files:
            print(f"Failed files ({len(result.failed_files)}):")
            for f in result.failed_files:
                print(f"  - {f}")
        print("=" * 50 + "\n")
    except Exception as exc:
        logger.bind(path=args.path).exception(f"Ingestion failed: {exc}")
        print(f"\n[ERROR] Ingestion failed: {exc}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

