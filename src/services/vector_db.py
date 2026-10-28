"""ChromaDB client (singleton) and vector operations."""

import os
import threading
from typing import Any
from urllib.parse import urlparse

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.config import settings
from src.utils.logger import logger

# Ensure local loopback connections bypass system proxies (protecting against httpx socks proxy issues)
_no_proxy = os.environ.get("NO_PROXY", "")
if "localhost" not in _no_proxy:
    os.environ["NO_PROXY"] = f"{_no_proxy},localhost,127.0.0.1".strip(",")

_client: ClientAPI | None = None
_lock = threading.Lock()


def get_chroma_client() -> ClientAPI:
    """Return the shared ChromaDB client, creating it on first call (thread-safe singleton)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:  # double-checked locking
                _client = _create_client(settings.VECTOR_DB_URL)
    return _client


def _create_client(url: str) -> ClientAPI:
    """Parse HTTP(S) URL and instantiate ChromaDB HttpClient with connection check."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"VECTOR_DB_URL must start with http:// or https://, got: {url!r}"
        )

    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    ssl = parsed.scheme == "https"

    try:
        client = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=ssl,
        )
        client.heartbeat()  # fail-fast check to ensure server is reachable
        logger.bind(url=url, host=host, port=port).info("Connected to ChromaDB server")
        return client
    except Exception as e:
        logger.bind(url=url).error(f"Failed to connect to ChromaDB server: {e}")
        raise


def get_collection(name: str | None = None) -> Collection:
    """Get or create a ChromaDB collection configured for cosine similarity and external embeddings."""
    collection_name = name or settings.CHROMA_COLLECTION_NAME
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,  # Embeddings are precomputed externally via Gemini
        metadata={"hnsw:space": "cosine"},
    )


def upsert_vectors(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    collection_name: str | None = None,
) -> int:
    """Insert or update vectors in the collection.

    Args:
        ids: Unique record identifiers.
        embeddings: Precomputed embedding vectors.
        documents: Raw text chunks associated with vectors.
        metadatas: Optional metadata dictionaries per record.
        collection_name: Target collection name (defaults to configured collection).

    Returns:
        int: Number of records upserted.
    """
    total = len(ids)
    if not (total == len(embeddings) == len(documents)):
        raise ValueError(
            f"Mismatched input lengths: ids={total}, embeddings={len(embeddings)}, documents={len(documents)}"
        )
    if metadatas is not None and len(metadatas) != total:
        raise ValueError(
            f"Mismatched metadatas length: expected {total}, got {len(metadatas)}"
        )
    if total == 0:
        return 0

    # Chroma does not accept empty dictionaries in metadata
    cleaned_metadatas: list[dict[str, Any] | None] | None = None
    if metadatas is not None:
        cleaned_metadatas = [m if (m and len(m) > 0) else None for m in metadatas]

    collection = get_collection(collection_name)
    client = get_chroma_client()
    try:
        max_batch_size = client.get_max_batch_size()
    except Exception:
        max_batch_size = 1000

    for i in range(0, total, max_batch_size):
        batch_ids = ids[i : i + max_batch_size]
        batch_embeddings = embeddings[i : i + max_batch_size]
        batch_docs = documents[i : i + max_batch_size]
        batch_metadatas = (
            cleaned_metadatas[i : i + max_batch_size]
            if cleaned_metadatas is not None
            else None
        )

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metadatas,
        )

    logger.bind(
        collection=collection_name or settings.CHROMA_COLLECTION_NAME,
        count=total,
    ).info("Successfully upserted vectors into ChromaDB")
    return total


def query_vectors(
    query_embedding: list[float],
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Query the collection using a precomputed query embedding.

    Args:
        query_embedding: Precomputed vector of the search query.
        top_k: Number of most similar items to return.
        where: Optional metadata filter dictionary.
        collection_name: Target collection name.

    Returns:
        list[dict[str, Any]]: Ranked results with keys:
            id, content, metadata, distance, score.
    """
    collection = get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0] if results.get("ids") else []
    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    formatted_results: list[dict[str, Any]] = []
    for i, item_id in enumerate(ids):
        dist = distances[i] if i < len(distances) and distances[i] is not None else 0.0
        # For cosine space: cosine similarity = 1 - cosine distance
        score = round(1.0 - dist, 4)
        formatted_results.append(
            {
                "id": item_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if (i < len(metas) and metas[i]) else {},
                "distance": dist,
                "score": score,
            }
        )

    logger.bind(
        collection=collection_name or settings.CHROMA_COLLECTION_NAME,
        top_k=top_k,
        returned=len(formatted_results),
    ).debug("Queried vectors from ChromaDB")

    return formatted_results


def get_all_documents(
    where: dict[str, Any] | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch all documents and metadata from a ChromaDB collection for BM25 indexing.

    Args:
        where: Optional metadata filter dictionary.
        collection_name: Target collection name.

    Returns:
        list[dict[str, Any]]: List of dicts with keys: id, content, metadata.
    """
    collection = get_collection(collection_name)
    kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where

    results = collection.get(**kwargs)

    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    items: list[dict[str, Any]] = []
    for i, item_id in enumerate(ids):
        items.append(
            {
                "id": item_id,
                "content": docs[i] if i < len(docs) and docs[i] else "",
                "metadata": metas[i] if (i < len(metas) and metas[i]) else {},
            }
        )

    logger.bind(
        collection=collection_name or settings.CHROMA_COLLECTION_NAME,
        count=len(items),
    ).debug("Fetched collection documents for BM25 indexing")

    return items


def reset_client() -> None:
    """Reset the singleton client instance (useful for unit tests)."""
    global _client
    with _lock:
        _client = None

