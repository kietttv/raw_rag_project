"""Pure vector-search retriever for Core RAG pipeline.

Receives user query, embeds with task_type="RETRIEVAL_QUERY" using Gemini embedding,
and retrieves Top-K most similar chunks from ChromaDB.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.config import settings
from src.core.ingestion import embed_texts
from src.models.schemas import SourceDocument
from src.services import vector_db
from src.utils.logger import logger


def embed_query(query: str) -> list[float]:
    """Embed a search query with task_type RETRIEVAL_QUERY.

    Args:
        query: User search query string.

    Returns:
        list[float]: Embedding vector with dimension matching settings.EMBEDDING_DIMENSION.

    Raises:
        RuntimeError: If embedding generation returns no vectors.
    """
    embeddings = embed_texts([query], task_type="RETRIEVAL_QUERY")
    if not embeddings or len(embeddings[0]) == 0:
        raise RuntimeError("Failed to generate embedding for query")
    return embeddings[0]


def retrieve(
    query: str,
    top_k: int | None = None,
    where: dict[str, Any] | None = None,
    min_score: float | None = None,
    collection_name: str | None = None,
) -> list[SourceDocument]:
    """Retrieve top-k most similar document chunks for a query using vector search.

    Args:
        query: Query string to search for.
        top_k: Number of chunks to retrieve (defaults to settings.DEFAULT_TOP_K).
        where: Optional metadata filter dict for ChromaDB.
        min_score: Optional minimum similarity score threshold (in range [-1.0, 1.0]).
        collection_name: Optional target collection name in ChromaDB.

    Returns:
        list[SourceDocument]: Ranked list of retrieved document chunks with similarity scores.

    Raises:
        ValueError: If query is empty or top_k is outside valid range [1, settings.MAX_TOP_K].
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty")

    k = top_k if top_k is not None else settings.DEFAULT_TOP_K
    if not (1 <= k <= settings.MAX_TOP_K):
        raise ValueError(
            f"top_k must be between 1 and {settings.MAX_TOP_K}, got {k}"
        )

    target_collection = collection_name or settings.CHROMA_COLLECTION_NAME

    logger.bind(
        query=cleaned_query,
        top_k=k,
        collection=target_collection,
    ).debug("Generating query embedding and querying vector DB")

    query_vector = embed_query(cleaned_query)

    hits = vector_db.query_vectors(
        query_embedding=query_vector,
        top_k=k,
        where=where,
        collection_name=target_collection,
    )

    if min_score is not None:
        hits = [h for h in hits if (h.get("score") is not None and h["score"] >= min_score)]

    documents: list[SourceDocument] = []
    for h in hits:
        meta = dict(h.get("metadata") or {})
        if "id" in h and "chunk_id" not in meta:
            meta["chunk_id"] = h["id"]

        documents.append(
            SourceDocument(
                content=h.get("content", ""),
                score=h.get("score"),
                metadata=meta,
            )
        )

    logger.bind(
        query=cleaned_query,
        top_k=k,
        returned=len(documents),
        collection=target_collection,
    ).info("Retrieved relevant documents from vector store")

    return documents


def main() -> None:
    """CLI entrypoint for vector search retrieval."""
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from ChromaDB for a given query."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Number of chunks to retrieve (1 to {settings.MAX_TOP_K}, default: {settings.DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum similarity score threshold.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (defaults to CHROMA_COLLECTION_NAME in .env).",
    )

    args = parser.parse_args()

    try:
        results = retrieve(
            query=args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            collection_name=args.collection,
        )

        print("\n" + "=" * 60)
        print("RETRIEVAL SEARCH RESULTS")
        print("=" * 60)
        print(f"Query:        {args.query}")
        print(f"Collection:   {args.collection or settings.CHROMA_COLLECTION_NAME}")
        print(f"Found:        {len(results)} chunks")
        print("=" * 60)

        if not results:
            print("\nNo matching chunks found.\n")
            return

        for idx, doc in enumerate(results, start=1):
            score_str = f"{doc.score:.4f}" if doc.score is not None else "N/A"
            source = doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
            page = doc.metadata.get("page")
            page_str = f" | Page: {page}" if page is not None else ""
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            print(f"\n[#{idx}] Score: {score_str} | Source: {source}{page_str}")
            print(f"      Chunk ID: {chunk_id}")
            # Snippet of content (first 250 characters)
            snippet = doc.content.strip().replace("\n", " ")
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            print(f"      Content:  {snippet}")

        print("\n" + "=" * 60 + "\n")

    except Exception as exc:
        logger.bind(query=args.query).exception(f"Retrieval failed: {exc}")
        print(f"\n[ERROR] Retrieval failed: {exc}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

