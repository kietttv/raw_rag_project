"""Cross-Encoder Reranker module for RAG pipeline.

Re-scores candidate chunks from Hybrid Search (Top 20-30) using BAAI/bge-reranker-v2-m3
via sentence-transformers CrossEncoder, and selects the Top 3-5 most relevant chunks
for Gemini context.
"""

from __future__ import annotations

import argparse
import math
import threading
from typing import Any

from src.config import settings
from src.models.schemas import SourceDocument
from src.utils.logger import logger

_reranker_model: Any | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> Any:
    """Return the shared CrossEncoder instance, loading it on first call (thread-safe singleton)."""
    global _reranker_model
    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                from sentence_transformers import CrossEncoder

                logger.bind(
                    model_name=settings.RERANKER_MODEL_NAME,
                    max_length=settings.RERANKER_MAX_LENGTH,
                ).info("Loading CrossEncoder reranker model")

                _reranker_model = CrossEncoder(
                    model_name=settings.RERANKER_MODEL_NAME,
                    max_length=settings.RERANKER_MAX_LENGTH,
                )
    return _reranker_model


def reset_reranker() -> None:
    """Reset the singleton reranker model instance (useful for testing or memory release)."""
    global _reranker_model
    with _reranker_lock:
        _reranker_model = None


def normalize_rerank_score(raw_score: float | int) -> float:
    """Convert raw CrossEncoder output logit to a normalized probability in [0.0, 1.0] using Sigmoid."""
    val = float(raw_score)
    clipped = max(-50.0, min(50.0, val))
    prob = 1.0 / (1.0 + math.exp(-clipped))
    return round(prob, 4)


def rerank(
    query: str,
    documents: list[SourceDocument],
    top_k: int | None = None,
) -> list[SourceDocument]:
    """Rerank candidate documents against the query using CrossEncoder model.

    Args:
        query: User search query text.
        documents: Candidate SourceDocument instances from Hybrid Search (e.g. Top 20-30).
        top_k: Number of most relevant documents to return (defaults to settings.DEFAULT_TOP_K).

    Returns:
        list[SourceDocument]: Reranked documents sorted by relevance score descending.

    Raises:
        ValueError: If query is empty or top_k <= 0.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty")

    k = top_k if top_k is not None else settings.DEFAULT_TOP_K
    if k <= 0:
        raise ValueError(f"top_k must be greater than 0, got {k}")

    if not documents:
        return []

    logger.bind(
        query=cleaned_query,
        candidates_count=len(documents),
        top_k=k,
        model=settings.RERANKER_MODEL_NAME,
    ).debug("Scoring candidate pairs with CrossEncoder")

    model = get_reranker()
    pairs = [[cleaned_query, doc.content] for doc in documents]

    raw_scores = model.predict(pairs)

    reranked_docs: list[SourceDocument] = []
    for doc, raw_score in zip(documents, raw_scores):
        score_val = normalize_rerank_score(raw_score)
        meta = dict(doc.metadata or {})

        # Preserve previous retrieval score before reranking
        if "initial_score" not in meta and doc.score is not None:
            meta["initial_score"] = doc.score

        meta["rerank_score"] = score_val

        reranked_docs.append(
            SourceDocument(
                content=doc.content,
                score=score_val,
                metadata=meta,
            )
        )

    # Sort descending by CrossEncoder relevance score
    reranked_docs.sort(key=lambda d: (d.score if d.score is not None else -1.0), reverse=True)

    final_results = reranked_docs[:k]

    logger.bind(
        query=cleaned_query,
        input_candidates=len(documents),
        returned_top_k=len(final_results),
    ).info("Completed candidate reranking")

    return final_results


def retrieve_and_rerank(
    query: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    where: dict[str, Any] | None = None,
    min_score: float | None = None,
    collection_name: str | None = None,
    use_hybrid: bool = True,
) -> list[SourceDocument]:
    """Convenience function: retrieve candidates via Hybrid Search, then rerank with CrossEncoder.

    Args:
        query: User search query.
        top_k: Final number of best chunks to keep (Top 3-5).
        candidate_k: Number of hybrid search chunks to retrieve for reranking (Top 20-30).
        where: Optional metadata filter dict.
        min_score: Optional minimum score threshold for dense search.
        collection_name: Optional target collection name.
        use_hybrid: Whether to use hybrid search (Dense + BM25 + RRF) for candidates.

    Returns:
        list[SourceDocument]: Top-K reranked documents.
    """
    from src.core.retriever import retrieve

    fetch_k = candidate_k if candidate_k is not None else settings.RERANK_CANDIDATE_K
    final_k = top_k if top_k is not None else settings.DEFAULT_TOP_K

    # 1. Hybrid Search Stage (fetch 20-30 candidates)
    candidate_docs = retrieve(
        query=query,
        top_k=fetch_k,
        where=where,
        min_score=min_score,
        collection_name=collection_name,
        use_hybrid=use_hybrid,
        use_reranker=False,  # Avoid double reranking
    )

    # 2. CrossEncoder Rerank Stage (keep Top 3-5)
    return rerank(query=query, documents=candidate_docs, top_k=final_k)


def main() -> None:
    """CLI entrypoint for standalone reranking and pipeline verification."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Retrieve candidates via Hybrid Search and rerank with BAAI/bge-reranker-v2-m3."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.DEFAULT_TOP_K,
        help=f"Number of top chunks to keep after reranking (default: {settings.DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=settings.RERANK_CANDIDATE_K,
        help=f"Number of candidates to fetch from Hybrid Search (default: {settings.RERANK_CANDIDATE_K}).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name.",
    )

    args = parser.parse_args()

    try:
        results = retrieve_and_rerank(
            query=args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            collection_name=args.collection,
        )

        print("\n" + "=" * 65)
        print("CROSS-ENCODER RERANKING REPORT [BAAI/bge-reranker-v2-m3]")
        print("=" * 65)
        print(f"Query:               {args.query}")
        print(f"Candidates fetched:  {args.candidate_k}")
        print(f"Top-K kept:          {len(results)}")
        print("=" * 65)

        if not results:
            print("\nNo matching chunks found.\n")
            return

        for idx, doc in enumerate(results, start=1):
            score_str = f"{doc.score:.4f}" if doc.score is not None else "N/A"
            source = doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
            page = doc.metadata.get("page")
            page_str = f" | Page: {page}" if page is not None else ""
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            initial = doc.metadata.get("initial_score")
            init_str = f" (Initial: {initial:.4f})" if initial is not None else ""

            print(f"\n[#{idx}] Rerank Score: {score_str}{init_str} | Source: {source}{page_str}")
            print(f"      Chunk ID: {chunk_id}")
            snippet = doc.content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            print(f"      Content:  {snippet}")

        print("\n" + "=" * 65 + "\n")

    except Exception as exc:
        logger.bind(query=args.query).exception(f"Reranking failed: {exc}")
        print(f"\n[ERROR] Reranking failed: {exc}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

