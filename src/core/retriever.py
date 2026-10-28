"""Hybrid search retriever combining Dense Vector and Sparse BM25 via RRF.

Receives user query, embeds with task_type="RETRIEVAL_QUERY" using Gemini embedding,
runs BM25 keyword matching for precise token/code retrieval, and combines results
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import argparse
import math
import re
import threading
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import settings
from src.core.ingestion import embed_texts
from src.models.schemas import SourceDocument
from src.services import vector_db
from src.utils.logger import logger


class LuceneBM25(BM25Okapi):
    """BM25Okapi with standard Lucene/Elasticsearch positive IDF formula.

    Standard rank_bm25 Okapi calculates log((N - n + 0.5)/(n + 0.5)), which drops to 0 or negative
    when a term appears in >= N/2 documents or on small corpora (N <= 2).
    Adding 1 inside the log ensures IDF is strictly positive for all matching terms.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class _BM25CacheEntry:
    """In-memory cache for a collection's BM25 index."""

    def __init__(
        self,
        doc_ids: list[str],
        documents: list[dict[str, Any]],
        bm25: LuceneBM25,
    ) -> None:
        self.doc_ids = doc_ids
        self.documents = documents
        self.bm25 = bm25


_bm25_cache: dict[str, _BM25CacheEntry] = {}
_cache_lock = threading.Lock()


def reset_bm25_cache() -> None:
    """Clear in-memory BM25 index cache (useful for testing or cache eviction)."""
    global _bm25_cache
    with _cache_lock:
        _bm25_cache.clear()


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase words, preserving alphanumeric codes and compound identifiers.

    Supports all Vietnamese Unicode characters (including 'đ', 'Đ'), punctuation-joined
    identifiers (e.g. 'NV-001', 'ISO_9001', 'HĐ-2024-999'), and splits compounds
    into sub-tokens for flexible keyword matching.

    Args:
        text: Input text string.

    Returns:
        list[str]: Normalized token list.
    """
    if not text:
        return []

    # \w matches any Unicode alphanumeric word character in Python 3
    raw_tokens = re.findall(r"\w+(?:[-_]\w+)*", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        # If token is a compound identifier (e.g. 'nv-001'), also add individual parts
        if "-" in token or "_" in token:
            parts = re.split(r"[-_]", token)
            for part in parts:
                if part and part != token:
                    tokens.append(part)
    return tokens


def bm25_search(
    query: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k documents using Sparse BM25 keyword matching with in-memory caching.

    Args:
        query: User search query.
        top_k: Number of candidate documents to retrieve.
        where: Optional metadata filter dict.
        collection_name: Target ChromaDB collection name.

    Returns:
        list[dict[str, Any]]: Ranked results with keys: id, content, metadata, score.
    """
    target_collection = collection_name or settings.CHROMA_COLLECTION_NAME
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    try:
        all_docs = vector_db.get_all_documents(where=where, collection_name=target_collection)
    except Exception as exc:
        logger.bind(collection=target_collection, error=str(exc)).warning(
            "Failed to fetch collection documents for BM25; skipping BM25 search"
        )
        return []

    if not all_docs:
        return []

    current_ids = [d["id"] for d in all_docs]
    bm25_instance: LuceneBM25 | None = None

    # Use in-memory cache when no metadata filter is specified
    if not where:
        with _cache_lock:
            cached_entry = _bm25_cache.get(target_collection)
            if cached_entry and cached_entry.doc_ids == current_ids:
                bm25_instance = cached_entry.bm25
                all_docs = cached_entry.documents
            else:
                tokenized_corpus = [tokenize_text(d.get("content", "")) for d in all_docs]
                bm25_instance = LuceneBM25(tokenized_corpus)
                _bm25_cache[target_collection] = _BM25CacheEntry(
                    doc_ids=current_ids,
                    documents=all_docs,
                    bm25=bm25_instance,
                )
    else:
        # Dynamic BM25 index on filtered subset
        tokenized_corpus = [tokenize_text(d.get("content", "")) for d in all_docs]
        bm25_instance = LuceneBM25(tokenized_corpus)

    scores = bm25_instance.get_scores(query_tokens)

    # Collect documents with strictly positive BM25 score
    scored_docs: list[dict[str, Any]] = []
    for doc, score in zip(all_docs, scores):
        if score > 0.0:
            scored_docs.append(
                {
                    "id": doc["id"],
                    "content": doc.get("content", ""),
                    "metadata": dict(doc.get("metadata") or {}),
                    "score": round(float(score), 4),
                }
            )

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    results = scored_docs[:top_k]

    logger.bind(
        collection=target_collection,
        query=query,
        matched=len(results),
        top_k=top_k,
    ).debug("Completed BM25 sparse search")

    return results


def reciprocal_rank_fusion(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine dense vector and sparse BM25 search results using Reciprocal Rank Fusion (RRF).

    Formula: RRF_Score(d) = sum(1 / (rrf_k + rank_i(d))) for rank_i in {dense, bm25} (1-indexed).

    Args:
        dense_hits: Ranked hits from dense vector search.
        bm25_hits: Ranked hits from sparse BM25 search.
        rrf_k: Smoothing constant parameter (standard default: 60).

    Returns:
        list[dict[str, Any]]: Unified list of hits ranked by combined RRF score.
    """
    fused_scores: dict[str, float] = {}
    doc_registry: dict[str, dict[str, Any]] = {}
    dense_score_map: dict[str, float] = {}
    bm25_score_map: dict[str, float] = {}

    # 1. Process dense hits
    for rank, hit in enumerate(dense_hits, start=1):
        item_id = hit["id"]
        rrf_score = 1.0 / (rrf_k + rank)
        fused_scores[item_id] = fused_scores.get(item_id, 0.0) + rrf_score
        dense_score_map[item_id] = hit.get("score", 0.0)
        if item_id not in doc_registry:
            doc_registry[item_id] = hit

    # 2. Process BM25 hits
    for rank, hit in enumerate(bm25_hits, start=1):
        item_id = hit["id"]
        rrf_score = 1.0 / (rrf_k + rank)
        fused_scores[item_id] = fused_scores.get(item_id, 0.0) + rrf_score
        bm25_score_map[item_id] = hit.get("score", 0.0)
        if item_id not in doc_registry:
            doc_registry[item_id] = hit

    # 3. Sort by total RRF score descending
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for item_id in sorted_ids:
        raw_hit = doc_registry[item_id]
        meta = dict(raw_hit.get("metadata") or {})
        if "chunk_id" not in meta:
            meta["chunk_id"] = item_id

        meta["rrf_score"] = round(fused_scores[item_id], 6)
        if item_id in dense_score_map:
            meta["dense_score"] = dense_score_map[item_id]
        if item_id in bm25_score_map:
            meta["bm25_score"] = bm25_score_map[item_id]

        fused_results.append(
            {
                "id": item_id,
                "content": raw_hit.get("content", ""),
                "metadata": meta,
                "score": round(fused_scores[item_id], 4),
            }
        )

    return fused_results


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
    use_hybrid: bool = True,
    rrf_k: int = 60,
    candidate_k: int | None = None,
    use_reranker: bool | None = None,
) -> list[SourceDocument]:
    """Retrieve top-k most relevant document chunks for a query using Hybrid Search and optional Reranking.

    Pipeline:
    1. Candidate retrieval via Hybrid Search (Dense Vector + BM25 + RRF).
       If use_reranker is enabled, retrieves Top 20-30 candidates (settings.RERANK_CANDIDATE_K).
    2. CrossEncoder Reranking via BAAI/bge-reranker-v2-m3 (keeps Top 3-5).

    Args:
        query: Query string to search for.
        top_k: Number of chunks to retrieve (defaults to settings.DEFAULT_TOP_K).
        where: Optional metadata filter dict for ChromaDB.
        min_score: Optional minimum similarity score threshold for dense vector search [-1.0, 1.0].
        collection_name: Optional target collection name in ChromaDB.
        use_hybrid: Whether to perform hybrid search (Dense + BM25 via RRF). Defaults to True.
        rrf_k: RRF constant parameter (defaults to 60).
        candidate_k: Optional candidate pool size before fusion/reranking.
        use_reranker: Whether to run CrossEncoder reranking. Defaults to settings.USE_RERANKER.

    Returns:
        list[SourceDocument]: Ranked list of retrieved document chunks.

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
    should_rerank = settings.USE_RERANKER if use_reranker is None else use_reranker

    if candidate_k is not None:
        fetch_k = candidate_k
    elif should_rerank:
        fetch_k = max(k, settings.RERANK_CANDIDATE_K)
    else:
        fetch_k = k

    logger.bind(
        query=cleaned_query,
        top_k=k,
        fetch_k=fetch_k,
        collection=target_collection,
        use_hybrid=use_hybrid,
        use_reranker=should_rerank,
    ).debug("Starting document retrieval")

    # 1. Dense Vector Search
    query_vector = embed_query(cleaned_query)
    dense_hits = vector_db.query_vectors(
        query_embedding=query_vector,
        top_k=fetch_k,
        where=where,
        collection_name=target_collection,
    )

    if min_score is not None:
        dense_hits = [h for h in dense_hits if (h.get("score") is not None and h["score"] >= min_score)]

    # 2. Sparse BM25 Search & RRF Fusion (if enabled)
    final_hits: list[dict[str, Any]] = []
    if use_hybrid:
        bm25_hits = bm25_search(
            query=cleaned_query,
            top_k=fetch_k,
            where=where,
            collection_name=target_collection,
        )

        if bm25_hits:
            fused_hits = reciprocal_rank_fusion(
                dense_hits=dense_hits,
                bm25_hits=bm25_hits,
                rrf_k=rrf_k,
            )
            final_hits = fused_hits[:fetch_k]
        else:
            # Fallback to dense hits if no BM25 matches were found
            final_hits = dense_hits[:fetch_k]
    else:
        final_hits = dense_hits[:fetch_k]

    # 3. Format candidates as SourceDocument
    candidate_documents: list[SourceDocument] = []
    for h in final_hits:
        meta = dict(h.get("metadata") or {})
        if "id" in h and "chunk_id" not in meta:
            meta["chunk_id"] = h["id"]

        candidate_documents.append(
            SourceDocument(
                content=h.get("content", ""),
                score=h.get("score"),
                metadata=meta,
            )
        )

    # 4. Optional Cross-Encoder Reranking
    if should_rerank and candidate_documents:
        from src.core.reranker import rerank

        documents = rerank(
            query=cleaned_query,
            documents=candidate_documents,
            top_k=k,
        )
    else:
        documents = candidate_documents[:k]

    logger.bind(
        query=cleaned_query,
        top_k=k,
        candidates=len(candidate_documents),
        returned=len(documents),
        collection=target_collection,
        reranked=should_rerank,
    ).info("Retrieved relevant documents")

    return documents


def main() -> None:
    """CLI entrypoint for hybrid search retrieval."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from ChromaDB for a given query using Hybrid Search (BM25 + Dense + RRF)."
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
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable BM25 sparse search and run pure dense vector search.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable CrossEncoder reranking stage.",
    )

    args = parser.parse_args()

    try:
        results = retrieve(
            query=args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            collection_name=args.collection,
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_rerank,
        )

        mode_parts = []
        mode_parts.append("DENSE" if args.no_hybrid else "HYBRID (DENSE+BM25+RRF)")
        if not args.no_rerank:
            mode_parts.append("RERANKED")
        mode_label = " + ".join(mode_parts)
        print("\n" + "=" * 60)
        print(f"RETRIEVAL SEARCH RESULTS [{mode_label}]")
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

            extra_scores = []
            if "rrf_score" in doc.metadata:
                extra_scores.append(f"RRF: {doc.metadata['rrf_score']:.6f}")
            if "dense_score" in doc.metadata:
                extra_scores.append(f"Dense: {doc.metadata['dense_score']:.4f}")
            if "bm25_score" in doc.metadata:
                extra_scores.append(f"BM25: {doc.metadata['bm25_score']:.4f}")

            extra_str = f" ({', '.join(extra_scores)})" if extra_scores else ""

            print(f"\n[#{idx}] Score: {score_str}{extra_str} | Source: {source}{page_str}")
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
