"""LLM Generator module for RAG core pipeline.

Formats retrieved contexts into a standard prompt template, calls Gemini
(gemini-3.5-flash) to generate grounded responses, and extracts source metadata.
"""

from __future__ import annotations

import argparse
from typing import Any

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.core.ingestion import _is_retryable_exception, get_genai_client
from src.core.retriever import retrieve
from src.models.schemas import ChatResponse, SourceDocument
from src.utils.logger import logger

RAG_PROMPT_TEMPLATE = """Bạn là trợ lý AI trả lời câu hỏi dựa trên các tài liệu được cung cấp dưới đây.

NGỮ CẢNH (Context):
{context}

CÂU HỎI (Question):
{question}

HƯỚNG DẪN TRẢ LỜI:
1. Chỉ sử dụng thông tin có trong phần NGỮ CẢNH để trả lời câu hỏi.
2. Trả lời chính xác, rõ ràng, trung thực và có tính dẫn chứng theo tài liệu.
3. Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi, hãy nói rõ: "Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này." Tuyệt đối không tự suy diễn hoặc bịa đặt thông tin.

CÂU TRẢ LỜI:"""


def format_context(sources: list[SourceDocument]) -> str:
    """Format a list of SourceDocument objects into a numbered context block for prompt.

    Args:
        sources: Retrieved documents to format as context.

    Returns:
        str: Clean formatted context string.
    """
    if not sources:
        return "Không có tài liệu tham khảo nào."

    blocks: list[str] = []
    for idx, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        file_name = meta.get("file_name") or meta.get("source") or "Tài liệu"
        page = meta.get("page")
        page_str = f", Trang {page}" if page is not None else ""
        header = f"[{idx}] Nguồn: {file_name}{page_str}"
        content = doc.content.strip()
        blocks.append(f"{header}\n{content}")

    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    sources: list[SourceDocument],
    template: str = RAG_PROMPT_TEMPLATE,
) -> str:
    """Combine context and user query into formatted prompt string.

    Args:
        query: User question.
        sources: Context documents.
        template: Prompt template containing {context} and {question}.

    Returns:
        str: Ready-to-send prompt.
    """
    context_text = format_context(sources)
    return template.format(context=context_text, question=query.strip())


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable_exception),
    reraise=True,
)
def _generate_with_retry(client: genai.Client, prompt: str) -> str:
    """Call Gemini generate_content with tenacity retry logic on transient errors."""
    config = types.GenerateContentConfig(
        temperature=settings.GENERATION_TEMPERATURE,
    )
    response = client.models.generate_content(
        model=settings.GENERATION_MODEL_NAME,
        contents=prompt,
        config=config,
    )
    if not response.text:
        return ""
    return response.text.strip()


def generate_answer(
    query: str,
    sources: list[SourceDocument],
    template: str = RAG_PROMPT_TEMPLATE,
) -> ChatResponse:
    """Generate answer from user query and list of retrieved source documents.

    Args:
        query: User question.
        sources: Context documents retrieved for this question.
        template: Optional custom prompt template.

    Returns:
        ChatResponse: Generated answer text with source document citations.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty")

    client = get_genai_client()
    prompt = build_prompt(cleaned_query, sources, template=template)

    logger.bind(
        query=cleaned_query,
        sources_count=len(sources),
        model=settings.GENERATION_MODEL_NAME,
        temperature=settings.GENERATION_TEMPERATURE,
    ).debug("Sending prompt to Gemini for answer generation")

    answer = _generate_with_retry(client, prompt)

    logger.bind(
        query=cleaned_query,
        sources_count=len(sources),
        answer_length=len(answer),
    ).info("Successfully generated RAG answer")

    return ChatResponse(
        answer=answer,
        sources=sources,
    )


def generate_rag_response(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
    collection_name: str | None = None,
) -> ChatResponse:
    """Execute full RAG generation pipeline: retrieve sources -> generate answer.

    Args:
        query: User question.
        top_k: Number of chunks to retrieve (defaults to settings.DEFAULT_TOP_K).
        min_score: Optional similarity score threshold.
        collection_name: Optional target collection name in ChromaDB.

    Returns:
        ChatResponse: Generated answer accompanied by source chunks.
    """
    sources = retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
        collection_name=collection_name,
    )
    return generate_answer(query=query, sources=sources)


def main() -> None:
    """CLI entrypoint for asking questions and generating RAG answers."""
    parser = argparse.ArgumentParser(
        description="Generate grounded answers using RAG pipeline (Gemini + ChromaDB)."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Question to ask the RAG system.",
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
        response = generate_rag_response(
            query=args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            collection_name=args.collection,
        )

        print("\n" + "=" * 65)
        print("RAG ANSWER GENERATION REPORT")
        print("=" * 65)
        print(f"QUESTION: {args.query}\n")
        print("ANSWER:")
        print(response.answer)
        print("\n" + "-" * 65)
        print(f"SOURCES CITED ({len(response.sources)} references):")
        print("-" * 65)

        if not response.sources:
            print("No source documents were retrieved for this query.")
        else:
            for idx, doc in enumerate(response.sources, start=1):
                score_str = f"{doc.score:.4f}" if doc.score is not None else "N/A"
                source = doc.metadata.get("file_name") or doc.metadata.get("source") or "unknown"
                page = doc.metadata.get("page")
                page_str = f" | Page: {page}" if page is not None else ""
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                print(f"[{idx}] Score: {score_str} | Source: {source}{page_str}")
                print(f"    Chunk ID: {chunk_id}")
                snippet = doc.content.strip().replace("\n", " ")
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                print(f"    Snippet:  {snippet}\n")

        print("=" * 65 + "\n")

    except Exception as exc:
        logger.bind(query=args.query).exception(f"Generation failed: {exc}")
        print(f"\n[ERROR] Generation failed: {exc}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

