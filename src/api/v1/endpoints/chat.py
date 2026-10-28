"""Chat endpoints for querying RAG pipeline with optional real-time streaming."""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.config import settings
from src.core.generator import generate_answer, stream_answer
from src.core.retriever import retrieve
from src.guardrails.input_guard import (
    InputGuardViolationError,
    validate_chat_input,
)
from src.guardrails.output_guard import (
    FALLBACK_NO_INFO_MESSAGE,
    apply_output_guard,
    sanitize_sources,
)
from src.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    SourceDocument,
)
from src.services import cache_service
from src.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])


def _format_sse(event: str, data: dict | list) -> str:
    """Format SSE event block compliant with Server-Sent Events standard."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _generate_sse_stream(
    query: str,
    sources: list[SourceDocument],
    history: list[ChatMessage],
) -> Iterator[str]:
    """Yield SSE events for sources, tokens, and completion/error signal."""
    # 1. Sanitize and send retrieved sources first
    clean_sources = sanitize_sources(sources) if settings.ENABLE_OUTPUT_GUARD else sources
    sources_data = [s.model_dump() for s in clean_sources]
    yield _format_sse("sources", sources_data)

    total_chars = 0
    try:
        # 2. Stream tokens from LLM
        for token in stream_answer(query=query, sources=clean_sources, history=history):
            total_chars += len(token)
            yield _format_sse("token", {"text": token})

        # 2b. Check for empty hallucination in stream
        if total_chars == 0 and settings.ENABLE_OUTPUT_GUARD:
            fallback = FALLBACK_NO_INFO_MESSAGE
            total_chars = len(fallback)
            yield _format_sse("token", {"text": fallback})

        # 3. Send completion event
        yield _format_sse("done", {"answer_length": total_chars})
    except Exception as exc:
        logger.bind(query=query).exception(f"Error during streaming generation: {exc}")
        yield _format_sse("error", {"detail": "LLM generation failed during streaming"})


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        200: {
            "description": (
                "Standard ChatResponse JSON (when stream=false) or "
                "text/event-stream SSE chunks (when stream=true)."
            ),
            "content": {
                "application/json": {},
                "text/event-stream": {},
            },
        }
    },
    summary="Ask a question to the RAG system",
)
def chat_endpoint(request: ChatRequest):
    """Answer user questions using retrieved context and Gemini generation.

    - **stream=false**: Returns a JSON `ChatResponse` containing answer and source references.
    - **stream=true**: Returns a `text/event-stream` delivering `sources`, `token`, and `done` events.
    """
    # 0. Validate input against security guardrails (Prompt Injection, Out-of-Scope, Bounds)
    try:
        clean_query = validate_chat_input(request.query, request.history)
    except InputGuardViolationError as guard_err:
        logger.bind(reason=guard_err.reason, rule=guard_err.matched_rule).warning(
            "Chat request rejected by input guardrail"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=guard_err.reason,
        )

    # 1. Extract and trim history to MAX_HISTORY_TURNS
    max_turns = settings.MAX_HISTORY_TURNS
    trimmed_history: list[ChatMessage] = []
    if request.history and max_turns > 0:
        # Keep up to max_turns * 2 individual turns (user + assistant)
        trimmed_history = request.history[-(max_turns * 2) :]

    # 1.1 Check Redis Query Cache for standalone questions (0 API calls on cache hit)
    if not trimmed_history and settings.ENABLE_QUERY_CACHE:
        cached_resp = cache_service.get_cached_response(
            query=clean_query,
            collection_name=request.collection_name,
            top_k=request.top_k,
        )
        if cached_resp is not None:
            if not request.stream:
                return cached_resp

            def _stream_from_cache(cached: ChatResponse) -> Iterator[str]:
                yield _format_sse("sources", [s.model_dump() for s in cached.sources])
                yield _format_sse("token", {"text": cached.answer})
                yield _format_sse("done", {"answer_length": len(cached.answer), "cached": True})

            return StreamingResponse(
                _stream_from_cache(cached_resp),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    # 2. Retrieve relevant documents
    try:
        sources = retrieve(
            query=clean_query,
            top_k=request.top_k,
            min_score=request.min_score,
            collection_name=request.collection_name,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.bind(query=clean_query).exception(f"Retrieval failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service failed or vector database unreachable",
        )

    # 3. Branch: non-streaming response
    if not request.stream:
        try:
            resp = generate_answer(
                query=clean_query,
                sources=sources,
                history=trimmed_history,
            )
            if settings.ENABLE_OUTPUT_GUARD:
                resp = apply_output_guard(resp)

            if not trimmed_history and settings.ENABLE_QUERY_CACHE:
                cache_service.set_cached_response(
                    query=clean_query,
                    response=resp,
                    collection_name=request.collection_name,
                    top_k=request.top_k,
                )
            return resp
        except Exception as exc:
            logger.bind(query=clean_query).exception(f"LLM generation failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM generation service failed to produce an answer",
            )

    # 4. Branch: streaming response via Server-Sent Events
    return StreamingResponse(
        _generate_sse_stream(
            query=clean_query,
            sources=sources,
            history=trimmed_history,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

