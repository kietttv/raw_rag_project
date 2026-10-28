"""Chat endpoints for querying RAG pipeline with optional real-time streaming."""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.config import settings
from src.core.generator import generate_answer, stream_answer
from src.core.retriever import retrieve
from src.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    SourceDocument,
)
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
    # 1. Send retrieved sources first
    sources_data = [s.model_dump() for s in sources]
    yield _format_sse("sources", sources_data)

    total_chars = 0
    try:
        # 2. Stream tokens from LLM
        for token in stream_answer(query=query, sources=sources, history=history):
            total_chars += len(token)
            yield _format_sse("token", {"text": token})

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
    # 1. Extract and trim history to MAX_HISTORY_TURNS
    max_turns = settings.MAX_HISTORY_TURNS
    trimmed_history: list[ChatMessage] = []
    if request.history and max_turns > 0:
        # Keep up to max_turns * 2 individual turns (user + assistant)
        trimmed_history = request.history[-(max_turns * 2) :]

    # 2. Retrieve relevant documents
    try:
        sources = retrieve(
            query=request.query,
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
        logger.bind(query=request.query).exception(f"Retrieval failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service failed or vector database unreachable",
        )

    # 3. Branch: non-streaming response
    if not request.stream:
        try:
            return generate_answer(
                query=request.query,
                sources=sources,
                history=trimmed_history,
            )
        except Exception as exc:
            logger.bind(query=request.query).exception(f"LLM generation failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM generation service failed to produce an answer",
            )

    # 4. Branch: streaming response via Server-Sent Events
    return StreamingResponse(
        _generate_sse_stream(
            query=request.query,
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

