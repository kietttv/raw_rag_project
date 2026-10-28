"""Tracing and LLMOps Observability service integrating Langfuse.

Captures end-to-end query lifecycles, per-step latencies (Retriever, Re-ranker, LLM),
token usage metrics, and user feedback (thumbs up/down).
Provides safe no-op fallback when tracing is disabled or unconfigured.
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import Any

from src.config import settings
from src.utils.logger import logger

_client_lock = threading.Lock()
_langfuse_client: Any | None = None
_client_initialized = False


def get_langfuse_client() -> Any | None:
    """Get or initialize singleton Langfuse client instance.

    Returns:
        Langfuse instance if ENABLE_TRACING is True and credentials exist, else None.
    """
    global _langfuse_client, _client_initialized

    if not settings.ENABLE_TRACING:
        return None

    if _client_initialized:
        return _langfuse_client

    with _client_lock:
        if _client_initialized:
            return _langfuse_client

        pub_key = (settings.LANGFUSE_PUBLIC_KEY or "").strip()
        sec_key = settings.LANGFUSE_SECRET_KEY.get_secret_value().strip()

        if not pub_key or not sec_key:
            logger.warning("ENABLE_TRACING is true but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is empty; tracing disabled")
            _client_initialized = True
            _langfuse_client = None
            return None

        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=pub_key,
                secret_key=sec_key,
                host=settings.LANGFUSE_HOST,
                release=settings.LANGFUSE_RELEASE,
            )
            logger.bind(host=settings.LANGFUSE_HOST, release=settings.LANGFUSE_RELEASE).info(
                "Successfully initialized Langfuse tracing client"
            )
        except Exception as exc:
            logger.bind(error=str(exc)).warning(
                "Failed to initialize Langfuse client; continuing in no-op tracing mode"
            )
            _langfuse_client = None

        _client_initialized = True
        return _langfuse_client


def reset_tracing_client() -> None:
    """Reset singleton tracing client (useful for unit testing)."""
    global _langfuse_client, _client_initialized
    with _client_lock:
        _langfuse_client = None
        _client_initialized = False


def flush_tracing() -> None:
    """Flush pending traces to Langfuse backend synchronously."""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
            logger.debug("Flushed Langfuse traces successfully")
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Error flushing Langfuse traces")


# Register flush on normal python interpreter exit
atexit.register(flush_tracing)


class TraceContext:
    """Context wrapper for tracking a single RAG query interaction."""

    def __init__(
        self,
        trace_id: str,
        name: str = "rag_chat_query",
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.name = name
        self.start_time = time.perf_counter()
        self.client = client
        self.metadata = metadata or {}
        self.langfuse_trace = None

        if self.client is not None:
            try:
                self.langfuse_trace = self.client.trace(
                    id=self.trace_id,
                    name=self.name,
                    input=input_data,
                    metadata=self.metadata,
                    release=settings.LANGFUSE_RELEASE,
                )
            except Exception as exc:
                logger.bind(trace_id=trace_id, error=str(exc)).debug("Failed creating Langfuse trace")
                self.langfuse_trace = None

    def record_span(
        self,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an execution step (e.g. retriever, reranker, input_guard)."""
        logger.bind(
            trace_id=self.trace_id,
            span=name,
            duration_ms=duration_ms,
        ).debug(f"Span executed: {name}")

        if self.langfuse_trace is not None:
            try:
                span = self.langfuse_trace.span(
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=metadata or {},
                )
                span.end()
            except Exception as exc:
                logger.bind(span=name, error=str(exc)).debug("Failed recording Langfuse span")

    def record_generation(
        self,
        name: str = "gemini_generation",
        model: str = "",
        prompt: Any = None,
        completion: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an LLM generation step with token counts and latency."""
        meta = metadata or {}
        if duration_ms is not None:
            meta["latency_ms"] = duration_ms

        logger.bind(
            trace_id=self.trace_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        ).info(f"LLM Generation completed ({model})")

        if self.langfuse_trace is not None:
            try:
                usage = {
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": total_tokens or (prompt_tokens + completion_tokens),
                }
                gen = self.langfuse_trace.generation(
                    name=name,
                    model=model or settings.GENERATION_MODEL_NAME,
                    input=prompt,
                    output=completion,
                    usage=usage,
                    metadata=meta,
                )
                gen.end()
            except Exception as exc:
                logger.bind(trace_id=self.trace_id, error=str(exc)).debug("Failed recording Langfuse generation")

    def end(self, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        """Conclude the query trace."""
        duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        if self.langfuse_trace is not None:
            try:
                update_kwargs: dict[str, Any] = {"output": output}
                if metadata:
                    update_kwargs["metadata"] = {**self.metadata, **metadata, "duration_ms": duration_ms}
                self.langfuse_trace.update(**update_kwargs)
            except Exception as exc:
                logger.bind(trace_id=self.trace_id, error=str(exc)).debug("Failed closing Langfuse trace")


def start_trace(
    query: str,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceContext:
    """Start an observability trace for a RAG question.

    Args:
        query: Incoming user question.
        trace_id: Optional trace / request ID (defaults to new hex uuid).
        metadata: Optional request metadata (collection, top_k, streaming).

    Returns:
        TraceContext: Context manager object for spans and generation metrics.
    """
    import uuid

    tid = trace_id or uuid.uuid4().hex
    client = get_langfuse_client()
    return TraceContext(
        trace_id=tid,
        name="rag_chat_query",
        input_data={"query": query},
        metadata=metadata,
        client=client,
    )


def record_user_feedback(
    trace_id: str,
    score: int,
    comment: str | None = None,
) -> bool:
    """Record user satisfaction rating (thumbs up = 1, thumbs down = -1).

    Args:
        trace_id: Target interaction trace ID.
        score: Rating value (-1, 0, or 1).
        comment: Optional textual justification or correction.

    Returns:
        bool: True if recorded successfully or handled by no-op.
    """
    logger.bind(
        trace_id=trace_id,
        score=score,
        comment=comment,
    ).info("Received user feedback rating for chat query")

    client = get_langfuse_client()
    if client is not None:
        try:
            client.score(
                trace_id=trace_id,
                name="user_feedback",
                value=float(score),
                comment=comment,
            )
            client.flush()
            return True
        except Exception as exc:
            logger.bind(trace_id=trace_id, error=str(exc)).warning(
                "Failed to send user feedback to Langfuse"
            )
            return False

    return True

