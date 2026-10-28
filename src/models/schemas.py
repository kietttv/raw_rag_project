"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single turn in a chat conversation."""

    role: Literal["user", "assistant"] = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message text content")


class SourceDocument(BaseModel):
    """Retrieved chunk reference used to answer a question."""

    content: str = Field(..., description="Retrieved chunk text")
    score: float | None = Field(None, description="Relevance similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata (source, page, id, etc.)")


class ChatRequest(BaseModel):
    """Request payload for RAG chat endpoint."""

    query: str = Field(..., min_length=1, description="User question")
    history: list[ChatMessage] = Field(default_factory=list, description="Previous conversation turns")
    top_k: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    collection_name: str | None = Field(None, description="Target ChromaDB collection (defaults to settings)")
    min_score: float | None = Field(None, ge=-1.0, le=1.0, description="Minimum relevance score threshold")
    stream: bool = Field(False, description="Stream response tokens in real-time via Server-Sent Events")


class ChatResponse(BaseModel):
    """Response payload from RAG chat endpoint."""

    answer: str = Field(..., description="Generated answer from LLM")
    sources: list[SourceDocument] = Field(default_factory=list, description="Source chunks used as context")
    trace_id: str | None = Field(None, description="Observability trace ID for this query interaction")


class FeedbackScore(int, Enum):
    """User feedback rating score."""

    THUMBS_DOWN = -1
    NEUTRAL = 0
    THUMBS_UP = 1


class ChatFeedbackRequest(BaseModel):
    """Payload to record user rating on a specific chat answer."""

    trace_id: str = Field(..., min_length=1, description="Trace ID or Request ID of the chat turn")
    score: int = Field(..., ge=-1, le=1, description="1 for thumbs-up, -1 for thumbs-down, 0 for neutral")
    comment: str | None = Field(None, max_length=1000, description="Optional user feedback comment")


class ChatFeedbackResponse(BaseModel):
    """Response payload acknowledging feedback recorded."""

    status: str = Field("success", description="Status confirmation")
    trace_id: str = Field(..., description="Target trace ID")
    score: int = Field(..., description="Recorded score")
    message: str = Field("Feedback recorded successfully", description="Status summary")


class DocumentInput(BaseModel):
    """Single raw document input for ingestion."""

    text: str = Field(..., min_length=1, description="Raw document text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary document metadata (e.g. source, title, author)",
    )


class IngestRequest(BaseModel):
    """Request payload for document ingestion endpoint."""

    documents: list[DocumentInput] = Field(..., min_length=1, description="List of documents to ingest")


class IngestResponse(BaseModel):
    """Response payload for document ingestion endpoint."""

    status: Literal["success", "partial", "failed"] = Field(..., description="Ingestion processing status")
    documents_processed: int = Field(..., ge=0, description="Number of documents successfully processed")
    chunks_created: int = Field(..., ge=0, description="Total number of chunks produced and embedded")
    message: str | None = Field(None, description="Status detail or error message if any")


class TaskStatus(str, Enum):
    """Execution status of an asynchronous background task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestTaskResponse(BaseModel):
    """Response payload returned immediately upon submitting an ingestion task."""

    task_id: str = Field(..., description="Unique identifier for the background task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task state")
    file_name: str = Field(..., description="Name of the uploaded file")
    message: str = Field("File accepted for processing", description="Human-readable status summary")


class IngestTaskDetail(BaseModel):
    """Detailed response payload returned when polling an ingestion task's status."""

    task_id: str = Field(..., description="Unique identifier for the background task")
    status: TaskStatus = Field(..., description="Current task state")
    file_name: str = Field(..., description="Name of the uploaded file")
    collection_name: str = Field(..., description="Target ChromaDB collection")
    files_processed: int | None = Field(None, description="Number of files successfully processed")
    chunks_created: int | None = Field(None, description="Number of chunks embedded and stored")
    failed_files: list[str] = Field(default_factory=list, description="List of failed file names/paths")
    error: str | None = Field(None, description="Error message if the task failed")
    created_at: datetime = Field(..., description="Timestamp when task was received")
    updated_at: datetime = Field(..., description="Timestamp when task status was last updated")


