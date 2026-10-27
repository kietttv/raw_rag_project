"""Pydantic schemas for API requests and responses."""

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


class ChatResponse(BaseModel):
    """Response payload from RAG chat endpoint."""

    answer: str = Field(..., description="Generated answer from LLM")
    sources: list[SourceDocument] = Field(default_factory=list, description="Source chunks used as context")


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

