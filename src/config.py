"""Quản lý cấu hình hệ thống, đọc từ file .env bằng pydantic-settings."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure local loopback bypasses system proxies (protecting against httpx socks proxy issues on Windows)
_no_proxy = os.environ.get("NO_PROXY", "")
if "localhost" not in _no_proxy:
    os.environ["NO_PROXY"] = f"{_no_proxy},localhost,127.0.0.1".strip(",")

# Thư mục gốc của project (chứa file .env)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Các biến môi trường bắt buộc của ứng dụng RAG."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    BASE_DIR: Path = Field(default=BASE_DIR, description="Project root directory")
    GEMINI_API_KEY: SecretStr = Field(..., min_length=1, description="Gemini API key")
    VECTOR_DB_URL: str = Field(..., min_length=1, description="Vector database connection URL")
    EMBEDDING_MODEL_NAME: str = Field(..., min_length=1, description="Embedding model name")
    EMBEDDING_DIMENSION: int = Field(768, gt=0, le=3072, description="Output dimensionality of embeddings")
    EMBEDDING_BATCH_SIZE: int = Field(100, gt=0, le=100, description="Number of texts per embed request")
    CHUNK_SIZE: int = Field(..., gt=0, description="Size of each chunk")
    CHUNK_OVERLAP: int = Field(..., ge=0, description="Overlap between consecutive chunks")
    CHROMA_COLLECTION_NAME: str = Field("documents", min_length=3, description="ChromaDB collection name")
    DEFAULT_TOP_K: int = Field(5, gt=0, description="Default number of chunks to retrieve")
    MAX_TOP_K: int = Field(50, gt=0, description="Maximum allowed chunks to retrieve")
    GENERATION_MODEL_NAME: str = Field("gemini-3.5-flash", min_length=1, description="Gemini text generation model")
    GENERATION_TEMPERATURE: float = Field(0.2, ge=0.0, le=2.0, description="Temperature for generation")
    UPLOAD_DIR: str = Field("data/uploads", description="Directory for temporary file uploads")
    MAX_UPLOAD_MB: int = Field(20, gt=0, description="Maximum allowed upload file size in megabytes")
    MAX_HISTORY_TURNS: int = Field(6, ge=0, description="Max previous chat turns included in prompt")
    USE_RERANKER: bool = Field(True, description="Enable Cross-Encoder reranking after hybrid search")
    RERANKER_MODEL_NAME: str = Field("BAAI/bge-reranker-v2-m3", min_length=1, description="CrossEncoder reranker model name")
    RERANK_CANDIDATE_K: int = Field(25, gt=0, le=50, description="Number of candidates retrieved from Hybrid Search before reranking (Top 20-30)")
    RERANKER_MAX_LENGTH: int = Field(512, gt=0, le=8192, description="Max token length for CrossEncoder input pairs")
    ENABLE_CONTEXT_CACHING: bool = Field(False, description="Enable Gemini Context Caching for static system documents")
    STATIC_CONTEXT_PATH: str = Field("", description="Optional file or directory path for static system documents to cache")
    CACHE_TTL_SECONDS: int = Field(3600, gt=60, description="TTL in seconds for Gemini cached content")
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis connection URL for query caching")
    ENABLE_QUERY_CACHE: bool = Field(True, description="Enable Redis query/response caching")
    QUERY_CACHE_TTL_SECONDS: int = Field(3600, gt=0, description="TTL in seconds for cached RAG responses")
    ENABLE_SEMANTIC_CACHE: bool = Field(False, description="Enable semantic similarity matching in Redis query cache")
    SEMANTIC_CACHE_THRESHOLD: float = Field(0.92, gt=0.0, le=1.0, description="Minimum cosine similarity for semantic cache hit")
    SEMANTIC_CACHE_MAX_ENTRIES: int = Field(200, gt=0, description="Max cached embeddings per collection for semantic search")


    @model_validator(mode="after")
    def _check_settings_consistency(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP should be smaller than CHUNK_SIZE")
        if self.DEFAULT_TOP_K > self.MAX_TOP_K:
            raise ValueError("DEFAULT_TOP_K cannot be greater than MAX_TOP_K")
        if self.RERANK_CANDIDATE_K > self.MAX_TOP_K:
            raise ValueError("RERANK_CANDIDATE_K cannot be greater than MAX_TOP_K")
        return self


@lru_cache
def get_settings() -> Settings:
    """Trả về instance Settings (cache, chỉ đọc .env một lần)."""
    return Settings()


settings = get_settings()
