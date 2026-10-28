"""Quản lý cấu hình hệ thống, đọc từ file .env bằng pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    @model_validator(mode="after")
    def _check_settings_consistency(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP should be smaller than CHUNK_SIZE")
        if self.DEFAULT_TOP_K > self.MAX_TOP_K:
            raise ValueError("DEFAULT_TOP_K cannot be greater than MAX_TOP_K")
        return self


@lru_cache
def get_settings() -> Settings:
    """Trả về instance Settings (cache, chỉ đọc .env một lần)."""
    return Settings()


settings = get_settings()