"""Pytest configuration and fixtures for testing RAG core."""

import os
import pytest

# Ensure safe dummy environment variables exist before settings are imported
os.environ.setdefault("GEMINI_API_KEY", "dummy_test_gemini_api_key")
os.environ.setdefault("VECTOR_DB_URL", "http://localhost:8000")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
os.environ.setdefault("EMBEDDING_DIMENSION", "768")
os.environ.setdefault("EMBEDDING_BATCH_SIZE", "100")
os.environ.setdefault("CHUNK_SIZE", "1000")
os.environ.setdefault("CHUNK_OVERLAP", "200")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "test_collection")

