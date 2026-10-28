"""FastAPI Application entrypoint for Core RAG Service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.router import api_router
from src.config import BASE_DIR, settings
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifespan handler."""
    logger.info("Starting Core RAG API service...")
    # Ensure upload directory exists
    upload_path = BASE_DIR / settings.UPLOAD_DIR
    upload_path.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down Core RAG API service...")


app = FastAPI(
    title="Core RAG Service API",
    description="Production-grade asynchronous RAG API using FastAPI, ChromaDB, and Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["system"], summary="Service Health Check")
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify service availability."""
    return {"status": "ok", "service": "core-rag-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8080, reload=True)
