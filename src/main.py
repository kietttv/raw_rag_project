"""FastAPI Application entrypoint for Core RAG Service."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# 1. Request Timing & Request-ID Middleware
@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Measure request processing duration and attach tracing headers."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    response.headers["X-Request-ID"] = request_id

    logger.bind(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    ).info("Request completed")

    return response


# 2. CORS configuration (with exposed timing and tracking headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms", "X-Request-ID"],
)

# ---------------------------------------------------------------------------
# Centralized Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Standardized handler for HTTPExceptions."""
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or uuid.uuid4().hex

    if exc.status_code >= 500:
        logger.bind(request_id=request_id, path=request.url.path, status_code=exc.status_code).error(
            f"HTTPException: {exc.detail}"
        )
    else:
        logger.bind(request_id=request_id, path=request.url.path, status_code=exc.status_code).warning(
            f"HTTPException: {exc.detail}"
        )

    headers = getattr(exc, "headers", None) or {}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "error_type": "http_error",
            "request_id": request_id,
        },
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Standardized handler for request validation failures (422)."""
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or uuid.uuid4().hex

    logger.bind(request_id=request_id, path=request.url.path, errors=exc.errors()).warning(
        "Request validation error"
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "status_code": 422,
            "error_type": "validation_error",
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Fallback handler for unhandled server exceptions (500)."""
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or uuid.uuid4().hex

    logger.bind(request_id=request_id, path=request.url.path).exception(
        f"Unhandled server error: {exc}"
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error_type": "internal_error",
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# Routers & System Endpoints
# ---------------------------------------------------------------------------

# Include API Routers
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["system"], summary="Service Health Check")
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify service availability."""
    return {"status": "ok", "service": "core-rag-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8080, reload=True)
