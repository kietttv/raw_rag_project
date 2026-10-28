"""Aggregated API v1 router."""

from fastapi import APIRouter

from src.api.v1.endpoints import ingest

api_router = APIRouter()
api_router.include_router(ingest.router)

