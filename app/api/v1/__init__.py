"""API v1 router bundle."""

from fastapi import APIRouter
from app.api.v1.knowledge import router as knowledge_router

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(knowledge_router)

__all__ = ["api_v1_router", "knowledge_router"]
