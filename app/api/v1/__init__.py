"""API v1 router bundle."""

from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.calls import router as calls_router
from app.api.v1.audio import router as audio_router
from app.api.v1.conversation import router as conversation_router

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(calls_router)
api_v1_router.include_router(audio_router)
api_v1_router.include_router(conversation_router)

__all__ = ["api_v1_router"]
