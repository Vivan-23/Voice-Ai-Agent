"""Health and readiness check endpoints."""

from fastapi import APIRouter, Depends
from app.api.dependencies import get_knowledge_service, get_stt_service, get_tts_service
from app.interfaces.knowledge import BaseKnowledgeService
from app.interfaces.stt import BaseSTTService
from app.interfaces.tts import BaseTTSService

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/")
async def health():
    """Basic service liveness check."""
    return {"status": "ok", "service": "voice-ai-platform"}


@router.get("/ready")
async def readiness(
    knowledge: BaseKnowledgeService = Depends(get_knowledge_service),
    stt: BaseSTTService = Depends(get_stt_service),
    tts: BaseTTSService = Depends(get_tts_service),
):
    """Component readiness verification."""
    knowledge_ok = await knowledge.health_check()
    return {
        "status": "ready" if knowledge_ok else "degraded",
        "components": {
            "knowledge_mcp": "ok" if knowledge_ok else "error",
            "stt_provider": "configured",
            "tts_provider": "configured",
        },
    }
