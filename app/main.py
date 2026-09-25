import os
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

from config.settings import get_settings
from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.knowledge.store import RuntimeKnowledgeStore
from app.knowledge.sync import KnowledgeSyncService
from app.api.v1.knowledge import router as knowledge_router
from app.voice.router import router as voice_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize runtime knowledge and start background sync."""
    settings = get_settings()

    # 1. Load active runtime knowledge snapshot into memory immediately
    store = RuntimeKnowledgeStore(snapshot_dir=settings.knowledge_snapshot_dir)
    store.load_active_snapshot()

    # 2. Start background sync scheduler (non-blocking)
    sync_service = KnowledgeSyncService(store=store)
    sync_service.start_background_sync(interval_minutes=settings.knowledge_sync_interval_minutes)

    yield

    # 3. Shutdown cleanup
    sync_service.stop_background_sync()


app = FastAPI(title="Voice AI Platform POC", version="0.1.0", lifespan=lifespan)

# Configure CORS
settings = get_settings()
cors_origins = settings.get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knowledge_router)
app.include_router(voice_router)

FRONTEND_HTML_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
VOICE_HTML_PATH = Path(__file__).parent / "templates" / "voice.html"


@app.get("/", response_class=HTMLResponse)
@app.get("/voice", response_class=HTMLResponse)
async def voice_ui():
    """Serve the Coway Voice AI browser client."""
    if FRONTEND_HTML_PATH.exists():
        return HTMLResponse(content=FRONTEND_HTML_PATH.read_text(encoding="utf-8"))
    if VOICE_HTML_PATH.exists():
        return HTMLResponse(content=VOICE_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>Coway Voice AI UI template not found.</h2>", status_code=404)


class MessageRequest(BaseModel):
    department: str
    message: str
    history: Optional[List[Dict[str, str]]] = None


class MessageResponse(BaseModel):
    agent_name: str
    department: str
    response: str
    history: List[Dict[str, str]]
    llm_success: bool


@app.get("/health")
async def health_check():
    """Simple public health check endpoint (no external LLM or voice calls)."""
    return {"status": "ok", "service": "voice_ai_platform_poc"}


@app.get("/status")
async def runtime_status():
    """Safe runtime diagnostic status exposing strictly non-secret operational information."""
    current_settings = get_settings()
    store = RuntimeKnowledgeStore(snapshot_dir=current_settings.knowledge_snapshot_dir)
    kb_status = store.get_status()

    llm_prov = (current_settings.llm_provider or "groq").upper()
    llm_model = current_settings.llm_model

    elevenlabs_configured = bool(
        current_settings.elevenlabs_api_key
        and current_settings.elevenlabs_api_key.strip() not in ("", "your_elevenlabs_api_key_here")
        and current_settings.elevenlabs_speech_engine_id
        and current_settings.elevenlabs_speech_engine_id.strip() not in ("", "your_speech_engine_id_here")
    )
    groq_configured = bool(
        os.getenv("GROQ_API_KEY")
        and os.getenv("GROQ_API_KEY").strip() not in ("", "your_groq_api_key_here")
    )

    return {
        "status": "healthy",
        "service": "coway_voice_ai_backend",
        "environment": current_settings.app_env,
        "llm": {
            "provider": llm_prov,
            "model": llm_model,
            "configured": groq_configured if llm_prov == "GROQ" else True,
        },
        "knowledge": {
            "provider": "LOCAL RUNTIME",
            "version": kb_status.get("active_version", 1),
            "items": kb_status.get("indexed_item_count", 0),
            "status": kb_status.get("status", "READY"),
        },
        "elevenlabs": {
            "speech_engine_configured": elevenlabs_configured,
            "speech_engine_id_configured": bool(current_settings.elevenlabs_speech_engine_id),
            "public_ws_url_configured": bool(current_settings.elevenlabs_public_ws_url),
        },
        "cors_origins": current_settings.get_cors_origins(),
    }


@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    dept = request.department.upper()
    agent = SalesAgent() if dept == "SALES" else CustomerSupportAgent()
    conversation = Conversation(department=dept, active_agent=agent)
    if request.history:
        conversation.history = list(request.history)

    result = await conversation.process_message(request.message)
    return MessageResponse(
        agent_name=result.agent_name,
        department=result.department,
        response=result.text,
        history=conversation.history,
        llm_success=result.llm_result.success,
    )


if __name__ == "__main__":
    port = get_settings().effective_port
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=settings.app_debug)
