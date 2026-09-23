"""Voice AI Platform POC - Main Application Entrypoint with Knowledge Sync Lifespan."""

from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
app.include_router(knowledge_router)
app.include_router(voice_router)

VOICE_HTML_PATH = Path(__file__).parent / "templates" / "voice.html"


@app.get("/", response_class=HTMLResponse)
@app.get("/voice", response_class=HTMLResponse)
async def voice_ui():
    """Serve the minimal Coway Voice AI browser client."""
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
    return {"status": "ok", "service": "voice_ai_platform_poc"}


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
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
