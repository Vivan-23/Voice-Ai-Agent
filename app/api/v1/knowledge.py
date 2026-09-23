"""Knowledge Sync & Management Endpoints for Voice AI Platform POC."""

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.knowledge.store import RuntimeKnowledgeStore
from app.knowledge.sync import KnowledgeSyncService

router = APIRouter(prefix="", tags=["Knowledge"])


class SyncResponse(BaseModel):
    success: bool
    action: str
    active_version: int
    reason: Optional[str] = None
    source_count: Optional[int] = None
    indexed_item_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class KnowledgeStatusResponse(BaseModel):
    active_version: int
    status: str
    last_sync: Optional[str] = None
    source_count: int
    indexed_item_count: int
    last_error: Optional[str] = None
    is_syncing: bool
    snapshot_dir: str
    last_sync_duration_seconds: float


@router.get("/knowledge/status", response_model=KnowledgeStatusResponse)
async def get_knowledge_status():
    """Return runtime diagnostic status, active KB version, and last sync info."""
    store = RuntimeKnowledgeStore()
    status_data = store.get_status()
    return KnowledgeStatusResponse(**status_data)


@router.post("/knowledge/sync", response_model=SyncResponse)
async def trigger_knowledge_sync(
    force: bool = Query(default=True, description="Force sync even if metadata is unchanged")
):
    """Trigger synchronization between NotebookLM and runtime knowledge store."""
    service = KnowledgeSyncService()
    result = await service.sync(force=force)
    return SyncResponse(
        success=result.get("success", False),
        action=result.get("action", "UNKNOWN"),
        active_version=result.get("active_version", 0),
        reason=result.get("reason"),
        source_count=result.get("source_count"),
        indexed_item_count=result.get("indexed_item_count"),
        duration_seconds=result.get("duration_seconds"),
        error=result.get("error"),
    )
