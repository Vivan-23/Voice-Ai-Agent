"""Call lifecycle REST endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from app.schemas.call import CallSession, CallStatus
from app.api.dependencies import get_post_call_processor
from app.services.post_call.processor import PostCallProcessor
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/calls", tags=["Calls"])


class InitiateCallRequest(BaseModel):
    caller_phone: str
    recipient_phone: str


class EndCallRequest(BaseModel):
    call_id: str


@router.post("/initiate", response_model=CallSession)
async def initiate_call(request: InitiateCallRequest):
    """Initiate a new voice call session."""
    call_id = f"call_{abs(hash(request.caller_phone)) % 1000000}"
    logger.info(f"Initiating call {call_id} from {request.caller_phone}...")
    return CallSession(
        call_id=call_id,
        caller_phone=request.caller_phone,
        recipient_phone=request.recipient_phone,
        status=CallStatus.IN_PROGRESS,
    )


@router.post("/{call_id}/end")
async def end_call(
    call_id: str,
    background_tasks: BackgroundTasks,
    processor: PostCallProcessor = Depends(get_post_call_processor),
):
    """End call and trigger background CRM sync and analytics."""
    logger.info(f"Ending call {call_id} and scheduling post-call processing...")
    # Mock session
    session = CallSession(
        call_id=call_id,
        caller_phone="+919876543210",
        recipient_phone="+9118001026960",
        status=CallStatus.COMPLETED,
    )

    # Dispatch post-call processing as background task
    background_tasks.add_task(processor.process_completed_call, session)

    return {
        "status": "call_ended",
        "call_id": call_id,
        "message": "Post-call processing queued for background execution.",
    }
