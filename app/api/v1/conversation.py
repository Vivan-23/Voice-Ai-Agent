"""Text-first conversational AI endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from app.schemas.conversation import (
    ConversationTurnRequest,
    ConversationTurnResponse,
)
from app.services.conversation import ConversationService
from app.api.dependencies import get_conversation_service
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/conversation", tags=["Conversation"])


@router.post("/message", response_model=ConversationTurnResponse)
async def send_message(
    request: ConversationTurnRequest,
    service: ConversationService = Depends(get_conversation_service),
):
    """Process a conversational text message through multi-agent LangGraph workflow."""
    try:
        response = await service.process_message(request)
        return response
    except Exception as e:
        logger.error(f"Error processing message for session '{request.session_id}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing conversation: {str(e)}",
        )


@router.get("/{session_id}/state")
async def get_session_state(
    session_id: str,
    service: ConversationService = Depends(get_conversation_service),
):
    """Inspect active conversation state for debugging and supervision."""
    if session_id not in service.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = service.sessions[session_id]
    # Serialize state without raw BaseMessages
    return {
        "session_id": state.get("session_id"),
        "call_id": state.get("call_id"),
        "active_agent": state.get("active_agent"),
        "previous_agent": state.get("previous_agent"),
        "current_intent": state.get("current_intent"),
        "turn_count": state.get("turn_count"),
        "customer": state.get("customer"),
        "knowledge_grounded": state.get("knowledge_grounded"),
        "citations": state.get("citations"),
        "manager_interventions_count": state.get("manager_interventions_count"),
        "human_escalation_required": state.get("human_escalation_required"),
        "resolution_status": state.get("resolution_status"),
        "customer_frustrated": state.get("customer_frustrated"),
        "consecutive_failures": state.get("consecutive_failures"),
        "history_length": len(state.get("conversation_history", [])),
    }


@router.post("/{session_id}/reset")
async def reset_session(
    session_id: str,
    service: ConversationService = Depends(get_conversation_service),
):
    """Reset active session state."""
    success = service.reset_session(session_id)
    return {"session_id": session_id, "reset": success}
