"""In-memory multi-turn conversation session store."""

import asyncio
import uuid
from typing import Dict, Optional, Any
from app.agents.state import CallAgentState
from app.schemas.agent import AgentRole, UserIntent, CallLifecycleStage
from app.schemas.conversation import ResolutionStatus
from app.core.logging import get_logger

logger = get_logger(__name__)


class ConversationSessionStore:
    """Thread-safe in-memory store for conversational states across turns."""

    def __init__(self):
        self._sessions: Dict[str, CallAgentState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        session_id: str,
        customer_phone: str = "+919876543210",
        customer_name: str = "Valued Customer",
        language: str = "English",
        department: str = "SALES",
        specialist_agent: AgentRole = AgentRole.SALES,
        ivr_completed: bool = True,
    ) -> CallAgentState:
        """Retrieve existing session state or initialize a fresh state."""
        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            initial_state: CallAgentState = {
                "session_id": session_id,
                "call_id": f"call_{uuid.uuid4().hex[:8]}",
                "customer": {
                    "name": customer_name,
                    "phone": customer_phone,
                },
                "customer_phone": customer_phone,
                "language": language,
                "department": department,
                "specialist_agent": specialist_agent,
                "ivr_completed": ivr_completed,
                "messages": [],
                "conversation_history": [],
                "active_agent": specialist_agent,
                "previous_agent": None,
                "current_intent": UserIntent.GREETING,
                "stage": CallLifecycleStage.CONVERSATION_ACTIVE if ivr_completed else CallLifecycleStage.IVR_LANGUAGE_SELECTION,
                "active_topic": None,
                "active_product": None,
                "mentioned_products": [],
                "comparison_products": [],
                "last_product_question": None,
                "last_agent_question": None,
                "last_agent_instruction": None,
                "current_troubleshooting_step": None,
                "troubleshooting_context": {},
                "room_type": None,
                "room_size": None,
                "sales_stage": None,
                "retrieved_knowledge": [],
                "knowledge_grounded": True,
                "citations": [],
                "needs_manager": False,
                "manager_intervention": False,
                "manager_action": None,
                "manager_trigger_reason": None,
                "manager_interventions_count": 0,
                "manager_guidance": None,
                "human_escalation_required": False,
                "unresolved_issue": None,
                "resolution_status": ResolutionStatus.IN_PROGRESS.value,
                "turn_count": 0,
                "customer_frustrated": False,
                "frustration_score": 0.0,
                "turn_uncertainty": False,
                "consecutive_failures": 0,
                "last_response": None,
                "conversation_state": None,
                "latest_decision": None,
            }
            self._sessions[session_id] = initial_state
            logger.info(f"Initialized new conversational session: {session_id}")
            return initial_state

    async def save(self, session_id: str, state: CallAgentState) -> None:
        """Save updated state for a session."""
        async with self._lock:
            self._sessions[session_id] = state

    async def get(self, session_id: str) -> Optional[CallAgentState]:
        """Fetch session state if it exists."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def clear(self, session_id: str) -> bool:
        """Remove a session from memory."""
        async with self._lock:
            removed = self._sessions.pop(session_id, None)
            return removed is not None

    def get_sync(self, session_id: str) -> Optional[CallAgentState]:
        """Synchronous fetch for session state."""
        return self._sessions.get(session_id)

    def clear_sync(self, session_id: str) -> bool:
        """Synchronous session clear."""
        return self._sessions.pop(session_id, None) is not None

    def clear_all(self) -> None:
        """Clear all active sessions."""
        self._sessions.clear()


# Global singleton instance
session_store = ConversationSessionStore()
