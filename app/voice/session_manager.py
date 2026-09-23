"""Voice Session Manager for ElevenLabs Speech Engine integration.

Maintains isolated per-session conversation state, manages active specialists,
routes customer utterances to the persistent agent (Sarah / Alex), and tracks
low-latency turn telemetry.
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.knowledge.local import LocalKnowledgeProvider
from app.knowledge.store import RuntimeKnowledgeStore
from config.settings import get_settings

logger = logging.getLogger("app.voice.session_manager")


@dataclass
class VoiceTurnTiming:
    """Latency metrics for a single voice turn."""
    transcript: str
    turn_total_seconds: float = 0.0
    stt_seconds: Optional[float] = None
    llm_initial_seconds: float = 0.0
    knowledge_seconds: Optional[float] = None
    llm_final_seconds: Optional[float] = None
    time_to_first_text_seconds: float = 0.0
    time_to_first_audio_seconds: Optional[float] = None
    llm_provider: str = "GROQ"
    knowledge_provider: str = "LOCAL"
    notebooklm_called: bool = False
    manager_invoked: bool = False
    manager_action: Optional[str] = None
    assigned_agent: Optional[str] = None
    agent_name: str = "Sarah"
    department: str = "SALES"

    def format_debug_block(self) -> str:
        """Format clean voice turn telemetry block per specification."""
        lines = [
            "",
            "VOICE TURN",
            "-" * 45,
            f"STT TRANSCRIPT:        \"{self.transcript[:40]}{'...' if len(self.transcript) > 40 else ''}\"",
        ]
        if self.stt_seconds is not None:
            lines.append(f"STT:                   {self.stt_seconds:.2f}s")
        lines.append(f"GROQ / DECISION:       {self.llm_initial_seconds:.4f}s ({self.llm_provider})")
        if self.knowledge_seconds is not None:
            lines.append(f"LOCAL KNOWLEDGE:       {self.knowledge_seconds:.4f}s")
        else:
            lines.append(f"LOCAL KNOWLEDGE:       NOT NEEDED")
        if self.llm_final_seconds is not None:
            lines.append(f"GROQ RESPONSE:         {self.llm_final_seconds:.4f}s")
        if self.time_to_first_text_seconds > 0:
            lines.append(f"FIRST TEXT:            {self.time_to_first_text_seconds:.4f}s")
        if self.time_to_first_audio_seconds is not None:
            lines.append(f"FIRST AUDIO:           {self.time_to_first_audio_seconds:.2f}s")
        lines.append(f"TOTAL SERVER TURN:     {self.turn_total_seconds:.2f}s")
        lines.append(f"KNOWLEDGE SOURCE:      {self.knowledge_provider}")
        lines.append(f"NOTEBOOKLM:            {'CALLED' if self.notebooklm_called else 'NOT CALLED'}")
        lines.append(f"SPECIALIST AGENT:      {self.agent_name} ({self.department})")
        lines.append(f"MANAGER STATUS:        {self.manager_action if self.manager_invoked else 'NOT INVOKED'}")
        lines.append("-" * 45)
        return "\n".join(lines)


class VoiceSession:
    """Represents an isolated active voice conversation session."""

    def __init__(
        self,
        session_id: str,
        department: str,
        conversation: Conversation,
        greeting: str,
    ):
        self.session_id = session_id
        self.conversation_id: Optional[str] = None
        self.department = department.upper()
        self.conversation = conversation
        self.greeting = greeting
        self.created_at = time.perf_counter()
        self.last_activity = self.created_at
        self.timings_history: List[VoiceTurnTiming] = []
        self.is_active = True

    @property
    def agent_name(self) -> str:
        return self.conversation.active_agent.name

    def record_turn(self, timing: VoiceTurnTiming):
        self.timings_history.append(timing)
        self.last_activity = time.perf_counter()


class VoiceSessionManager:
    """Thread-safe manager for concurrent voice sessions."""

    def __init__(self):
        self._sessions: Dict[str, VoiceSession] = {}
        self._pending_sessions: Dict[str, VoiceSession] = {}
        self._recent_pending: List[str] = []

    def create_pending_session(self, department: str) -> VoiceSession:
        """Create a new pending session reserved with the chosen department."""
        settings = get_settings()
        dept = department.strip().upper()
        if dept not in ("SALES", "CUSTOMER_SUPPORT", "SUPPORT", "SERVICE"):
            dept = "SALES"
        if dept in ("SUPPORT", "SERVICE"):
            dept = "CUSTOMER_SUPPORT"

        # Explicitly use LocalKnowledgeProvider for voice sessions (near-zero latency)
        store = RuntimeKnowledgeStore(snapshot_dir=settings.knowledge_snapshot_dir)
        store.load_active_snapshot()
        knowledge_provider = LocalKnowledgeProvider(store=store)

        if dept == "SALES":
            agent = SalesAgent(knowledge_provider=knowledge_provider)
            greeting = "Hi, I'm Sarah from Coway Sales. How can I help you today?"
        else:
            agent = CustomerSupportAgent(knowledge_provider=knowledge_provider)
            greeting = "Hi, I'm Alex from Coway Customer Support. How can I help you today?"

        session_id = f"voice_{uuid.uuid4().hex[:10]}"
        conv = Conversation(
            department=dept,
            session_id=session_id,
            active_agent=agent,
            knowledge_provider=knowledge_provider,
        )
        # Pre-seed initial greeting into conversation history for the LLM
        conv.history.append({"role": "assistant", "content": greeting})

        session = VoiceSession(
            session_id=session_id,
            department=dept,
            conversation=conv,
            greeting=greeting,
        )

        self._pending_sessions[session_id] = session
        self._recent_pending.append(session_id)
        # Keep recent pending queue bounded
        if len(self._recent_pending) > 50:
            old = self._recent_pending.pop(0)
            self._pending_sessions.pop(old, None)

        logger.info(f"Created pending voice session {session_id} for department={dept}")
        return session

    def bind_conversation(
        self,
        conversation_id: str,
        session_id: Optional[str] = None,
        department: Optional[str] = None,
    ) -> VoiceSession:
        """Bind an ElevenLabs conversation_id to a pending or new VoiceSession."""
        if conversation_id in self._sessions:
            return self._sessions[conversation_id]

        session: Optional[VoiceSession] = None
        if session_id and session_id in self._pending_sessions:
            session = self._pending_sessions.pop(session_id)
            if session_id in self._recent_pending:
                self._recent_pending.remove(session_id)
        elif self._recent_pending:
            # Match latest pending session if session_id wasn't passed directly
            latest_id = self._recent_pending.pop(-1)
            session = self._pending_sessions.pop(latest_id, None)

        if session is None:
            # Fallback: create fresh session with specified department or default SALES
            fallback_dept = (department or "SALES").upper()
            session = self.create_pending_session(fallback_dept)
            self._pending_sessions.pop(session.session_id, None)
            if session.session_id in self._recent_pending:
                self._recent_pending.remove(session.session_id)

        session.conversation_id = conversation_id
        self._sessions[conversation_id] = session
        logger.info(
            f"Bound conversation_id={conversation_id} to session={session.session_id} "
            f"(agent={session.agent_name}, dept={session.department})"
        )
        return session

    def get_session(self, conversation_id: str) -> Optional[VoiceSession]:
        """Retrieve the active VoiceSession for an ElevenLabs conversation_id."""
        if conversation_id in self._sessions:
            return self._sessions[conversation_id]
        # If not explicitly bound yet, check if there is an available pending session
        if self._recent_pending:
            return self.bind_conversation(conversation_id)
        return None

    def remove_session(self, conversation_id: str) -> Optional[VoiceSession]:
        """Clean up and remove an ended voice session."""
        session = self._sessions.pop(conversation_id, None)
        if session:
            session.is_active = False
            logger.info(f"Cleaned up voice session for conversation_id={conversation_id}")
        return session

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)


_manager_instance: Optional[VoiceSessionManager] = None


def get_voice_session_manager() -> VoiceSessionManager:
    """Return singleton instance of VoiceSessionManager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = VoiceSessionManager()
    return _manager_instance
