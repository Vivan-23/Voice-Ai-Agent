"""Dedicated IVR Entry & Session Initialization Component.

Handles pre-call routing:
1. Language selection (English / Hindi)
2. Department selection (SALES / CUSTOMER_SERVICE)
3. Session initialization & Specialist assignment
"""

import re
from typing import Optional, Tuple
from app.agents.state import CallAgentState
from app.schemas.agent import AgentRole, CallLifecycleStage, UserIntent
from app.schemas.conversation import ConversationTurnResponse, ResolutionStatus
from app.services.conversation.session_store import ConversationSessionStore, session_store
from app.core.logging import get_logger

logger = get_logger(__name__)

# IVR Prompt Constants
IVR_LANGUAGE_PROMPT = "Welcome to Coway India. For English, press 1. For Hindi, press 2."
IVR_DEPARTMENT_PROMPT_EN = "For Sales, press 1. For Customer Support, press 2."
IVR_DEPARTMENT_PROMPT_HI = "बिक्री (Sales) के लिए 1 दबाएं। ग्राहक सेवा (Customer Support) के लिए 2 दबाएं।"


class CallSessionInitializer:
    """Manages pre-call IVR routing and specialist session creation."""

    def __init__(self, store: Optional[ConversationSessionStore] = None):
        self.store = store or session_store

    async def start_call(
        self,
        session_id: str,
        caller_phone: str = "+919876543210",
        customer_name: str = "Valued Customer",
    ) -> ConversationTurnResponse:
        """Initiate call into IVR Language Selection stage."""
        state = await self.store.get_or_create(
            session_id=session_id,
            customer_phone=caller_phone,
            customer_name=customer_name,
            language="English",
            department="SALES",
            specialist_agent=AgentRole.SALES,
            ivr_completed=False,
        )
        state["stage"] = CallLifecycleStage.IVR_LANGUAGE_SELECTION
        state["ivr_completed"] = False
        await self.store.save(session_id, state)

        logger.info(f"IVR Call started for session '{session_id}' in IVR_LANGUAGE_SELECTION.")
        return ConversationTurnResponse(
            session_id=session_id,
            message=IVR_LANGUAGE_PROMPT,
            agent_role=AgentRole.ROUTER,
            intent=UserIntent.GREETING,
            stage=CallLifecycleStage.IVR_LANGUAGE_SELECTION,
            language="English",
            department="UNASSIGNED",
            specialist_agent=AgentRole.ROUTER,
            grounded=True,
            turn_count=1,
            llm_used=False,
            resolution_status=ResolutionStatus.IN_PROGRESS,
        )

    async def initialize_direct(
        self,
        session_id: str,
        language: str = "English",
        department: str = "SALES",
        caller_phone: str = "+919876543210",
        customer_name: str = "Valued Customer",
    ) -> CallAgentState:
        """One-shot session initialization with pre-selected language and department."""
        dept_norm = department.upper().strip()
        spec_agent = AgentRole.CUSTOMER_SERVICE if "SUPPORT" in dept_norm or "SERVICE" in dept_norm else AgentRole.SALES
        dept_name = "CUSTOMER_SERVICE" if spec_agent == AgentRole.CUSTOMER_SERVICE else "SALES"

        state = await self.store.get_or_create(
            session_id=session_id,
            customer_phone=caller_phone,
            customer_name=customer_name,
            language=language.capitalize(),
            department=dept_name,
            specialist_agent=spec_agent,
            ivr_completed=True,
        )
        state["language"] = language.capitalize()
        state["department"] = dept_name
        state["specialist_agent"] = spec_agent
        state["active_agent"] = spec_agent
        state["ivr_completed"] = True
        state["stage"] = CallLifecycleStage.CONVERSATION_ACTIVE
        await self.store.save(session_id, state)

        logger.info(f"Initialized direct session '{session_id}': language={language}, department={dept_name}, specialist={spec_agent}")
        return state

    async def handle_ivr_turn(self, session_id: str, user_input: str) -> Optional[ConversationTurnResponse]:
        """Process DTMF / utterance during IVR stages. Returns None when IVR is completed."""
        state = await self.store.get(session_id)
        if not state or state.get("ivr_completed", False):
            return None

        current_stage = state.get("stage")
        cleaned = user_input.strip().lower()

        # 1. Language Selection Stage
        if current_stage == CallLifecycleStage.IVR_LANGUAGE_SELECTION:
            if "2" in cleaned or "hindi" in cleaned or "हिंदी" in cleaned:
                state["language"] = "Hindi"
                dept_prompt = IVR_DEPARTMENT_PROMPT_HI
            else:
                state["language"] = "English"
                dept_prompt = IVR_DEPARTMENT_PROMPT_EN

            state["stage"] = CallLifecycleStage.IVR_DEPARTMENT_SELECTION
            await self.store.save(session_id, state)
            logger.info(f"Session '{session_id}' IVR Language selected: {state['language']}")

            return ConversationTurnResponse(
                session_id=session_id,
                message=dept_prompt,
                agent_role=AgentRole.ROUTER,
                intent=UserIntent.GREETING,
                stage=CallLifecycleStage.IVR_DEPARTMENT_SELECTION,
                language=state["language"],
                department="UNASSIGNED",
                specialist_agent=AgentRole.ROUTER,
                grounded=True,
                turn_count=state.get("turn_count", 1) + 1,
                llm_used=False,
                resolution_status=ResolutionStatus.IN_PROGRESS,
            )

        # 2. Department Selection Stage
        if current_stage == CallLifecycleStage.IVR_DEPARTMENT_SELECTION:
            if "2" in cleaned or "support" in cleaned or "service" in cleaned or "help" in cleaned:
                state["department"] = "CUSTOMER_SERVICE"
                state["specialist_agent"] = AgentRole.CUSTOMER_SERVICE
                state["active_agent"] = AgentRole.CUSTOMER_SERVICE
            else:
                state["department"] = "SALES"
                state["specialist_agent"] = AgentRole.SALES
                state["active_agent"] = AgentRole.SALES

            state["stage"] = CallLifecycleStage.CONVERSATION_ACTIVE
            state["ivr_completed"] = True
            await self.store.save(session_id, state)
            logger.info(
                f"Session '{session_id}' IVR completed: Department={state['department']}, Specialist={state['specialist_agent']}"
            )
            return None  # Signal IVR is finished; caller should proceed directly with specialist

        return None
