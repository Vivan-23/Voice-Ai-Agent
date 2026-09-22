"""Conversation Service coordinating multi-turn state, JEV decision layer, and agent execution."""

import uuid
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from app.agents.graph import create_call_agent_graph
from app.agents.router import IntentRouter
from app.agents.state import CallAgentState
from app.interfaces.knowledge import BaseKnowledgeService
from app.interfaces.decision import BaseDecisionEngine
from app.services.decision import get_decision_engine
from app.schemas.agent import AgentRole, EscalationReason, UserIntent, CallLifecycleStage
from app.schemas.decision import DecisionAction, JevDecision
from app.schemas.conversation_state import ConversationState
from app.schemas.conversation import (
    ConversationTurnRequest,
    ConversationTurnResponse,
    ResolutionStatus,
)
from app.services.conversation.session_store import ConversationSessionStore, session_store
from config.settings import get_settings
from app.core.llm import get_chat_model
from app.core.logging import get_logger

logger = get_logger(__name__)


def _normalize_stage(stage_val: Any) -> CallLifecycleStage:
    """Safely convert any stage string, value, or enum into a valid CallLifecycleStage."""
    if isinstance(stage_val, CallLifecycleStage):
        return stage_val
    if not stage_val:
        return CallLifecycleStage.INTENT_DISCOVERY

    s = str(stage_val).strip().upper()
    if "." in s:
        s = s.split(".")[-1]

    for member in CallLifecycleStage:
        if member.name == s or member.value == s:
            return member

    mapping = {
        "TROUBLESHOOTING": CallLifecycleStage.ACTIVE_RESOLUTION,
        "RECOMMENDATION": CallLifecycleStage.ACTIVE_RESOLUTION,
        "PRODUCT_DETAILS": CallLifecycleStage.ACTIVE_RESOLUTION,
        "PRODUCT_DISCOVERY": CallLifecycleStage.INTENT_DISCOVERY,
        "PRICING": CallLifecycleStage.ACTIVE_RESOLUTION,
        "OFFER": CallLifecycleStage.ACTIVE_RESOLUTION,
        "PRODUCT_COMPARISON": CallLifecycleStage.ACTIVE_RESOLUTION,
        "CONSIDERATION": CallLifecycleStage.FOLLOW_UP,
        "RESOLUTION": CallLifecycleStage.ACTIVE_RESOLUTION,
    }
    return mapping.get(s, CallLifecycleStage.INTENT_DISCOVERY)


class ConversationService:
    """Core application service managing conversational AI turns and LangGraph orchestration."""

    def __init__(
        self,
        knowledge_service: Optional[BaseKnowledgeService] = None,
        llm: Optional[BaseChatModel] = None,
        store: Optional[ConversationSessionStore] = None,
        decision_engine: Optional[BaseDecisionEngine] = None,
        graph=None,
    ):
        self.store = store or session_store
        self.knowledge_service = knowledge_service
        self.llm = llm or get_chat_model()
        self.router = IntentRouter()
        self.decision_engine = decision_engine or get_decision_engine(get_settings())
        self._graph = graph
        if graph is None and knowledge_service is not None:
            self._graph = create_call_agent_graph(knowledge_service=knowledge_service, llm=self.llm)
        logger.info(f"Initialized ConversationService with DecisionEngine: {self.decision_engine.engine_name}.")

    def _get_graph(self):
        if self._graph is None:
            if self.knowledge_service is not None:
                self._graph = create_call_agent_graph(knowledge_service=self.knowledge_service, llm=self.llm)
            else:
                raise RuntimeError("ConversationService requires a knowledge_service or precompiled graph.")
        return self._graph

    @property
    def sessions(self) -> Dict[str, CallAgentState]:
        """Access stored sessions dictionary directly."""
        return self.store._sessions

    def get_or_create_session(
        self,
        session_id: str,
        caller_phone: Optional[str] = "+919876543210",
        customer_name: Optional[str] = "Customer",
    ) -> CallAgentState:
        """Synchronously get or initialize a session state."""
        if session_id in self.store._sessions:
            return self.store._sessions[session_id]

        initial_state: CallAgentState = {
            "session_id": session_id,
            "call_id": f"call_{uuid.uuid4().hex[:8]}",
            "customer": {
                "name": customer_name or "Customer",
                "phone": caller_phone or "+919876543210",
            },
            "customer_phone": caller_phone or "+919876543210",
            "messages": [],
            "conversation_history": [],
            "active_agent": AgentRole.CUSTOMER_SERVICE,
            "previous_agent": None,
            "current_intent": UserIntent.UNKNOWN,
            "retrieved_knowledge": [],
            "knowledge_grounded": True,
            "citations": [],
            "evidence_text": None,
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
            "latest_decision": None,
            "conversation_state": None,
        }
        self.store._sessions[session_id] = initial_state
        return initial_state

    def reset_session(self, session_id: str) -> bool:
        """Reset conversation session state."""
        return self.store.clear_sync(session_id)

    def _get_or_init_conversation_state(self, state: CallAgentState) -> ConversationState:
        """Reconstruct typed ConversationState from CallAgentState session dictionary."""
        saved = state.get("conversation_state")
        if saved and isinstance(saved, dict):
            try:
                return ConversationState(**saved)
            except Exception:
                pass

        conv = ConversationState(
            call_id=state.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
            session_id=state.get("session_id", "session_default"),
            language=(state.get("language") or "ENGLISH").upper(),
            department=(state.get("department") or "CUSTOMER_SERVICE").upper(),
            active_specialist=(state.get("active_agent") or AgentRole.CUSTOMER_SERVICE).value if hasattr(state.get("active_agent"), "value") else str(state.get("active_agent") or "CUSTOMER_SERVICE"),
            turn_count=state.get("turn_count", 0),
        )
        if state.get("active_product"):
            conv.product.active = state.get("active_product")
        if state.get("mentioned_products"):
            conv.product.mentioned = list(state.get("mentioned_products", []))
        if state.get("room_type"):
            conv.sales.room_type = state.get("room_type")
        if state.get("room_size"):
            conv.sales.room_size = state.get("room_size")
        if state.get("current_troubleshooting_step"):
            conv.issue.current_step = state.get("current_troubleshooting_step")
        if state.get("last_agent_instruction"):
            conv.conversation.last_agent_instruction = state.get("last_agent_instruction")
        if state.get("last_agent_question"):
            conv.conversation.last_agent_question = state.get("last_agent_question")
        return conv

    def _log_turn_debug(self, state: CallAgentState, user_msg: str, decision: JevDecision):
        """Format and emit debug trace matching target architecture Section 25 spec."""
        call_id = state.get("call_id", "call_unknown")
        lang = (state.get("language") or "English").upper()
        dept = (state.get("department") or "CUSTOMER_SERVICE").upper()
        active_ag = state.get("active_agent", state.get("specialist_agent", AgentRole.CUSTOMER_SERVICE))
        ag_str = (active_ag.value if hasattr(active_ag, "value") else str(active_ag)).upper()

        if decision.state_updates:
            updates_str = "\n".join(f"  {k} = {v}" for k, v in decision.state_updates.items())
        else:
            updates_str = "  none"

        llm_model = getattr(self.llm, "model_name", getattr(self.llm, "model", type(self.llm).__name__))
        knowledge_str = f"required ({decision.knowledge_query})" if decision.knowledge_required else "not required"
        manager_str = "invoked" if (decision.manager_required or decision.human_escalation_required) else "none"

        trace = (
            f"\n"
            f"CALL:\n{call_id}\n\n"
            f"LANGUAGE:\n{lang}\n\n"
            f"DEPARTMENT:\n{dept}\n\n"
            f"ACTIVE AGENT:\n{ag_str}\n\n"
            f"CUSTOMER:\n\"{user_msg}\"\n\n"
            f"DECISION_ENGINE:\n{decision.engine_used.upper()}\n\n"
            f"JEV:\n"
            f"  action = {decision.action.value}\n"
            f"  confidence = {decision.confidence:.2f}\n"
            f"  knowledge_required = {str(decision.knowledge_required).lower()}\n\n"
            f"STATE UPDATE:\n{updates_str}\n\n"
            f"LLM:\n{llm_model}\n\n"
            f"LLM_USED:\ntrue\n\n"
            f"KNOWLEDGE:\n{knowledge_str}\n\n"
            f"MANAGER:\n{manager_str}\n"
        )
        print(trace)
        logger.info(trace)

    async def process_message(self, request: ConversationTurnRequest) -> ConversationTurnResponse:
        """Process a conversational turn end-to-end through LangGraph with persistent specialist ownership."""
        session_id = request.session_id
        user_message_text = request.message.strip()

        logger.info(f"Processing turn for session '{session_id}': '{user_message_text[:60]}'...")

        # 1. Resolve / Initialize Session State (IVR routing on call start)
        existing_state = await self.store.get(session_id)
        turn_count = (existing_state.get("turn_count", 0) + 1) if existing_state else 1
        previous_intent = existing_state.get("current_intent") if existing_state else None
        intent = self.router.classify_intent(
            user_message_text,
            previous_intent=previous_intent,
            turn_count=turn_count,
        )

        if not existing_state:
            req_lang = request.language or "English"
            if request.department:
                req_dept = request.department.upper()
                default_spec = AgentRole.CUSTOMER_SERVICE if ("SUPPORT" in req_dept or "SERVICE" in req_dept) else AgentRole.SALES
            else:
                default_spec = self.router.get_target_agent_for_intent(intent)
            dept_name = "CUSTOMER_SERVICE" if default_spec == AgentRole.CUSTOMER_SERVICE else "SALES"

            state = await self.store.get_or_create(
                session_id=session_id,
                customer_phone=request.caller_phone or "+919876543210",
                customer_name=request.customer_name or "Customer",
                language=req_lang.capitalize(),
                department=dept_name,
                specialist_agent=default_spec,
                ivr_completed=True,
            )
        else:
            state = existing_state
            if request.department:
                req_dept = request.department.upper()
                spec = AgentRole.CUSTOMER_SERVICE if ("SUPPORT" in req_dept or "SERVICE" in req_dept) else AgentRole.SALES
                state["department"] = "CUSTOMER_SERVICE" if spec == AgentRole.CUSTOMER_SERVICE else "SALES"
                state["specialist_agent"] = spec
                state["active_agent"] = spec
            if request.language:
                state["language"] = request.language.capitalize()

        # Ensure persistent specialist is assigned
        if not state.get("active_agent"):
            state["active_agent"] = state.get("specialist_agent", AgentRole.SALES)

        # 2. Reconstruct or Initialize First-Class ConversationState
        conv_state = self._get_or_init_conversation_state(state)
        conv_state.turn_count = turn_count
        conv_state.conversation.last_customer_message = user_message_text
        if state.get("active_agent"):
            ag = state["active_agent"]
            conv_state.active_specialist = ag.value if hasattr(ag, "value") else str(ag)
        if state.get("department"):
            conv_state.department = state["department"]
        if state.get("language"):
            conv_state.language = state["language"]

        # 3. Execute JEV Decision Layer (System One structured evaluation)
        decision: JevDecision = await self.decision_engine.decide(conv_state, user_message_text)

        # 4. Emit Section 25 Debug Trace
        self._log_turn_debug(state, user_message_text, decision)

        # 5. Apply JEV State Updates to ConversationState and session state
        conv_state.apply_updates(decision.state_updates)
        for k, v in decision.state_updates.items():
            if k in ("active_product", "room_type", "room_size"):
                state[k] = v
            elif k == "fan_running":
                state.setdefault("troubleshooting_context", {})["fan_running"] = v
            elif k == "power_indicator":
                state.setdefault("troubleshooting_context", {})["power_indicator"] = v
            elif k == "current_step":
                state["current_troubleshooting_step"] = v

        # Maintain product mentions & comparisons
        extracted_products = self.router.extract_all_products(user_message_text)
        mentioned = state.get("mentioned_products", [])
        for prod in extracted_products:
            if prod not in mentioned:
                mentioned.append(prod)
        state["mentioned_products"] = mentioned

        is_comparison = self.router.is_comparison_query(user_message_text)
        if is_comparison or len(extracted_products) >= 2:
            state["comparison_products"] = extracted_products if len(extracted_products) >= 2 else state.get("comparison_products", [])
            state["last_product_question"] = user_message_text
            conv_state.product.comparison = list(state["comparison_products"])
        elif extracted_products:
            state["active_product"] = extracted_products[0]
            state["active_topic"] = extracted_products[0]
            state["last_product_question"] = user_message_text
            conv_state.product.active = extracted_products[0]

        # Synchronize stage and intent
        default_stage = self.router.determine_lifecycle_stage(
            intent=intent,
            turn_count=turn_count,
            previous_stage=state.get("stage"),
        )
        final_stage = _normalize_stage(decision.conversation_stage) if decision.conversation_stage else default_stage
        state["stage"] = final_stage
        conv_state.conversation.stage = final_stage.value

        if decision.customer_intent:
            try:
                intent = UserIntent(decision.customer_intent.upper())
            except ValueError:
                pass

        state["current_intent"] = intent
        state["turn_count"] = turn_count
        state["manager_intervention"] = False  # Reset per turn
        state["needs_manager"] = False  # Reset per turn
        state["manager_guidance"] = None

        # 6. Check Escalation / Handoff / Manager triggers
        if decision.human_escalation_required:
            logger.warning(f"Session {session_id}: Human escalation signaled by JEV decision.")
            state["needs_manager"] = True
            state["manager_trigger_reason"] = EscalationReason.HUMAN_REQUESTED
            state["human_escalation_required"] = True
        elif decision.customer_frustrated:
            logger.warning(f"Session {session_id}: Customer frustration signaled by JEV decision.")
            state["customer_frustrated"] = True
            state["frustration_score"] = 0.95
            state["needs_manager"] = True
            state["manager_trigger_reason"] = EscalationReason.CUSTOMER_FRUSTRATION
        elif decision.specialist_handoff_required:
            logger.info(f"Session {session_id}: Specialist handoff signaled by JEV decision.")
            state["needs_manager"] = True
            state["manager_trigger_reason"] = EscalationReason.WRONG_SPECIALIST

        # 7. Knowledge Retrieval (executed only when knowledge_required == True)
        retrieved_items: List[Dict[str, Any]] = []
        citations: List[str] = []
        is_grounded: bool = True
        evidence_text: str = ""

        if decision.knowledge_required and decision.knowledge_query and self.knowledge_service:
            logger.info(f"Knowledge required by JEV. Querying: '{decision.knowledge_query}'")
            try:
                k_res = await self.knowledge_service.query(decision.knowledge_query, session_id=session_id)
                if k_res and k_res.is_grounded and k_res.answer and len(k_res.answer.strip()) >= 5:
                    evidence_text = k_res.answer.strip()
                    retrieved_items = [item.model_dump() for item in k_res.items]
                    citations = [item.source_title for item in k_res.items if item.source_title]
                    is_grounded = True
                else:
                    is_grounded = False
                    state["needs_manager"] = True
                    state["manager_trigger_reason"] = EscalationReason.UNKNOWN_ANSWER
            except Exception as ex:
                logger.warning(f"Knowledge query failed: {ex}")
                is_grounded = False

        state["retrieved_knowledge"] = retrieved_items
        state["citations"] = citations
        state["knowledge_grounded"] = is_grounded
        if evidence_text:
            state["evidence_text"] = evidence_text

        # 8. Store decision and first-class conversation state into CallAgentState
        state["latest_decision"] = decision.model_dump()
        state["conversation_state"] = conv_state.to_summary_dict()

        # 9. Append User Message
        state["messages"].append(HumanMessage(content=user_message_text))
        state["conversation_history"].append({
            "role": "user",
            "content": user_message_text,
            "intent": intent.value,
            "stage": state.get("stage").value if hasattr(state.get("stage"), "value") else str(state.get("stage")),
        })

        # 10. Execute Multi-Agent Graph (Directly executes persistent specialist node)
        graph = self._get_graph()
        updated_state = await graph.ainvoke(state)

        # 11. Extract Final Response and Status
        response_text = updated_state.get("last_response")
        if not response_text:
            msgs = updated_state.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    response_text = str(m.content)
                    break

        if not response_text:
            response_text = "I am checking this for you. How else may I assist with your Coway air purifier?"

        active_agent = updated_state.get("active_agent", state.get("specialist_agent", AgentRole.SALES))
        citations = updated_state.get("citations", citations)
        is_grounded = updated_state.get("knowledge_grounded", is_grounded)
        manager_intervened = updated_state.get("manager_intervention", False)
        manager_guidance = updated_state.get("manager_guidance")
        is_escalated = updated_state.get("human_escalation_required", False)

        # Update ConversationState with agent response and next expected input
        conv_state.conversation.last_agent_message = response_text
        if "?" in response_text:
            updated_state["last_agent_question"] = response_text
            conv_state.conversation.last_agent_question = response_text
        if decision.next_expected_input:
            conv_state.conversation.awaiting_answer_to = decision.next_expected_input
        updated_state["conversation_state"] = conv_state.to_summary_dict()

        updated_state["conversation_history"].append({
            "role": "assistant",
            "agent": active_agent.value if hasattr(active_agent, "value") else str(active_agent),
            "content": response_text,
            "citations": citations,
            "grounded": is_grounded,
            "manager_intervened": manager_intervened,
        })

        # Persist updated state
        await self.store.save(session_id, updated_state)

        # 12. Build Clean API Response
        resolution = (
            ResolutionStatus.ESCALATED_HUMAN
            if is_escalated
            else ResolutionStatus.IN_PROGRESS
        )

        return ConversationTurnResponse(
            session_id=session_id,
            message=response_text,
            agent_role=active_agent,
            intent=intent,
            stage=_normalize_stage(updated_state.get("stage", final_stage)),
            language=updated_state.get("language", "English"),
            department=updated_state.get("department", "SALES"),
            specialist_agent=updated_state.get("specialist_agent", active_agent),
            grounded=is_grounded,
            citations=citations,
            manager_intervention=manager_intervened,
            manager_guidance=manager_guidance,
            human_escalation_required=is_escalated,
            resolution_status=resolution,
            turn_count=updated_state.get("turn_count", 1),
            llm_used=updated_state.get("llm_used", True),
            llm_model=updated_state.get("llm_model") or getattr(self.llm, "model_name", getattr(self.llm, "model", type(self.llm).__name__)),
            knowledge_retrieved=bool(updated_state.get("retrieved_knowledge")),
        )

    # Alias for process_message
    async def process_turn(self, request: ConversationTurnRequest) -> ConversationTurnResponse:
        return await self.process_message(request)
