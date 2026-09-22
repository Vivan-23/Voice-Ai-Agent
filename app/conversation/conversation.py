"""Conversation Session and Multi-turn Management for Voice AI Platform POC (Step 4: Manager & Escalation)."""

import uuid
from typing import List, Dict, Optional
from app.agents.base_agent import BaseAgent, AgentResponse
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.agents.manager import ManagerAgent, ManagerAction, ManagerDecision
from app.knowledge.base import BaseKnowledgeProvider
from app.knowledge.notebooklm import NotebookLMKnowledgeProvider
from app.llm.gemini import LLMCallResult


class Conversation:
    """Represents an ongoing conversational session with specialist agents, conversation history, and manager escalation."""

    def __init__(
        self,
        department: str,
        session_id: Optional[str] = None,
        active_agent: Optional[BaseAgent] = None,
        knowledge_provider: Optional[BaseKnowledgeProvider] = None,
        manager_agent: Optional[ManagerAgent] = None,
    ):
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.department = department.upper()
        self.history: List[Dict[str, str]] = []
        self.active: bool = True
        self.knowledge_provider = knowledge_provider or NotebookLMKnowledgeProvider()

        if active_agent is not None:
            self.active_agent = active_agent
            if self.active_agent.knowledge_provider is None:
                self.active_agent.knowledge_provider = self.knowledge_provider
        elif self.department == "SALES":
            self.active_agent = SalesAgent(knowledge_provider=self.knowledge_provider)
        elif self.department in ("CUSTOMER_SUPPORT", "SUPPORT", "SERVICE"):
            self.active_agent = CustomerSupportAgent(knowledge_provider=self.knowledge_provider)
        else:
            self.active_agent = SalesAgent(knowledge_provider=self.knowledge_provider)

        self.manager = manager_agent or ManagerAgent(llm_client=self.active_agent.llm_client)

    def get_initial_greeting(self) -> str:
        """Return the specialist agent's greeting and record it in conversation history."""
        greeting = getattr(self.active_agent, "get_greeting", lambda: f"Hi, I'm {self.active_agent.name} from Coway. How can I help you today?")()
        self.history.append({"role": "assistant", "content": greeting})
        return greeting

    async def process_message(self, customer_message: str) -> AgentResponse:
        """Process a customer utterance in context of conversation history and coordinate manager escalation if needed."""
        # 1. Ask active specialist agent to respond using full conversation history and grounded knowledge
        initial_response: AgentResponse = await self.active_agent.respond(
            customer_message=customer_message,
            history=self.history,
        )

        response = initial_response

        # 2. Check if Manager Intervention was requested
        if initial_response.manager_invoked:
            decision: ManagerDecision = await self.manager.decide(
                current_agent=self.active_agent.department,
                conversation_history=self.history,
                current_customer_message=customer_message,
                reason=initial_response.manager_reason,
            )

            # Outcome A: ESCALATE_TO_HUMAN
            if decision.action == ManagerAction.ESCALATE_TO_HUMAN:
                escalation_text = "This conversation has been escalated to a human executive."
                response = AgentResponse(
                    text=escalation_text,
                    agent_name=self.active_agent.name,
                    department=self.active_agent.department,
                    llm_result=LLMCallResult(
                        content=escalation_text,
                        model=self.active_agent.llm_client.model_name,
                        success=True,
                    ),
                    knowledge_retrieved=False,
                    manager_invoked=True,
                    manager_action="ESCALATE_TO_HUMAN",
                    assigned_agent=None,
                    history_preserved=True,
                )
                self.active = False

            # Outcome B: ASSIGN_AGENT (Cross-specialist handoff with full history preservation)
            elif decision.action == ManagerAction.ASSIGN_AGENT:
                target_dept = (decision.assigned_agent or "CUSTOMER_SUPPORT").upper()
                if "SUPPORT" in target_dept or "SERVICE" in target_dept:
                    self.active_agent = CustomerSupportAgent(
                        llm_client=self.active_agent.llm_client,
                        knowledge_provider=self.knowledge_provider,
                    )
                    self.department = "CUSTOMER_SUPPORT"
                else:
                    self.active_agent = SalesAgent(
                        llm_client=self.active_agent.llm_client,
                        knowledge_provider=self.knowledge_provider,
                    )
                    self.department = "SALES"

                # Newly assigned specialist continues seamlessly with full existing history
                handoff_resp = await self.active_agent.respond(
                    customer_message=customer_message,
                    history=self.history,
                )
                handoff_resp.manager_invoked = True
                handoff_resp.manager_action = "ASSIGN_AGENT"
                handoff_resp.assigned_agent = self.department
                handoff_resp.history_preserved = True
                response = handoff_resp

            # Outcome C: CONTINUE_GUIDE (Specialist continues with internal guidance)
            elif decision.action == ManagerAction.CONTINUE_GUIDE:
                guided_resp = await self.active_agent.respond(
                    customer_message=customer_message,
                    history=self.history,
                    guidance=decision.internal_guidance,
                )
                guided_resp.manager_invoked = True
                guided_resp.manager_action = "CONTINUE_GUIDE"
                guided_resp.history_preserved = True
                response = guided_resp

        # 3. Append turns to conversation history
        self.history.append({"role": "user", "content": customer_message})
        self.history.append({"role": "assistant", "content": response.text})

        return response
