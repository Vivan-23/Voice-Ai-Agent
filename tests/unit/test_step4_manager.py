"""Unit and Behavioral Tests for Step 4: Manager & Escalation.

Verifies:
1. TEST 1: Normal conversation does NOT invoke Manager.
2. TEST 2: Explicit human request invokes Manager -> ESCALATE_TO_HUMAN.
3. TEST 3: Unresolved support issue invokes Manager.
4. TEST 4: Specialist handoff (Sales -> Support) retains full conversation history.
5. TEST 5: Conversation continuity after handoff (no duplicate questions asked).
"""

import pytest
from typing import List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.llm.gemini import GeminiClient, LLMCallResult
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.agents.manager import ManagerAgent, ManagerAction, ManagerDecision
from app.agents.base_agent import BaseAgent, AgentResponse
from app.conversation.conversation import Conversation
from app.knowledge.mock import MockKnowledgeProvider


class MockManagerAwareLLM(GeminiClient):
    """Accurately simulates Gemini tool-calling, knowledge retrieval, and manager intervention triggers."""

    def __init__(self, knowledge_provider=None, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._is_configured = True
        self.knowledge_provider = knowledge_provider or MockKnowledgeProvider()
        self.call_history: List[List[BaseMessage]] = []

    async def generate_with_manager(
        self,
        messages: List[BaseMessage],
        system_prompt: str,
    ) -> AgentResponse:
        self.call_history.append(messages)
        user_msgs = [m.content for m in messages if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage"]
        latest_user = user_msgs[-1] if user_msgs else ""
        latest_lower = latest_user.lower()
        all_user_text = " ".join(user_msgs).lower()

        # Check for Manager Intervention Triggers (Simulating LLM tool call request_manager_assistance)
        # Trigger A: Explicit human request
        if any(phrase in latest_lower for phrase in ["senior executive", "talk to a human", "speak to a human", "human representative", "speak to someone", "someone real"]):
            return AgentResponse(
                text="",
                agent_name="Sarah" if "sales" in system_prompt.lower() else "Alex",
                department="SALES" if "sales" in system_prompt.lower() else "CUSTOMER_SUPPORT",
                llm_result=LLMCallResult(content="", model=self.model_name, success=True),
                manager_invoked=True,
                manager_reason="Customer requested human representative or senior executive.",
            )

        # Trigger B: Cross-specialist handoff (Sales customer has broken unit)
        if "sales" in system_prompt.lower() and ("bought an airmega" in all_user_text or "own an airmega" in all_user_text or "isn't working" in all_user_text or "broken" in all_user_text or "repair" in all_user_text):
            if "not working" in latest_lower or "isn't working" in latest_lower or "fan isn't running" in latest_lower:
                return AgentResponse(
                    text="",
                    agent_name="Sarah",
                    department="SALES",
                    llm_result=LLMCallResult(content="", model=self.model_name, success=True),
                    manager_invoked=True,
                    manager_reason="Customer has an existing unit requiring technical service support.",
                )

        # Trigger C: Unresolved support issue after troubleshooting
        if "support" in system_prompt.lower() and ("still doesn't work" in latest_lower or "still does not work" in latest_lower or "tried that and it still" in latest_lower):
            return AgentResponse(
                text="",
                agent_name="Alex",
                department="CUSTOMER_SUPPORT",
                llm_result=LLMCallResult(content="", model=self.model_name, success=True),
                manager_invoked=True,
                manager_reason="Customer completed basic troubleshooting steps but unit remains non-functional.",
            )

        # Normal turns (Knowledge or Direct General Conversation)
        knowledge_called = False
        knowledge_query = None

        if "sales" in system_prompt.lower():
            if "what products" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway India air purifier product models"
                spoken = "Coway sells our signature Airmega air purifiers in India, including the Airmega 150 for bedrooms up to 355 sqft (INR 16,999) and the Airmega 250 for larger living rooms up to 600 sqft (INR 34,999)."
            elif "price of airmega 250" in latest_lower or "price" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Airmega 250 price INR"
                spoken = "The Airmega 250 is listed at INR 34,999 with free doorstep delivery across India."
            elif "coverage" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Airmega 250 specifications and coverage"
                spoken = "The Airmega 250 covers rooms up to 600 square feet with multi-directional airflow."
            elif "bought an airmega 150" in latest_lower:
                spoken = "I see that you purchased an Airmega 150. How can I assist you with it today?"
            else:
                spoken = "I'd be happy to help you with Coway products."

        # Support Agent Responses (including continuation after handoff)
        else:
            if "fan isn't running" in all_user_text and ("bought" in all_user_text or "own" in all_user_text):
                # Continuation acknowledging history without asking product again
                spoken = "Hi, I'm Alex from Coway Customer Support. I understand from your previous message that your Airmega 150 fan is not running. Let's check if the front panel safety interlock is firmly engaged."
            elif "isn't working" in latest_lower or "is not working" in latest_lower:
                spoken = "I understand. Let's check a few things for your Airmega 150. Is the power indicator light on?"
            elif "fan isn't running" in latest_lower or "fan is not running" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Airmega fan silent front cover safety interlock troubleshooting"
                spoken = "Since the power is on but the fan is silent, please remove the front panel, verify the pre-filter is locked firmly in place, and snap the front panel shut so the safety interlock switch engages."
            else:
                spoken = "I can help troubleshoot your Coway purifier."

        return AgentResponse(
            text=spoken,
            agent_name="Sarah" if "sales" in system_prompt.lower() else "Alex",
            department="SALES" if "sales" in system_prompt.lower() else "CUSTOMER_SUPPORT",
            llm_result=LLMCallResult(content=spoken, model=self.model_name, success=True),
            knowledge_retrieved=knowledge_called,
            knowledge_query=knowledge_query,
            knowledge_provider_name="MOCK",
            manager_invoked=False,
        )


@pytest.mark.asyncio
async def test_1_normal_conversation_does_not_invoke_manager():
    """Test 1: Normal multi-turn conversation proceeds without invoking the Manager."""
    mock_llm = MockManagerAwareLLM()
    knowledge_provider = MockKnowledgeProvider()
    sales_agent = SalesAgent(llm_client=mock_llm, knowledge_provider=knowledge_provider)
    conv = Conversation(department="SALES", active_agent=sales_agent, knowledge_provider=knowledge_provider)

    # Turn 1
    r1 = await conv.process_message("What products does Coway sell?")
    assert r1.manager_invoked is False
    assert "Airmega 150" in r1.text

    # Turn 2
    r2 = await conv.process_message("What is the price of Airmega 250?")
    assert r2.manager_invoked is False
    assert "34,999" in r2.text

    # Turn 3
    r3 = await conv.process_message("What is its coverage?")
    assert r3.manager_invoked is False
    assert "600" in r3.text


@pytest.mark.asyncio
async def test_2_explicit_human_request_invokes_manager_escalation():
    """Test 2: Explicit human request triggers Manager -> ESCALATE_TO_HUMAN."""
    mock_llm = MockManagerAwareLLM()
    knowledge_provider = MockKnowledgeProvider()
    sales_agent = SalesAgent(llm_client=mock_llm, knowledge_provider=knowledge_provider)
    conv = Conversation(department="SALES", active_agent=sales_agent, knowledge_provider=knowledge_provider)

    r = await conv.process_message("I want to speak to a senior executive.")
    assert r.manager_invoked is True
    assert r.manager_action == "ESCALATE_TO_HUMAN"
    assert "escalated to a human executive" in r.text.lower()
    assert conv.active is False


@pytest.mark.asyncio
async def test_3_unresolved_support_issue_invokes_manager():
    """Test 3: Unresolved support issue after troubleshooting triggers Manager intervention."""
    mock_llm = MockManagerAwareLLM()
    knowledge_provider = MockKnowledgeProvider()
    support_agent = CustomerSupportAgent(llm_client=mock_llm, knowledge_provider=knowledge_provider)
    conv = Conversation(department="CUSTOMER_SUPPORT", active_agent=support_agent, knowledge_provider=knowledge_provider)

    await conv.process_message("I own an Airmega 150.")
    await conv.process_message("It isn't working.")
    await conv.process_message("The fan isn't running.")
    r4 = await conv.process_message("I tried that and it still doesn't work.")

    assert r4.manager_invoked is True
    assert r4.manager_action in ("ESCALATE_TO_HUMAN", "CONTINUE_GUIDE")


@pytest.mark.asyncio
async def test_4_specialist_handoff_sales_to_support_preserves_history():
    """Test 4: Customer reaching Sales with broken unit is assigned to Support without losing context."""
    mock_llm = MockManagerAwareLLM()
    knowledge_provider = MockKnowledgeProvider()
    sales_agent = SalesAgent(llm_client=mock_llm, knowledge_provider=knowledge_provider)
    conv = Conversation(department="SALES", active_agent=sales_agent, knowledge_provider=knowledge_provider)

    # Customer enters Sales department but reports broken unit
    r = await conv.process_message("I bought an Airmega 150 and it isn't working.")

    assert r.manager_invoked is True
    assert r.manager_action == "ASSIGN_AGENT"
    assert r.assigned_agent == "CUSTOMER_SUPPORT"
    assert conv.department == "CUSTOMER_SUPPORT"
    assert conv.active_agent.name == "Alex"
    # Verify history is preserved across handoff
    assert len(conv.history) >= 2


@pytest.mark.asyncio
async def test_5_conversation_continuity_after_handoff():
    """Test 5: Support specialist continues smoothly after handoff without asking repetitive questions."""
    mock_llm = MockManagerAwareLLM()
    knowledge_provider = MockKnowledgeProvider()
    sales_agent = SalesAgent(llm_client=mock_llm, knowledge_provider=knowledge_provider)
    conv = Conversation(department="SALES", active_agent=sales_agent, knowledge_provider=knowledge_provider)

    # Turn 1 in Sales
    r1 = await conv.process_message("I bought an Airmega 150 last month.")
    assert conv.department == "SALES"

    # Turn 2 in Sales triggers handoff to Support
    r2 = await conv.process_message("The fan isn't running.")
    assert r2.manager_invoked is True
    assert r2.manager_action == "ASSIGN_AGENT"
    assert conv.department == "CUSTOMER_SUPPORT"
    assert conv.active_agent.name == "Alex"

    # Verify that Alex knows from conversation history that customer has Airmega 150 and fan isn't running
    assert "Airmega 150" in r2.text or "fan" in r2.text
    assert "what product do you have" not in r2.text.lower()
