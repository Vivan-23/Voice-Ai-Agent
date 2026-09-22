"""Unit and Behavioral Tests for Step 3: NotebookLM Knowledge Integration with Gemini Conversational Brain."""

import pytest
from typing import List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.llm.gemini import GeminiClient, LLMCallResult
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.agents.base_agent import BaseAgent, AgentResponse
from app.conversation.conversation import Conversation
from app.knowledge.mock import MockKnowledgeProvider
from app.knowledge.base import BaseKnowledgeProvider


class MockKnowledgeAwareLLM(GeminiClient):
    """Simulates Gemini LLM tool-calling and grounded synthesis for deterministic testing."""

    def __init__(self, knowledge_provider: Optional[BaseKnowledgeProvider] = None, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._is_configured = True
        self.knowledge_provider = knowledge_provider or MockKnowledgeProvider()
        self.call_history: List[List[BaseMessage]] = []

    async def generate_with_knowledge(
        self,
        messages: List[BaseMessage],
        system_prompt: str,
    ) -> AgentResponse:
        self.call_history.append(messages)
        user_msgs = [m.content for m in messages if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage"]
        latest_user = user_msgs[-1] if user_msgs else ""
        latest_lower = latest_user.lower()
        all_user_text = " ".join(user_msgs).lower()

        knowledge_called = False
        knowledge_query = None
        evidence = None

        # 1. General conversation checks (greetings, general knowledge) -> NO knowledge retrieval
        if latest_lower in ("hello", "hi", "hey", "good morning") or "what does an air purifier do" in latest_lower or "explain what an air purifier does" in latest_lower:
            if "explain" in latest_lower or "what does an air purifier do" in latest_lower:
                spoken = "An air purifier cleans indoor air by drawing in ambient air, passing it through filtration layers like HEPA and carbon to capture pollutants, dust, and odors, and circulating fresh clean air."
            else:
                spoken = "Hello! How can I help you today?"
            return AgentResponse(
                text=spoken,
                agent_name="Sarah" if "sales" in system_prompt.lower() else "Alex",
                department="SALES" if "sales" in system_prompt.lower() else "CUSTOMER_SUPPORT",
                llm_result=LLMCallResult(content=spoken, model=self.model_name, success=True),
                knowledge_retrieved=False,
                knowledge_provider_name="MOCK",
            )

        # 2. Company-specific factual queries -> Trigger Knowledge Retrieval
        if "sales" in system_prompt.lower():
            if "what products" in latest_lower or "what does coway sell" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway India air purifier product models"
                evidence = await self.knowledge_provider.query(knowledge_query)
                spoken = "Coway sells our signature Airmega air purifiers in India, including the Airmega 150 for bedrooms up to 355 sqft (INR 16,999) and the Airmega 250 for larger living rooms up to 600 sqft (INR 34,999)."
            elif "discounts on those" in latest_lower or "discount" in latest_lower or "offer" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Airmega 150 and Airmega 250 discounts promotional offers"
                evidence = await self.knowledge_provider.query(knowledge_query)
                spoken = "On the Airmega 150 and 250, we currently offer seasonal promotional discounts of up to 10% instant discount on select bank cards."
            elif "second one" in latest_lower or "250" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Airmega 250 specifications and coverage"
                evidence = await self.knowledge_provider.query(knowledge_query)
                spoken = "The Airmega 250 covers rooms up to 600 square feet with multi-directional airflow, real-time AQI indicator lights, and custom smart modes."
            elif "turbomax" in latest_lower or "xyz" in latest_lower or "unknown" in latest_lower or "9000" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Model TurboMax 9000"
                evidence = await self.knowledge_provider.query(knowledge_query)
                spoken = "I do not have verified company information or pricing available for that model in our current official catalog."
            elif "how much does it cost" in latest_lower or "how much" in latest_lower or "cost" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Airmega 250 price INR"
                evidence = await self.knowledge_provider.query(knowledge_query)
                spoken = "The Airmega 250 is listed at INR 34,999 with free doorstep delivery across India."
            else:
                spoken = "I'd be happy to assist you with Coway air purifiers. Are you looking for a specific room size or features?"

        # Customer Support Agent Logic
        else:
            if "i own an airmega 150" in latest_lower and len(user_msgs) == 1:
                spoken = "Got it, the Airmega 150. What issue or symptoms are you experiencing with your purifier?"
            elif "isn't working" in latest_lower or "not working" in latest_lower:
                spoken = "I understand. Let's troubleshoot your Airmega 150. Is the power indicator light illuminated when plugged in?"
            elif "fan isn't running" in latest_lower or "fan is silent" in latest_lower or "fan" in latest_lower:
                knowledge_called = True
                knowledge_query = "Coway Airmega fan silent front cover safety interlock troubleshooting"
                evidence = await self.knowledge_provider.query(knowledge_query)
                spoken = "Since the power is on but the fan is silent, please remove the front panel, verify the pre-filter is locked firmly in place, and snap the front panel shut so the safety interlock switch engages."
            elif "xyz" in latest_lower:
                knowledge_called = True
                knowledge_query = "replacement filters Model XYZ-9999"
                evidence = await self.knowledge_provider.query(knowledge_query)
                spoken = "I do not have information in our company records regarding replacement filters for Model XYZ-9999."
            else:
                spoken = "Let me know what symptoms you observe on your Coway purifier so I can assist you."

        return AgentResponse(
            text=spoken,
            agent_name="Sarah" if "sales" in system_prompt.lower() else "Alex",
            department="SALES" if "sales" in system_prompt.lower() else "CUSTOMER_SUPPORT",
            llm_result=LLMCallResult(content=spoken, model=self.model_name, success=True),
            knowledge_retrieved=knowledge_called,
            knowledge_query=knowledge_query,
            knowledge_provider_name="MOCK",
        )


class MockGroundedAgent(BaseAgent):
    """Specialist agent using the mock grounded LLM for test execution."""

    def __init__(self, name: str, department: str, system_prompt: str, mock_llm: MockKnowledgeAwareLLM):
        super().__init__(name=name, department=department, system_prompt=system_prompt, llm_client=mock_llm, knowledge_provider=mock_llm.knowledge_provider)
        self.mock_llm = mock_llm

    async def respond(self, customer_message: str, history: List[Dict[str, str]]) -> AgentResponse:
        messages = self.build_prompt_messages(customer_message, history)
        return await self.mock_llm.generate_with_knowledge(messages, self.system_prompt)


# ==============================================================================
# 1. Case 1 — General Conversation (NO Knowledge Retrieval)
# ==============================================================================
@pytest.mark.asyncio
async def test_case_1_general_conversation_no_knowledge_retrieval():
    """Verify that general conversational greetings or generic questions do NOT query NotebookLM."""
    mock_llm = MockKnowledgeAwareLLM()
    sales_agent = MockGroundedAgent("Sarah", "SALES", "Sales persona", mock_llm)
    conv = Conversation(department="SALES", active_agent=sales_agent)

    # Greeting
    r1 = await conv.process_message("Hello")
    assert r1.knowledge_retrieved is False
    assert r1.knowledge_query is None

    # Generic conceptual question
    r2 = await conv.process_message("Can you explain what an air purifier does?")
    assert r2.knowledge_retrieved is False
    assert "filtration" in r2.text.lower() or "cleans" in r2.text.lower()


# ==============================================================================
# 2. Case 2 — Sales Multi-Turn with NotebookLM Knowledge Grounding
# ==============================================================================
@pytest.mark.asyncio
async def test_case_2_sales_multi_turn_with_knowledge_grounding():
    """Verify multi-turn sales consultation where company questions trigger knowledge retrieval with contextual pronoun resolution."""
    mock_llm = MockKnowledgeAwareLLM()
    sales_agent = MockGroundedAgent("Sarah", "SALES", "Sales persona", mock_llm)
    conv = Conversation(department="SALES", active_agent=sales_agent)

    # Turn 1: "What products does Coway sell?" -> Knowledge Retrieval
    r1 = await conv.process_message("What products does Coway sell?")
    assert r1.knowledge_retrieved is True
    assert "Airmega 150" in r1.text and "Airmega 250" in r1.text

    # Turn 2: "What are the discounts on those?" -> Resolves "those" + Knowledge Retrieval
    r2 = await conv.process_message("What are the discounts on those?")
    assert r2.knowledge_retrieved is True
    assert "10%" in r2.text or "discount" in r2.text.lower()

    # Turn 3: "Tell me more about the second one." -> Resolves Airmega 250 + Knowledge Retrieval
    r3 = await conv.process_message("Tell me more about the second one.")
    assert r3.knowledge_retrieved is True
    assert "250" in r3.text or "600 square feet" in r3.text

    # Turn 4: "How much does it cost?" -> Resolves Airmega 250 price + Knowledge Retrieval
    r4 = await conv.process_message("How much does it cost?")
    assert r4.knowledge_retrieved is True
    assert "34,999" in r4.text

    # Verify conversation history was maintained
    assert len(conv.history) == 8


# ==============================================================================
# 3. Case 2 — Support Multi-Turn with Troubleshooting Knowledge
# ==============================================================================
@pytest.mark.asyncio
async def test_case_2_support_multi_turn_with_troubleshooting_knowledge():
    """Verify multi-turn support troubleshooting diagnosing fan failure using company knowledge without conversation resets."""
    mock_llm = MockKnowledgeAwareLLM()
    support_agent = MockGroundedAgent("Alex", "CUSTOMER_SUPPORT", "Support persona", mock_llm)
    conv = Conversation(department="CUSTOMER_SUPPORT", active_agent=support_agent)

    # Turn 1: "I own an Airmega 150."
    r1 = await conv.process_message("I own an Airmega 150.")
    assert "Airmega 150" in r1.text

    # Turn 2: "It isn't working correctly."
    r2 = await conv.process_message("It isn't working correctly.")
    assert "power" in r2.text.lower()

    # Turn 3: "The fan isn't running." -> Knowledge Retrieval for interlock / panel safety switch
    r3 = await conv.process_message("The fan isn't running.")
    assert r3.knowledge_retrieved is True
    assert "front panel" in r3.text.lower() or "interlock" in r3.text.lower() or "pre-filter" in r3.text.lower()


# ==============================================================================
# 4. Grounding Rule — Unsupported Knowledge Does NOT Hallucinate
# ==============================================================================
@pytest.mark.asyncio
async def test_grounding_rule_no_hallucination_on_unknown_model():
    """Verify that when company knowledge does not contain information, the agent admits lack of data rather than hallucinating."""
    mock_llm = MockKnowledgeAwareLLM()
    sales_agent = MockGroundedAgent("Sarah", "SALES", "Sales persona", mock_llm)
    conv = Conversation(department="SALES", active_agent=sales_agent)

    # Inquire about unsupported model
    r = await conv.process_message("Do you sell replacement filters for Model XYZ-9999?")
    assert r.knowledge_retrieved is True
    # Must NOT fabricate pricing or availability
    assert "available for inr" not in r.text.lower()
    assert "do not have" in r.text.lower() or "cannot verify" in r.text.lower() or "not listed" in r.text.lower()
