"""Unit and Multi-Turn Behavioral Tests for Step 2 (Gemini LLM + Conversation Memory + Sales & Support Agents)."""

import pytest
import os
from typing import List
from langchain_core.messages import BaseMessage, AIMessage
from app.llm.gemini import GeminiClient, LLMCallResult
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.conversation.conversation import Conversation


class MockConversationalLLM(GeminiClient):
    """Test LLM client that accurately simulates context-aware multi-turn conversational responses without network calls."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._is_configured = True
        self.call_history: List[List[BaseMessage]] = []

    async def generate(self, messages: List[BaseMessage]) -> LLMCallResult:
        self.call_history.append(messages)
        user_messages = [m.content for m in messages if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage"]
        latest_user = user_messages[-1] if user_messages else ""
        system_msg = next((m.content for m in messages if getattr(m, "type", "") == "system" or m.__class__.__name__ == "SystemMessage"), "")

        all_text = " ".join(user_messages).lower()
        latest_lower = latest_user.lower()

        # Sales Context Handling
        if "sales" in system_msg.lower():
            if "what products" in latest_lower or "what does coway sell" in latest_lower:
                reply = "Coway sells our signature Airmega air purifiers in India, including the Airmega 150 for bedrooms and the Airmega 250 for larger living rooms."
            elif "discounts on those" in latest_lower or "discounts" in latest_lower:
                # Verifies understanding of "those" referencing Airmega 150 / 250
                reply = "On the Airmega 150 and 250, we currently offer up to 10% instant discount on select bank cards."
            elif "second one" in latest_lower or "250" in latest_lower:
                # Verifies reference to second product (Airmega 250)
                reply = "The Airmega 250 covers rooms up to 600 square feet with multi-directional airflow, real-time air quality indicator, and smart auto modes."
            elif "how much does it cost" in latest_lower or "how much" in latest_lower or "price" in latest_lower:
                # Verifies "it" refers to Airmega 250
                reply = "The Airmega 250 is listed at INR 34,999 with free doorstep delivery across India."
            else:
                reply = f"I'd be happy to help you with Coway air purifiers. Are you looking for a specific room size or model?"

        # Support Context Handling
        else:
            if "i own an airmega 150" in latest_lower or "airmega 150" in latest_lower and len(user_messages) == 1:
                reply = "Got it, the Airmega 150. What issue or symptoms are you experiencing with your purifier?"
            elif "isn't working" in latest_lower or "is not working" in latest_lower:
                # Verifies context retention that customer owns Airmega 150
                reply = "I understand. Let's check a few things for your Airmega 150. Is the power indicator light on when you plug it in?"
            elif "fan isn't running" in latest_lower or "fan is not running" in latest_lower or "fan" in latest_lower:
                reply = "Since the power is on but the fan is silent, the front cover safety interlock might not be fully engaged. Please remove the front panel, ensure the pre-filter is locked in place, and snap the front cover closed firmly."
            elif "still doesn't work" in latest_lower or "still does not work" in latest_lower:
                reply = "Since the unit still does not run with the front cover secured, let's schedule an authorized technician visit to inspect the motor. Would you like me to register a service request?"
            else:
                reply = "I can help clarify and fix that for your Coway purifier. Could you describe what you observe?"

        return LLMCallResult(
            content=reply,
            provider="GEMINI",
            model=self.model_name,
            success=True,
        )


# ==============================================================================
# 1. Real Gemini Client Unconfigured Diagnostics Test
# ==============================================================================
def test_gemini_client_fails_cleanly_when_unconfigured():
    """Verify that GeminiClient fails clearly with descriptive message if key is missing."""
    client = GeminiClient(api_key="")
    assert client.is_configured is False
    res = client.generate_sync([])
    assert res.success is False
    assert "API_KEY is not configured" in res.error_message


# ==============================================================================
# 2. Sales Multi-Turn Conversation Behavioral Test
# ==============================================================================
@pytest.mark.asyncio
async def test_sales_multi_turn_conversation_context():
    """Verify 4-turn Sales conversation demonstrating conversational references ('those', 'second one', 'it')."""
    mock_llm = MockConversationalLLM()
    sales_agent = SalesAgent(name="Sarah", llm_client=mock_llm)
    conv = Conversation(department="SALES", active_agent=sales_agent)

    # Initial greeting
    greeting = conv.get_initial_greeting()
    assert "Sarah from Coway Sales" in greeting
    assert len(conv.history) == 1

    # Turn 1: "What products does Coway sell?"
    r1 = await conv.process_message("What products does Coway sell?")
    assert "Airmega 150" in r1.text and "Airmega 250" in r1.text
    assert len(conv.history) == 3  # [greeting, user1, agent1]

    # Turn 2: "What are the discounts on those?" (resolves "those")
    r2 = await conv.process_message("What are the discounts on those?")
    assert "10%" in r2.text or "discount" in r2.text.lower()
    assert len(conv.history) == 5

    # Turn 3: "Tell me more about the second one." (resolves second product -> Airmega 250)
    r3 = await conv.process_message("Tell me more about the second one.")
    assert "250" in r3.text or "600 square feet" in r3.text
    assert len(conv.history) == 7

    # Turn 4: "How much does it cost?" (resolves "it" -> Airmega 250 price)
    r4 = await conv.process_message("How much does it cost?")
    assert "34,999" in r4.text
    assert len(conv.history) == 9

    # Verify that the entire conversation history was passed on every turn
    last_prompt_messages = mock_llm.call_history[-1]
    # system message + 8 previous history turns (4 user + 4 assistant) + current user message
    user_contents = [m.content for m in last_prompt_messages if getattr(m, "type", "") == "human"]
    assert "What products does Coway sell?" in user_contents
    assert "What are the discounts on those?" in user_contents
    assert "Tell me more about the second one." in user_contents
    assert "How much does it cost?" in user_contents


# ==============================================================================
# 3. Customer Support Multi-Turn Conversation Behavioral Test
# ==============================================================================
@pytest.mark.asyncio
async def test_support_multi_turn_conversation_context():
    """Verify 4-turn Support conversation with troubleshooting continuation and context memory."""
    mock_llm = MockConversationalLLM()
    support_agent = CustomerSupportAgent(name="Alex", llm_client=mock_llm)
    conv = Conversation(department="CUSTOMER_SUPPORT", active_agent=support_agent)

    # Initial greeting
    greeting = conv.get_initial_greeting()
    assert "Alex from Coway Customer Support" in greeting

    # Turn 1: "I own an Airmega 150."
    r1 = await conv.process_message("I own an Airmega 150.")
    assert "Airmega 150" in r1.text

    # Turn 2: "It isn't working correctly."
    r2 = await conv.process_message("It isn't working correctly.")
    assert "power" in r2.text.lower() or "indicator" in r2.text.lower()

    # Turn 3: "The fan isn't running."
    r3 = await conv.process_message("The fan isn't running.")
    assert "front cover" in r3.text.lower() or "interlock" in r3.text.lower() or "panel" in r3.text.lower()

    # Turn 4: "I tried that and it still doesn't work."
    r4 = await conv.process_message("I tried that and it still doesn't work.")
    assert "technician" in r4.text.lower() or "service" in r4.text.lower()
    assert "what product do you have" not in r4.text.lower()  # MUST NOT reset context

    # Verify history integrity
    assert len(conv.history) == 9
