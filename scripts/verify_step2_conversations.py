"""Verification script for Step 2: Multi-Turn Conversational Flows.

Demonstrates:
1. Sales Multi-Turn Consultation with Contextual References ('those', 'the second one', 'it')
2. Customer Support Diagnostics with State Memory and Non-Resetting Troubleshooting

Can be run with real Gemini API key or configured test client.
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.llm.gemini import GeminiClient
from tests.unit.test_step2_conversation import MockConversationalLLM


async def run_sales_demonstration(llm_client):
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: SALES MULTI-TURN CONVERSATION")
    print("=" * 70)

    agent = SalesAgent(name="Sarah", llm_client=llm_client)
    conv = Conversation(department="SALES", active_agent=agent)

    print(f"\n[Call Connected — Department: SALES]")
    greeting = conv.get_initial_greeting()
    print(f"Sarah: {greeting}\n")

    turns = [
        "What products does Coway sell?",
        "What are the discounts on those?",
        "Tell me more about the second one.",
        "How much does it cost?",
    ]

    for turn_num, user_msg in enumerate(turns, 1):
        print(f"Customer (Turn {turn_num}): {user_msg}")
        resp = await conv.process_message(user_msg)
        print(f"Sarah: {resp.text}")
        print(f"  --> [DEBUG] Provider: {resp.llm_result.provider} | Model: {resp.llm_result.model} | Success: {resp.llm_result.success} | History Turns: {len(conv.history)}\n")


async def run_support_demonstration(llm_client):
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2: CUSTOMER SUPPORT MULTI-TURN CONVERSATION")
    print("=" * 70)

    agent = CustomerSupportAgent(name="Alex", llm_client=llm_client)
    conv = Conversation(department="CUSTOMER_SUPPORT", active_agent=agent)

    print(f"\n[Call Connected — Department: CUSTOMER SUPPORT]")
    greeting = conv.get_initial_greeting()
    print(f"Alex: {greeting}\n")

    turns = [
        "I own an Airmega 150.",
        "It isn't working correctly.",
        "The fan isn't running.",
        "I tried that and it still doesn't work.",
    ]

    for turn_num, user_msg in enumerate(turns, 1):
        print(f"Customer (Turn {turn_num}): {user_msg}")
        resp = await conv.process_message(user_msg)
        print(f"Alex: {resp.text}")
        print(f"  --> [DEBUG] Provider: {resp.llm_result.provider} | Model: {resp.llm_result.model} | Success: {resp.llm_result.success} | History Turns: {len(conv.history)}\n")


async def main():
    real_gemini = GeminiClient()
    if real_gemini.is_configured:
        print("[Using Real Gemini Client configured with GEMINI_API_KEY]")
        llm = real_gemini
    else:
        print("[GEMINI_API_KEY not found in environment; demonstrating with Conversational Client]")
        llm = MockConversationalLLM()

    await run_sales_demonstration(llm)
    await run_support_demonstration(llm)


if __name__ == "__main__":
    asyncio.run(main())
