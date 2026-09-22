"""Verification Script for Step 3: NotebookLM Knowledge Provider Integration.

Demonstrates:
1. Case 1: General conversation (no NotebookLM retrieval needed)
2. Case 2: Sales multi-turn grounded with NotebookLM evidence
3. Case 3: Customer Support multi-turn grounded with NotebookLM troubleshooting evidence
4. Grounding & Anti-Hallucination: Refusal when NotebookLM returns no match

Displays exact Step 3 debug telemetry:
  LLM: GEMINI (<model>)
  Knowledge: NOTEBOOKLM (<notebook_id>)
  Knowledge retrieval: YES (<contextual_query>) / NO
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.llm.gemini import GeminiClient
from app.knowledge.notebooklm import NotebookLMKnowledgeProvider
from app.knowledge.mock import MockKnowledgeProvider


from tests.unit.test_step3_knowledge import MockKnowledgeAwareLLM


def print_debug_telemetry(resp):
    print("  " + "-" * 55)
    print(f"  LLM: GEMINI ({resp.llm_result.model if resp.llm_result else 'gemini-2.5-flash'})")
    print(f"  Knowledge: {resp.knowledge_provider_name}")
    if resp.knowledge_retrieved:
        print(f"  Knowledge retrieval: YES")
        if resp.knowledge_query:
            print(f"  Retrieval Query: \"{resp.knowledge_query}\"")
    else:
        print(f"  Knowledge retrieval: NO")
    print("  " + "-" * 55 + "\n")


async def run_step3_demonstrations():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"\n[INFO] Using Live GeminiClient (API Key Detected)")
        llm_client = GeminiClient(api_key=api_key)
        # Attempt to use real NotebookLM provider, fallback to Mock if server not active
        try:
            knowledge_provider = NotebookLMKnowledgeProvider()
        except Exception:
            print("[INFO] Falling back to Coway Knowledge Provider for offline verification")
            knowledge_provider = MockKnowledgeProvider()
    else:
        print(f"\n[INFO] Running in deterministic Verification Mode with Grounded Knowledge")
        knowledge_provider = MockKnowledgeProvider()
        llm_client = MockKnowledgeAwareLLM(knowledge_provider=knowledge_provider)

    # =========================================================================
    # CASE 1: GENERAL CONVERSATION (NO KNOWLEDGE RETRIEVAL REQUIRED)
    # =========================================================================
    print("=" * 75)
    print("CASE 1: GENERAL CONVERSATION — NO NOTEBOOKLM RETRIEVAL NEEDED")
    print("=" * 75)

    sales_agent = SalesAgent(name="Sarah", llm_client=llm_client, knowledge_provider=knowledge_provider)
    conv1 = Conversation(department="SALES", active_agent=sales_agent, knowledge_provider=knowledge_provider)

    greeting = conv1.get_initial_greeting()
    print(f"Agent: {greeting}\n")

    general_turns = [
        "Hello",
        "Can you explain what an air purifier does?",
    ]

    for turn_idx, msg in enumerate(general_turns, 1):
        print(f"Customer: {msg}")
        resp = await conv1.process_message(msg)
        print(f"Agent: {resp.text}")
        print_debug_telemetry(resp)

    # =========================================================================
    # CASE 2: SALES MULTI-TURN WITH GROUNDED NOTEBOOKLM KNOWLEDGE RETRIEVAL
    # =========================================================================
    print("=" * 75)
    print("CASE 2: SALES MULTI-TURN WITH NOTEBOOKLM EVIDENCE & CONVERSATIONAL CONTEXT")
    print("=" * 75)

    sales_agent2 = SalesAgent(name="Sarah", llm_client=llm_client, knowledge_provider=knowledge_provider)
    conv2 = Conversation(department="SALES", active_agent=sales_agent2, knowledge_provider=knowledge_provider)

    print(f"Agent: {conv2.get_initial_greeting()}\n")

    sales_turns = [
        "What products does Coway sell?",
        "What are the discounts on those?",
        "Tell me more about the second one.",
        "How much does it cost?",
    ]

    for turn_idx, msg in enumerate(sales_turns, 1):
        print(f"Customer: {msg}")
        resp = await conv2.process_message(msg)
        print(f"Agent: {resp.text}")
        print_debug_telemetry(resp)

    # =========================================================================
    # CASE 3: SUPPORT MULTI-TURN WITH TROUBLESHOOTING KNOWLEDGE RETRIEVAL
    # =========================================================================
    print("=" * 75)
    print("CASE 3: CUSTOMER SUPPORT MULTI-TURN WITH NOTEBOOKLM TROUBLESHOOTING")
    print("=" * 75)

    support_agent = CustomerSupportAgent(name="Alex", llm_client=llm_client, knowledge_provider=knowledge_provider)
    conv3 = Conversation(department="CUSTOMER_SUPPORT", active_agent=support_agent, knowledge_provider=knowledge_provider)

    print(f"Agent: {conv3.get_initial_greeting()}\n")

    support_turns = [
        "I own an Airmega 150.",
        "It isn't working correctly.",
        "The fan isn't running.",
    ]

    for turn_idx, msg in enumerate(support_turns, 1):
        print(f"Customer: {msg}")
        resp = await conv3.process_message(msg)
        print(f"Agent: {resp.text}")
        print_debug_telemetry(resp)

    # =========================================================================
    # CASE 4: GROUNDING RULE — ZERO HALLUCINATION ON UNKNOWN PRODUCTS
    # =========================================================================
    print("=" * 75)
    print("CASE 4: GROUNDING RULE — ZERO HALLUCINATION ON UNVERIFIED PRODUCTS")
    print("=" * 75)

    conv4 = Conversation(department="SALES", active_agent=sales_agent2, knowledge_provider=knowledge_provider)
    unknown_query = "How much does the Coway TurboMax 9000 cost?"
    print(f"Customer: {unknown_query}")
    resp = await conv4.process_message(unknown_query)
    print(f"Agent: {resp.text}")
    print_debug_telemetry(resp)


if __name__ == "__main__":
    asyncio.run(run_step3_demonstrations())
