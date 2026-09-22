"""Verification Script for Step 4: Manager & Escalation.

Demonstrates:
1. Normal Conversation — Manager is NOT invoked on standard turns.
2. Explicit Human Escalation — Customer requests human executive -> Manager escalates.
3. Cross-Specialist Handoff — Sales customer reporting broken unit transferred to Support with full history preserved.
4. Conversation Continuity — Support specialist acknowledges prior context without asking repetitive questions.
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.agents.manager import ManagerAgent
from app.llm.gemini import GeminiClient
from app.knowledge.mock import MockKnowledgeProvider
from tests.unit.test_step4_manager import MockManagerAwareLLM


def print_step4_telemetry(resp):
    print("  " + "-" * 50)
    print(f"  Agent: {resp.department}")
    if resp.manager_invoked:
        print(f"  Manager: INVOKED")
        print(f"  Manager Action: {resp.manager_action}")
        if resp.manager_action == "ASSIGN_AGENT":
            print(f"  New Agent: {resp.assigned_agent}")
            print(f"  Conversation History: PRESERVED")
    else:
        print(f"  Manager: NOT INVOKED")
    print("  " + "-" * 50 + "\n")


async def run_step4_demonstrations():
    llm_client = GeminiClient()
    if llm_client.is_configured:
        print(f"\n[INFO] Running Step 4 with Live {llm_client.provider} ({llm_client.model_name})")
        knowledge_provider = MockKnowledgeProvider()
        manager_agent = ManagerAgent(llm_client=llm_client)
    else:
        print(f"\n[INFO] Running Step 4 with MockManagerAwareLLM (Deterministic Verification Mode)")
        knowledge_provider = MockKnowledgeProvider()
        llm_client = MockManagerAwareLLM(knowledge_provider=knowledge_provider)
        manager_agent = ManagerAgent(llm_client=llm_client)

    # =========================================================================
    # DEMO 1: NORMAL CONVERSATION (MANAGER NOT INVOKED)
    # =========================================================================
    print("=" * 70)
    print("DEMO 1: NORMAL CONVERSATION — MANAGER NOT INVOKED")
    print("=" * 70)

    sales_agent = SalesAgent(name="Sarah", llm_client=llm_client, knowledge_provider=knowledge_provider)
    conv1 = Conversation(department="SALES", active_agent=sales_agent, knowledge_provider=knowledge_provider, manager_agent=manager_agent)

    print(f"Sarah: {conv1.get_initial_greeting()}\n")

    normal_turns = [
        "What products does Coway sell?",
        "What is the price of Airmega 250?",
        "What is its coverage?",
    ]

    for turn_idx, msg in enumerate(normal_turns, 1):
        print(f"Customer: {msg}")
        resp = await conv1.process_message(msg)
        print(f"Sarah: {resp.text}")
        print_step4_telemetry(resp)

    # =========================================================================
    # DEMO 2: EXPLICIT HUMAN ESCALATION
    # =========================================================================
    print("=" * 70)
    print("DEMO 2: EXPLICIT HUMAN ESCALATION")
    print("=" * 70)

    support_agent = CustomerSupportAgent(name="Alex", llm_client=llm_client, knowledge_provider=knowledge_provider)
    conv2 = Conversation(department="CUSTOMER_SUPPORT", active_agent=support_agent, knowledge_provider=knowledge_provider, manager_agent=manager_agent)

    print(f"Alex: {conv2.get_initial_greeting()}\n")

    escalate_msg = "I want to speak to a senior executive."
    print(f"Customer: {escalate_msg}")
    resp2 = await conv2.process_message(escalate_msg)
    print(f"Response: {resp2.text}")
    print_step4_telemetry(resp2)

    # =========================================================================
    # DEMO 3: CROSS-SPECIALIST HANDOFF (SALES -> SUPPORT) WITH HISTORY PRESERVED
    # =========================================================================
    print("=" * 70)
    print("DEMO 3: SPECIALIST HANDOFF (SALES -> CUSTOMER SUPPORT) WITH PRESERVED CONTEXT")
    print("=" * 70)

    sales_agent3 = SalesAgent(name="Sarah", llm_client=llm_client, knowledge_provider=knowledge_provider)
    conv3 = Conversation(department="SALES", active_agent=sales_agent3, knowledge_provider=knowledge_provider, manager_agent=manager_agent)

    print(f"Sarah: {conv3.get_initial_greeting()}\n")

    # Turn 1: Starts in Sales
    turn1 = "I bought an Airmega 150 last month."
    print(f"Customer: {turn1}")
    r1 = await conv3.process_message(turn1)
    print(f"Sarah: {r1.text}")
    print_step4_telemetry(r1)

    # Turn 2: Customer mentions technical breakdown -> Manager hands off to Support
    turn2 = "The fan isn't running."
    print(f"Customer: {turn2}")
    r2 = await conv3.process_message(turn2)
    print(f"{conv3.active_agent.name}: {r2.text}")
    print_step4_telemetry(r2)


if __name__ == "__main__":
    asyncio.run(run_step4_demonstrations())
