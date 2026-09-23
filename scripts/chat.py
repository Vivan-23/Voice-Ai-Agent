"""Interactive CLI Chat Simulator for Voice AI Platform POC (Conversational Presentation).

Usage:
    python scripts/chat.py
    python scripts/chat.py --debug
    python scripts/chat.py --provider local
    python scripts/chat.py --provider notebooklm
"""

import sys
import os
import asyncio
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.agents.base_agent import AgentResponse
from app.llm.gemini import GeminiClient
from app.knowledge import get_knowledge_provider, RuntimeKnowledgeStore


def print_banner():
    print("=" * 60)
    print("COWAY AI CUSTOMER SERVICE / SALES")
    print("=" * 60)


def prompt_department_selection() -> str:
    print("\nWelcome to Coway India.\n")
    print("Please choose a department:\n")
    print("1. Sales")
    print("2. Customer Support")
    print("=" * 60)

    while True:
        choice = input("\nSelect department (1 or 2): ").strip()
        if choice in ("1", "sales", "1. sales"):
            return "SALES"
        elif choice in ("2", "support", "customer support", "2. customer support"):
            return "CUSTOMER_SUPPORT"
        print("Invalid choice. Please enter 1 for Sales or 2 for Customer Support.")


async def main():
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    forced_provider = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            forced_provider = sys.argv[idx + 1]

    print_banner()
    department = prompt_department_selection()

    llm_client = GeminiClient()
    knowledge_provider = get_knowledge_provider(forced_provider)
    store = RuntimeKnowledgeStore()

    if not llm_client.is_configured:
        print("\n" + "!" * 60)
        print("WARNING: GROQ_API_KEY or GEMINI_API_KEY is not configured.")
        print("!" * 60 + "\n")

    if department == "SALES":
        agent = SalesAgent(llm_client=llm_client, knowledge_provider=knowledge_provider)
    else:
        agent = CustomerSupportAgent(llm_client=llm_client, knowledge_provider=knowledge_provider)

    conversation = Conversation(department=department, active_agent=agent, knowledge_provider=knowledge_provider)

    provider_label = knowledge_provider.__class__.__name__
    if "Local" in provider_label:
        knowledge_display = f"Local Runtime v{store.active_version}"
    elif "NotebookLM" in provider_label:
        knowledge_display = f"NotebookLM ({getattr(knowledge_provider, 'notebook_id', 'remote')})"
    else:
        knowledge_display = provider_label

    dept_label = "Coway Sales" if department == "SALES" else "Coway Customer Support"
    llm_display = "Groq" if "groq" in getattr(llm_client, "provider", "").lower() else getattr(llm_client, "provider", "LLM")

    print("\n" + "=" * 60)
    print(f"{agent.name} — {dept_label}")
    print(f"LLM: {llm_display}")
    print(f"Knowledge: {knowledge_display}")
    print("=" * 60 + "\n")

    # Initial Agent Greeting
    greeting = conversation.get_initial_greeting()
    print(f"{agent.name}:\n{greeting}\n")

    # Interactive Conversation Loop
    while conversation.active:
        try:
            user_input = input("You:\n").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n" + "=" * 60)
            print("Call ended.")
            print("=" * 60)
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
            print("\n" + "=" * 60)
            print("Call ended.")
            print("=" * 60)
            break

        response: AgentResponse = await conversation.process_message(user_input)

        # 1. ALWAYS print the customer-facing spoken agent response first
        active_agent_name = conversation.active_agent.name if hasattr(conversation, "active_agent") else agent.name
        print(f"\n{active_agent_name}:\n{response.text}\n")

        # 2. In debug mode, print clean telemetry AFTER the spoken response
        if debug_mode:
            p_lbl = response.timings.llm_provider if response.timings else "GROQ"
            k_name = response.knowledge_provider_name
            k_retrieved = "YES" if response.knowledge_retrieved else "NO"
            nb_status = "CALLED" if k_name == "NOTEBOOKLM" and response.knowledge_retrieved else "NOT CALLED"
            dept_name = response.department
            m_status = response.manager_action if response.manager_invoked else "NOT INVOKED"

            print("[debug]")
            print(f"LLM: {p_lbl} ({llm_client.model_name})")
            print(f"Knowledge: {k_name}")
            print(f"Knowledge retrieval: {k_retrieved}")
            if response.knowledge_retrieved and response.knowledge_query:
                print(f"Knowledge query: {response.knowledge_query}")
            print(f"NotebookLM: {nb_status}")
            print(f"Agent: {dept_name}")
            print(f"Manager: {m_status}")
            if response.timings:
                t = response.timings
                print("Timing:")
                print(f"  Total: {t.turn_total_seconds:.2f}s")
                print(f"  {p_lbl.capitalize()} decision: {t.llm_initial_seconds:.2f}s")
                if t.knowledge_seconds is not None:
                    if k_name == "LOCAL":
                        print(f"  Local retrieval: {t.knowledge_seconds:.4f}s")
                    else:
                        print(f"  NotebookLM retrieval: {t.knowledge_seconds:.2f}s")
                if t.llm_final_seconds is not None:
                    print(f"  Final {p_lbl.capitalize()} response: {t.llm_final_seconds:.2f}s")
            print("-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
