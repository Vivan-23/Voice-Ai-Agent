"""Interactive CLI Chat Simulator for Voice AI Platform POC (Step 3).

Usage:
    python scripts/chat.py
    python scripts/chat.py --debug
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
from app.knowledge.notebooklm import NotebookLMKnowledgeProvider


def print_banner():
    print("=" * 60)
    print("        COWAY AI CUSTOMER SERVICE / SALES — POC")
    print("=" * 60)


def prompt_department_selection() -> str:
    print("\nWelcome to Coway India.\n")
    print("Please choose a department:")
    print("  1. Sales")
    print("  2. Customer Support")
    print("=" * 60)

    while True:
        choice = input("\nSelect department (1 or 2): ").strip()
        if choice == "1" or choice.lower() in ("sales", "1. sales"):
            return "SALES"
        elif choice == "2" or choice.lower() in ("support", "customer support", "2. customer support"):
            return "CUSTOMER_SUPPORT"
        print("Invalid choice. Please enter 1 for Sales or 2 for Customer Support.")


async def main():
    debug_mode = "--debug" in sys.argv

    print_banner()
    department = prompt_department_selection()

    llm_client = GeminiClient()
    knowledge_provider = NotebookLMKnowledgeProvider()

    if not llm_client.is_configured:
        print("\n" + "!" * 60)
        print("WARNING: GEMINI_API_KEY is not set in environment or .env file.")
        print("Please configure GEMINI_API_KEY to test with the real Gemini LLM.")
        print("!" * 60 + "\n")

    if department == "SALES":
        agent = SalesAgent(llm_client=llm_client, knowledge_provider=knowledge_provider)
    else:
        agent = CustomerSupportAgent(llm_client=llm_client, knowledge_provider=knowledge_provider)

    conversation = Conversation(department=department, active_agent=agent, knowledge_provider=knowledge_provider)

    print("\n" + "-" * 60)
    print(f"  Agent:               {agent.name}")
    print(f"  Department:          {department}")
    print(f"  LLM:                 {llm_client.provider} ({llm_client.model_name})")
    print(f"  Knowledge:           NOTEBOOKLM ({knowledge_provider.notebook_id})")
    print("-" * 60 + "\n")

    # Initial Agent Greeting
    greeting = conversation.get_initial_greeting()
    print(f"{agent.name}:\n{greeting}\n")

    # Interactive Conversation Loop
    while conversation.active:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nCall ended. Thank you for calling Coway India.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
            print(f"\n{agent.name}:\nThank you for reaching out to Coway India. Have a wonderful day!\n")
            break

        response: AgentResponse = await conversation.process_message(user_input)

        if debug_mode:
            print("\n" + "-" * 50)
            print(f"LLM: {llm_client.provider} ({llm_client.model_name})")
            print(f"Knowledge: {response.knowledge_provider_name}")
            print(f"Knowledge retrieval: {'YES' if response.knowledge_retrieved else 'NO'}")
            if response.knowledge_retrieved and response.knowledge_query:
                print(f"Knowledge Query: {response.knowledge_query}")
            print(f"Agent: {response.department}")
            if response.manager_invoked:
                print(f"Manager: INVOKED")
                print(f"Manager Action: {response.manager_action}")
                if response.manager_action == "ASSIGN_AGENT":
                    print(f"New Agent: {response.assigned_agent}")
                    print(f"Conversation History: PRESERVED")
            else:
                print(f"Manager: NOT INVOKED")

            # Timing Telemetry
            if response.timings:
                provider_label = response.timings.llm_provider
                print("Timing Telemetry:")
                print(f"  TURN TOTAL:            {response.timings.turn_total_seconds:.2f}s")
                print(f"  LLM / TOOL DECISION:   {response.timings.llm_initial_seconds:.2f}s ({provider_label})")
                if response.timings.knowledge_seconds is not None:
                    print(f"  NOTEBOOKLM TOTAL:      {response.timings.knowledge_seconds:.2f}s")
                    bd = response.timings.knowledge_breakdown
                    if bd:
                        if "mcp_startup" in bd:
                            print(f"    - MCP Process Startup:   {bd['mcp_startup']:.2f}s")
                        if "session_init" in bd:
                            print(f"    - Session Init:          {bd['session_init']:.2f}s")
                        if "ask_question_wait" in bd:
                            print(f"    - Ask Question & Wait:   {bd['ask_question_wait']:.2f}s")
                        if "parsing" in bd:
                            print(f"    - Parsing & Cleanup:     {bd['parsing']:.2f}s")
                if response.timings.llm_final_seconds is not None:
                    print(f"  FINAL {provider_label} RESPONSE: {response.timings.llm_final_seconds:.2f}s")
            print("-" * 50)

        print(f"\n{agent.name}:\n{response.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
