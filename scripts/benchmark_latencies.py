"""Latency Benchmark & Breakdown Script for Voice AI Platform POC.

Runs the exact 3 requested user tests:
1. "hello"
2. "what products do you sell?"
3. "what is the discount on Airmega 250?"

Outputs detailed timing breakdown:
- LLM / Tool Decision (Groq)
- NotebookLM MCP Breakdown (Process Startup, Session Init, Ask Question Wait, Parsing/Normalization)
- Final Response Synthesis (Groq)
- Turn Total Latency
"""

import sys
import os
import asyncio
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.llm.gemini import GeminiClient
from app.knowledge.notebooklm import NotebookLMKnowledgeProvider


async def run_benchmark():
    client = GeminiClient()
    kp = NotebookLMKnowledgeProvider()

    print("\n" + "=" * 75)
    print("RUNTIME CONFIGURATION")
    print("=" * 75)
    print(f"CONVERSATIONAL LLM:  {client.provider} ({client.model_name})")
    print(f"KNOWLEDGE PROVIDER:  NOTEBOOKLM ({kp.notebook_id})")
    print(f"MCP COMMAND:         {kp.command} {' '.join(kp.args)}")
    print("=" * 75 + "\n")

    agent = SalesAgent(name="Sarah", llm_client=client, knowledge_provider=kp)
    conv = Conversation(department="SALES", active_agent=agent, knowledge_provider=kp)

    queries = [
        "hello",
        "what products do you sell?",
        "what is the discount on Airmega 250?",
    ]

    for q_idx, q_text in enumerate(queries, 1):
        print("=" * 75)
        print(f"TEST {q_idx}: Customer Query -> \"{q_text}\"")
        print("-" * 75)

        resp = await conv.process_message(q_text)
        t = resp.timings
        provider_label = t.llm_provider

        print(f"Agent ({agent.name}):\n{resp.text}\n")
        print("Detailed Timing Breakdown:")
        print(f"  TURN TOTAL LATENCY:       {t.turn_total_seconds:.2f}s")
        print(f"  LLM / TOOL DECISION:      {t.llm_initial_seconds:.2f}s ({provider_label})")
        
        if resp.knowledge_retrieved:
            print(f"  NOTEBOOKLM TOTAL:         {t.knowledge_seconds:.2f}s")
            bd = t.knowledge_breakdown
            if bd:
                if "mcp_startup" in bd:
                    print(f"    - MCP Process Startup:  {bd['mcp_startup']:.2f}s")
                if "session_init" in bd:
                    print(f"    - Session Init:         {bd['session_init']:.2f}s")
                if "ask_question_wait" in bd:
                    print(f"    - Ask Question & Wait:  {bd['ask_question_wait']:.2f}s")
                if "parsing" in bd:
                    print(f"    - Response Normalizing: {bd['parsing']:.2f}s")
            if t.llm_final_seconds is not None:
                print(f"  FINAL {provider_label} RESPONSE:    {t.llm_final_seconds:.2f}s")
            print(f"  Knowledge Query:          \"{resp.knowledge_query}\"")
        else:
            print(f"  KNOWLEDGE RETRIEVAL:      NO (Answered directly by {provider_label})")

        print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
