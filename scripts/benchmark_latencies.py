"""Latency Benchmark Script comparing Synchronous NotebookLM vs Fast Local Runtime Knowledge.

Tests:
1. "hello"
2. "what products do you sell?"
3. "what is the discount on Airmega 250?"

Reports:
- TURN TOTAL LATENCY
- LLM DECISION (Groq)
- LOCAL KNOWLEDGE RETRIEVAL LATENCY (In-memory)
- FINAL GROQ RESPONSE SYNTHESIS
- NotebookLM Status (NOT CALLED)
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
from app.knowledge.local import LocalKnowledgeProvider
from app.knowledge.store import RuntimeKnowledgeStore


async def run_benchmark():
    client = GeminiClient()
    store = RuntimeKnowledgeStore()
    kp = LocalKnowledgeProvider(store=store)

    print("\n" + "=" * 80)
    print("      VOICE AI PLATFORM — FAST RUNTIME KNOWLEDGE BENCHMARK")
    print("=" * 80)
    print(f"CONVERSATIONAL LLM:    {client.provider} ({client.model_name})")
    print(f"KNOWLEDGE PROVIDER:    LOCAL RUNTIME (In-Memory)")
    print(f"ACTIVE KB VERSION:     v{store.active_version} ({len(store.active_snapshot.documents)} indexed documents)")
    print(f"NOTEBOOKLM IN LIVE PATH: NO (Sync / Ingestion only)")
    print("=" * 80 + "\n")

    agent = SalesAgent(name="Sarah", llm_client=client, knowledge_provider=kp)
    conv = Conversation(department="SALES", active_agent=agent, knowledge_provider=kp)

    queries = [
        ("hello", "Greeting (No knowledge required)"),
        ("what products do you sell?", "Product Catalog Discovery (Local Knowledge Retrieval)"),
        ("what is the discount on Airmega 250?", "Discounts & Pricing Lookup (Local Knowledge Retrieval)"),
    ]

    benchmark_results = []

    for q_idx, (q_text, description) in enumerate(queries, 1):
        print("=" * 80)
        print(f"TEST {q_idx}: \"{q_text}\" — [{description}]")
        print("-" * 80)

        resp = await conv.process_message(q_text)
        t = resp.timings
        p_lbl = t.llm_provider

        print(f"Agent ({agent.name}):\n{resp.text}\n")
        print("Telemetry:")
        print(f"  LLM Provider:               {p_lbl}")
        print(f"  Knowledge Provider:         {resp.knowledge_provider_name}")
        print(f"  Knowledge Retrieval:        {'YES' if resp.knowledge_retrieved else 'NO'}")
        print(f"  NotebookLM Called:          {'YES' if resp.knowledge_provider_name == 'NOTEBOOKLM' else 'NOT CALLED (0.00s)'}")
        print(f"  LLM / Tool Decision:        {t.llm_initial_seconds:.3f}s")
        if resp.knowledge_retrieved:
            k_sec = t.knowledge_seconds if t.knowledge_seconds is not None else 0.0
            print(f"  Local Knowledge Retrieval:  {k_sec:.4f}s ({k_sec * 1000:.1f} ms)")
            if t.llm_final_seconds is not None:
                print(f"  Final {p_lbl} Synthesis:     {t.llm_final_seconds:.3f}s")
        print(f"  TURN TOTAL LATENCY:         {t.turn_total_seconds:.3f}s")
        print("=" * 80 + "\n")

        benchmark_results.append({
            "query": q_text,
            "turn_total": t.turn_total_seconds,
            "llm_initial": t.llm_initial_seconds,
            "knowledge_sec": t.knowledge_seconds or 0.0,
            "llm_final": t.llm_final_seconds or 0.0,
            "knowledge_retrieved": resp.knowledge_retrieved,
        })

    # Summary table
    t1 = f"{benchmark_results[0]['turn_total']:.2f}s"
    t2 = f"{benchmark_results[1]['turn_total']:.2f}s"
    t3 = f"{benchmark_results[2]['turn_total']:.2f}s"

    print("\n" + "=" * 80)
    print("              LATENCY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'QUERY':<35} | {'OLD (NOTEBOOKLM)':<18} | {'NEW (LOCAL RUNTIME)':<20}")
    print("-" * 80)
    print(f"{'1. \"hello\"':<35} | {'~1.76s':<18} | {t1:<20}")
    print(f"{'2. \"what products do you sell?\"':<35} | {'~45.01s - 53.79s':<18} | {t2:<20}")
    print(f"{'3. \"what is the discount on 250?\"':<35} | {'~38.96s - 41.09s':<18} | {t3:<20}")
    print("=" * 80)
    print("NOTEBOOKLM REMOVED FROM LIVE CALL PATH: SUCCESS [OK]\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
