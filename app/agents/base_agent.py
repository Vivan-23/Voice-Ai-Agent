"""Base Specialist Agent for Voice AI Platform POC (Supports Groq & Gemini)."""

import json
import time
import logging
from typing import List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.llm.gemini import GeminiClient, LLMCallResult, normalize_llm_text
from app.knowledge.base import BaseKnowledgeProvider

# Suppress Google GenAI internal AFC advisory notices
logging.getLogger("google.genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


class TurnTimings:
    """Latency metrics for a single conversational turn."""
    def __init__(
        self,
        turn_total_seconds: float = 0.0,
        llm_initial_seconds: float = 0.0,
        knowledge_seconds: Optional[float] = None,
        llm_final_seconds: Optional[float] = None,
        llm_provider: str = "GROQ",
        knowledge_breakdown: Optional[Dict[str, float]] = None,
    ):
        self.turn_total_seconds = turn_total_seconds
        self.llm_initial_seconds = llm_initial_seconds
        self.knowledge_seconds = knowledge_seconds
        self.llm_final_seconds = llm_final_seconds
        self.llm_provider = llm_provider
        self.knowledge_breakdown = knowledge_breakdown or {}


class AgentResponse:
    """Agent output containing generated spoken text, diagnostic metadata, timings, and manager triggers."""

    def __init__(
        self,
        text: str,
        agent_name: str,
        department: str,
        llm_result: LLMCallResult,
        knowledge_retrieved: bool = False,
        knowledge_query: Optional[str] = None,
        knowledge_provider_name: str = "NOTEBOOKLM",
        manager_invoked: bool = False,
        manager_action: Optional[str] = None,
        assigned_agent: Optional[str] = None,
        manager_reason: Optional[str] = None,
        history_preserved: bool = True,
        timings: Optional[TurnTimings] = None,
    ):
        self.text = normalize_llm_text(text)
        self.agent_name = agent_name
        self.department = department
        self.llm_result = llm_result
        self.knowledge_retrieved = knowledge_retrieved
        self.knowledge_query = knowledge_query
        self.knowledge_provider_name = knowledge_provider_name
        self.manager_invoked = manager_invoked
        self.manager_action = manager_action
        self.assigned_agent = assigned_agent
        self.manager_reason = manager_reason
        self.history_preserved = history_preserved
        self.timings = timings or TurnTimings()

    def __repr__(self) -> str:
        return f"<AgentResponse agent={self.agent_name} manager={self.manager_invoked} action={self.manager_action} text='{self.text[:35]}...'>"


class BaseAgent:
    """Base class for specialist conversational agents with grounded knowledge retrieval and manager escalation."""

    def __init__(
        self,
        name: str,
        department: str,
        system_prompt: str,
        llm_client: Optional[GeminiClient] = None,
        knowledge_provider: Optional[BaseKnowledgeProvider] = None,
    ):
        self.name = name
        self.department = department
        self.system_prompt = system_prompt
        self.llm_client = llm_client or GeminiClient()
        self.knowledge_provider = knowledge_provider

    def build_prompt_messages(
        self,
        customer_message: str,
        history: List[Dict[str, str]],
        guidance: Optional[str] = None,
    ) -> List[BaseMessage]:
        """Construct the prompt message list containing system persona, history, guidance, and current turn."""
        prompt = self.system_prompt
        if guidance:
            prompt += f"\n\nMANAGER INTERNAL GUIDANCE:\n{guidance}\nFollow this guidance while responding naturally."

        messages: List[BaseMessage] = [SystemMessage(content=prompt)]

        # Append prior turns in chronological order
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "agent"):
                messages.append(AIMessage(content=content))

        # Append current user utterance
        messages.append(HumanMessage(content=customer_message))
        return messages

    async def respond(
        self,
        customer_message: str,
        history: List[Dict[str, str]],
        guidance: Optional[str] = None,
    ) -> AgentResponse:
        """Generate a contextual response using history, grounded knowledge, and manager escalation if required."""
        t_turn_start = time.perf_counter()
        messages = self.build_prompt_messages(customer_message, history, guidance)
        knowledge_retrieved = False
        knowledge_query = None
        provider_name = self.knowledge_provider.__class__.__name__ if self.knowledge_provider else "NONE"
        if "NotebookLM" in provider_name:
            provider_name = "NOTEBOOKLM"

        llm_provider = getattr(self.llm_client, "provider", "GROQ")

        # Check if real LLM model with tool calling is configured
        if self.llm_client.is_configured and getattr(self.llm_client, "_llm", None) is not None:
            tool_definitions = [
                {
                    "name": "lookup_company_knowledge",
                    "description": "Look up official company knowledge, product models, specifications, pricing, discounts, offers, warranties, or troubleshooting instructions for Coway India.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The contextual search query grounded in the conversation history (e.g. 'Coway Airmega 150 vs 250 discounts' or 'Airmega 250 pricing').",
                            }
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "request_manager_assistance",
                    "description": "Request assistance or intervention from the Operations Manager. Call this ONLY for exceptional situations: (1) Customer explicitly requests human representative/executive/supervisor; (2) Issue remains unresolved after troubleshooting; (3) Cross-specialist handoff needed (e.g. Sales customer needing Support for broken unit, or Support customer wanting to purchase).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Explanation of why manager intervention or handoff is required.",
                            },
                            "situation_type": {
                                "type": "string",
                                "enum": ["HUMAN_REQUEST", "UNRESOLVED_ISSUE", "CROSS_SPECIALIST_HANDOFF", "UNVERIFIED_KNOWLEDGE"],
                                "description": "Category of the exceptional situation.",
                            },
                        },
                        "required": ["reason", "situation_type"],
                    },
                },
            ]

            try:
                t_llm1_start = time.perf_counter()
                llm_with_tools = self.llm_client._llm.bind_tools(tool_definitions)
                first_response = await llm_with_tools.ainvoke(messages)
                t_llm1_end = time.perf_counter()
                llm_initial_sec = t_llm1_end - t_llm1_start

                tool_calls = getattr(first_response, "tool_calls", [])

                # 1. Check if Manager Assistance was requested
                manager_calls = [tc for tc in tool_calls if tc.get("name") == "request_manager_assistance"]
                if manager_calls:
                    m_call = manager_calls[0]
                    reason = m_call.get("args", {}).get("reason", "Exceptional situation detected.")
                    turn_total_sec = time.perf_counter() - t_turn_start
                    return AgentResponse(
                        text="",
                        agent_name=self.name,
                        department=self.department,
                        llm_result=LLMCallResult(content="", provider=llm_provider, model=self.llm_client.model_name, success=True),
                        manager_invoked=True,
                        manager_reason=reason,
                        timings=TurnTimings(
                            turn_total_seconds=turn_total_sec,
                            llm_initial_seconds=llm_initial_sec,
                            llm_provider=llm_provider,
                        ),
                    )

                # 2. Check if Company Knowledge was requested
                knowledge_calls = [tc for tc in tool_calls if tc.get("name") == "lookup_company_knowledge"]
                if knowledge_calls and self.knowledge_provider is not None:
                    knowledge_retrieved = True
                    tool_call = knowledge_calls[0]
                    call_id = tool_call.get("id", "call_knowledge")
                    knowledge_query = tool_call.get("args", {}).get("query", customer_message)

                    # Measure Knowledge Provider latency (NotebookLM)
                    t_k_start = time.perf_counter()
                    evidence = await self.knowledge_provider.query(knowledge_query)
                    t_k_end = time.perf_counter()
                    knowledge_sec = t_k_end - t_k_start
                    knowledge_breakdown = getattr(self.knowledge_provider, "last_timings", {})

                    if not evidence:
                        evidence_str = (
                            "No company knowledge found for this request. "
                            "Do not invent facts, specifications, or prices. "
                            "Politely inform the customer that you cannot verify that specific detail."
                        )
                    else:
                        evidence_str = f"OFFICIAL RETRIEVED EVIDENCE:\n{evidence}\n\nSTRICT RULE: Synthesize a concise, conversational answer using only this evidence. Never hallucinate unstated details."

                    # Pass retrieved evidence back to same LLM (Groq / Gemini) for final synthesis
                    tool_msg = ToolMessage(content=evidence_str, tool_call_id=call_id)
                    followup_messages = list(messages)
                    followup_messages.append(first_response)
                    followup_messages.append(tool_msg)

                    t_llm2_start = time.perf_counter()
                    final_response = await self.llm_client._llm.ainvoke(followup_messages)
                    t_llm2_end = time.perf_counter()
                    llm_final_sec = t_llm2_end - t_llm2_start

                    clean_text = normalize_llm_text(final_response.content)
                    turn_total_sec = time.perf_counter() - t_turn_start

                    return AgentResponse(
                        text=clean_text,
                        agent_name=self.name,
                        department=self.department,
                        llm_result=LLMCallResult(
                            content=clean_text,
                            provider=llm_provider,
                            model=self.llm_client.model_name,
                            success=True,
                        ),
                        knowledge_retrieved=True,
                        knowledge_query=knowledge_query,
                        knowledge_provider_name=provider_name,
                        manager_invoked=False,
                        timings=TurnTimings(
                            turn_total_seconds=turn_total_sec,
                            llm_initial_seconds=llm_initial_sec,
                            knowledge_seconds=knowledge_sec,
                            llm_final_seconds=llm_final_sec,
                            llm_provider=llm_provider,
                            knowledge_breakdown=knowledge_breakdown,
                        ),
                    )

                # 3. Normal direct response (no tools needed)
                clean_text = normalize_llm_text(first_response.content)
                turn_total_sec = time.perf_counter() - t_turn_start
                return AgentResponse(
                    text=clean_text,
                    agent_name=self.name,
                    department=self.department,
                    llm_result=LLMCallResult(
                        content=clean_text,
                        provider=llm_provider,
                        model=self.llm_client.model_name,
                        success=True,
                    ),
                    knowledge_retrieved=False,
                    knowledge_provider_name=provider_name,
                    manager_invoked=False,
                    timings=TurnTimings(
                        turn_total_seconds=turn_total_sec,
                        llm_initial_seconds=llm_initial_sec,
                        llm_provider=llm_provider,
                    ),
                )

            except Exception as e:
                t_fallback_start = time.perf_counter()
                llm_result = await self.llm_client.generate(messages)
                fallback_sec = time.perf_counter() - t_fallback_start
                turn_total_sec = time.perf_counter() - t_turn_start
                clean_text = normalize_llm_text(llm_result.content) if llm_result.success else f"I'm sorry, I'm having trouble retrieving that. ({e})"
                return AgentResponse(
                    text=clean_text,
                    agent_name=self.name,
                    department=self.department,
                    llm_result=llm_result,
                    knowledge_retrieved=False,
                    knowledge_provider_name=provider_name,
                    manager_invoked=False,
                    timings=TurnTimings(
                        turn_total_seconds=turn_total_sec,
                        llm_initial_seconds=fallback_sec,
                        llm_provider=llm_provider,
                    ),
                )

        # Check for test clients with custom manager / knowledge-aware generator
        t_mock_start = time.perf_counter()
        if hasattr(self.llm_client, "generate_with_manager"):
            resp = await self.llm_client.generate_with_manager(messages, self.system_prompt)
            resp.timings = TurnTimings(turn_total_seconds=time.perf_counter() - t_mock_start, llm_initial_seconds=0.01, llm_provider=llm_provider)
            return resp
        elif hasattr(self.llm_client, "generate_with_knowledge"):
            resp = await self.llm_client.generate_with_knowledge(messages, self.system_prompt)
            resp.timings = TurnTimings(turn_total_seconds=time.perf_counter() - t_mock_start, llm_initial_seconds=0.01, llm_provider=llm_provider)
            return resp

        # Fallback for unconfigured / mock LLM client
        llm_result = await self.llm_client.generate(messages)
        clean_text = normalize_llm_text(llm_result.content) if llm_result.success else f"I'm sorry, I'm having trouble connecting. ({llm_result.error_message})"
        return AgentResponse(
            text=clean_text,
            agent_name=self.name,
            department=self.department,
            llm_result=llm_result,
            knowledge_retrieved=knowledge_retrieved,
            knowledge_query=knowledge_query,
            knowledge_provider_name=provider_name,
            manager_invoked=False,
            timings=TurnTimings(
                turn_total_seconds=time.perf_counter() - t_turn_start,
                llm_initial_seconds=0.01,
                llm_provider=llm_provider,
            ),
        )
