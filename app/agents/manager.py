"""Minimal Manager Agent for Exceptional Situations (Step 4).

Handles:
1. CONTINUE_GUIDE - Current specialist continues with brief internal guidance.
2. ASSIGN_AGENT - Transfers conversation to the appropriate specialist agent (preserving full history).
3. ESCALATE_TO_HUMAN - Escalates the call to a human executive.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm.gemini import GeminiClient, LLMCallResult, normalize_llm_text


class ManagerAction(str, Enum):
    CONTINUE_GUIDE = "CONTINUE_GUIDE"
    ASSIGN_AGENT = "ASSIGN_AGENT"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class ManagerDecision(BaseModel):
    action: ManagerAction
    assigned_agent: Optional[str] = Field(
        default=None,
        description="Assigned specialist: 'SALES' or 'CUSTOMER_SUPPORT' when action is ASSIGN_AGENT.",
    )
    internal_guidance: Optional[str] = Field(
        default=None,
        description="Short internal guidance for the specialist agent if action is CONTINUE_GUIDE.",
    )


class ManagerAgent:
    """Minimal Manager Agent invoked only for exceptional situations."""

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm_client = llm_client or GeminiClient()

    async def decide(
        self,
        current_agent: str,
        conversation_history: List[Dict[str, str]],
        current_customer_message: str,
        reason: Optional[str] = None,
    ) -> ManagerDecision:
        """Evaluate an exceptional situation and return one of the 3 actions."""
        # Format conversation history
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-8:]
        )

        system_prompt = (
            "You are the Coway Helpline Operations Manager.\n"
            "You are invoked ONLY for exceptional situations:\n"
            "1. The customer wants a human representative or senior executive.\n"
            "2. The current agent cannot answer or the issue remains unresolved after troubleshooting.\n"
            "3. A genuine cross-specialist situation occurred (e.g., customer reached Sales but has an existing broken unit, or reached Support but wants to buy a new product).\n\n"
            "You have EXACTLY THREE possible actions:\n"
            "- ESCALATE_TO_HUMAN: Customer explicitly asked for human/supervisor/executive help, or the issue is completely unresolvable by AI.\n"
            "- ASSIGN_AGENT: The conversation belongs to the other specialist ('SALES' or 'CUSTOMER_SUPPORT').\n"
            "- CONTINUE_GUIDE: The current specialist should continue with a specific instruction.\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "action": "CONTINUE_GUIDE" | "ASSIGN_AGENT" | "ESCALATE_TO_HUMAN",\n'
            '  "assigned_agent": "SALES" | "CUSTOMER_SUPPORT" | null,\n'
            '  "internal_guidance": "short instruction for agent" | null\n'
            "}"
        )

        user_content = (
            f"Current Active Specialist: {current_agent}\n"
            f"Intervention Reason: {reason or 'Not specified'}\n\n"
            f"Recent Conversation History:\n{history_text}\n\n"
            f"Latest Customer Message: {current_customer_message}\n\n"
            "Make your managerial decision now in JSON."
        )

        if not self.llm_client.is_configured or getattr(self.llm_client, "_llm", None) is None:
            # Deterministic heuristic fallback when offline/test
            lower_msg = current_customer_message.lower()
            all_text = (history_text + " " + lower_msg).lower()

            if any(term in lower_msg for term in ["human", "executive", "supervisor", "representative", "real person", "someone real", "speak to someone"]):
                return ManagerDecision(action=ManagerAction.ESCALATE_TO_HUMAN)
            elif "sales" in current_agent.lower() and ("own" in all_text or "bought" in all_text or "working" in all_text or "repair" in all_text or "broken" in all_text or "troubleshoot" in all_text):
                return ManagerDecision(action=ManagerAction.ASSIGN_AGENT, assigned_agent="CUSTOMER_SUPPORT")
            elif "support" in current_agent.lower() and ("buy" in all_text or "purchase" in all_text or "catalog" in all_text or "quote" in all_text):
                return ManagerDecision(action=ManagerAction.ASSIGN_AGENT, assigned_agent="SALES")
            elif "still doesn't work" in lower_msg or "still not working" in lower_msg:
                return ManagerDecision(action=ManagerAction.ESCALATE_TO_HUMAN)
            else:
                return ManagerDecision(action=ManagerAction.CONTINUE_GUIDE, internal_guidance="Continue assisting the customer.")

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
            response = await self.llm_client._llm.ainvoke(messages)
            content = normalize_llm_text(response.content)

            # Clean JSON if wrapped in markdown code fence
            if "```" in content:
                # Extract JSON block between code fences
                parts = content.split("```")
                for part in parts:
                    clean_part = part.strip()
                    if clean_part.startswith("json"):
                        clean_part = clean_part[4:].strip()
                    if clean_part.startswith("{") and clean_part.endswith("}"):
                        content = clean_part
                        break

            parsed = json.loads(content)
            action_str = parsed.get("action", "CONTINUE_GUIDE").upper()
            action = ManagerAction(action_str)
            assigned_agent = parsed.get("assigned_agent")
            if assigned_agent:
                assigned_agent = str(assigned_agent).upper()
            internal_guidance = parsed.get("internal_guidance")

            return ManagerDecision(
                action=action,
                assigned_agent=assigned_agent,
                internal_guidance=internal_guidance,
            )
        except Exception:
            # Safe fallback to escalate if parsing fails during exceptional flow
            return ManagerDecision(action=ManagerAction.ESCALATE_TO_HUMAN)
