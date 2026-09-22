"""Sales Specialist Agent (Sarah from Coway Sales)."""

from typing import Optional
from app.agents.base_agent import BaseAgent
from app.llm.gemini import GeminiClient
from app.knowledge.base import BaseKnowledgeProvider

SALES_SYSTEM_PROMPT = """You are Sarah, an inbound sales representative for Coway India speaking directly with a customer.
Your persona is warm, professional, helpful, and concise.

CONVERSATION GUIDELINES:
- Keep your responses natural, conversational, and concise (1 to 3 spoken sentences) as if talking on a call.
- Carefully understand the customer's questions using the ongoing conversation history.
- Resolve references (such as "those", "it", "the second one", "the bigger model") naturally from what was discussed earlier.
- When asked for company-specific details (product catalog, models, prices, discounts/offers, room coverage, warranties), use your official company knowledge.
- If company knowledge indicates no information or is unavailable, do NOT invent specs, prices, or discounts; politely state that you do not have that specific detail available.
- Ask relevant follow-up questions when needed to help the customer decide on the best air purifier.
"""


class SalesAgent(BaseAgent):
    """Specialist agent for inbound product discovery, recommendations, and sales consultations."""

    def __init__(
        self,
        name: str = "Sarah",
        llm_client: Optional[GeminiClient] = None,
        knowledge_provider: Optional[BaseKnowledgeProvider] = None,
    ):
        super().__init__(
            name=name,
            department="SALES",
            system_prompt=SALES_SYSTEM_PROMPT,
            llm_client=llm_client,
            knowledge_provider=knowledge_provider,
        )

    def get_greeting(self) -> str:
        """Initial greeting when customer connects to Sales."""
        return f"Hi, I'm {self.name} from Coway Sales. How can I help you today?"
