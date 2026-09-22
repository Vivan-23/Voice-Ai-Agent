"""Abstract interface for System One structured decision engines."""

from abc import ABC, abstractmethod
from app.schemas.decision import JevDecision
from app.schemas.conversation_state import ConversationState


class BaseDecisionEngine(ABC):
    """Abstract interface defining contract for System One Decision Layer (JEV or Fallback)."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the decision engine (e.g. 'JEV' or 'FALLBACK')."""
        pass

    @abstractmethod
    async def decide(
        self,
        state: ConversationState,
        customer_message: str,
    ) -> JevDecision:
        """Determine what the customer's message means and what action should happen next."""
        pass
