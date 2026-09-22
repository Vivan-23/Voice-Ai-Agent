"""Base Knowledge Provider Interface for Voice AI Platform POC."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseKnowledgeProvider(ABC):
    """Abstract interface for factual company knowledge retrieval."""

    @abstractmethod
    async def query(self, question: str) -> Optional[str]:
        """Query the knowledge base and return relevant factual evidence or None."""
        pass
