"""Knowledge Retrieval Service Interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field


class KnowledgeItem(BaseModel):
    """Normalized knowledge snippet returned from a RAG or MCP source."""

    source_title: str = Field(description="Title or identifier of the source document")
    content: str = Field(description="Factual content or excerpt")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score")
    url: Optional[str] = Field(default=None, description="Direct URL or citation link")


class KnowledgeQueryResult(BaseModel):
    """Result container for knowledge queries."""

    query: str
    answer: str
    items: List[KnowledgeItem] = Field(default_factory=list)
    is_grounded: bool = Field(default=True)


class BaseKnowledgeService(ABC):
    """Abstract interface for knowledge retrieval (MCP, pgvector, or third-party RAG)."""

    @abstractmethod
    async def query(
        self, question: str, session_id: Optional[str] = None
    ) -> KnowledgeQueryResult:
        """Query knowledge base for an answer grounded on ingested company documents."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the knowledge source/MCP server is operational and authenticated."""
        pass
