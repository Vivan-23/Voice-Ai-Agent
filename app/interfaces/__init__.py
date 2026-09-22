"""Export abstract domain interfaces."""

from app.interfaces.stt import BaseSTTService
from app.interfaces.tts import BaseTTSService
from app.interfaces.knowledge import BaseKnowledgeService, KnowledgeItem, KnowledgeQueryResult
from app.interfaces.crm import BaseCRMService
from app.interfaces.decision import BaseDecisionEngine

__all__ = [
    "BaseSTTService",
    "BaseTTSService",
    "BaseKnowledgeService",
    "KnowledgeItem",
    "KnowledgeQueryResult",
    "BaseCRMService",
    "BaseDecisionEngine",
]
