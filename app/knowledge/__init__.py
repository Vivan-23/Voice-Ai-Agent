"""Knowledge subsystem providing local fast in-memory snapshots and background NotebookLM sync."""

import os
from typing import Optional
from app.knowledge.base import BaseKnowledgeProvider
from app.knowledge.local import LocalKnowledgeProvider
from app.knowledge.notebooklm import NotebookLMKnowledgeProvider
from app.knowledge.mock import MockKnowledgeProvider
from app.knowledge.store import RuntimeKnowledgeStore, KnowledgeSnapshot
from app.knowledge.sync import KnowledgeSyncService
from config.settings import get_settings


def get_knowledge_provider(provider_type: Optional[str] = None) -> BaseKnowledgeProvider:
    """Factory function to resolve active knowledge provider based on settings or env."""
    settings = get_settings()
    ptype = (provider_type or os.getenv("KNOWLEDGE_PROVIDER") or getattr(settings, "knowledge_provider", "local")).lower()

    if ptype in ("local", "runtime", "in_memory", "memory"):
        return LocalKnowledgeProvider()
    elif ptype in ("notebooklm", "mcp"):
        return NotebookLMKnowledgeProvider()
    elif ptype in ("mock", "test"):
        return MockKnowledgeProvider()
    else:
        return LocalKnowledgeProvider()


__all__ = [
    "BaseKnowledgeProvider",
    "LocalKnowledgeProvider",
    "NotebookLMKnowledgeProvider",
    "MockKnowledgeProvider",
    "RuntimeKnowledgeStore",
    "KnowledgeSnapshot",
    "KnowledgeSyncService",
    "get_knowledge_provider",
]
