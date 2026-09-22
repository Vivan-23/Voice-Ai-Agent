"""End-to-End Integration Test for NotebookLM MCP Knowledge Provider."""

import asyncio
import os
import shutil
import sys
import pytest
from app.services.knowledge.notebooklm_mcp import NotebookLMMCPKnowledgeService


@pytest.mark.asyncio
async def test_notebooklm_mcp_end_to_end():
    service = NotebookLMMCPKnowledgeService()

    # 1. Health check
    is_healthy = await service.health_check()
    assert is_healthy is True, "NotebookLM MCP server must report healthy and authenticated"

    # 2. List notebooks
    notebooks = await service.get_notebooks()
    assert len(notebooks) >= 1, "At least one notebook should be registered in the local library"

    # 3. Query grounded knowledge
    question = "What products and services does this company provide?"
    result = await service.query(question)

    assert result.is_grounded is True
    assert len(result.answer) > 50, "Answer should contain synthesized content"
    assert len(result.items) > 0, "Query should return citation items"
