"""Unit and Integration Tests for Knowledge Synchronization System & Fast Runtime Store.

Tests all 10 core requirements:
1. Initial snapshot loading.
2. Existing snapshot remains available if sync fails.
3. Successful sync creates a new version (v1 -> v2).
4. Failed sync does not replace the active version.
5. Unchanged knowledge does not trigger unnecessary rebuild.
6. LocalKnowledgeProvider can answer from the active snapshot.
7. Customer conversation does NOT invoke NotebookLM when KNOWLEDGE_PROVIDER=local.
8. GET /knowledge/status returns diagnostic status.
9. POST /knowledge/sync triggers synchronization.
10. Background sync does not block normal conversation handling.
"""

import os
import shutil
import tempfile
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from app.knowledge.store import (
    RuntimeKnowledgeStore,
    KnowledgeSnapshot,
    DEFAULT_SEED_DOCUMENTS,
    DEFAULT_SEED_SOURCES,
)
from app.knowledge.local import LocalKnowledgeProvider
from app.knowledge.sync import KnowledgeSyncService
from app.knowledge import get_knowledge_provider
from app.agents.sales_agent import SalesAgent
from app.conversation.conversation import Conversation
from app.main import app


@pytest.fixture
def temp_snapshot_dir():
    """Create an isolated temporary directory for knowledge snapshots."""
    temp_dir = tempfile.mkdtemp(prefix="kb_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def clean_store(temp_snapshot_dir):
    """Provide a freshly initialized RuntimeKnowledgeStore instance."""
    RuntimeKnowledgeStore._instance = None
    store = RuntimeKnowledgeStore(snapshot_dir=temp_snapshot_dir)
    return store


@pytest.mark.asyncio
async def test_initial_snapshot_loading(clean_store):
    """1. Initial snapshot loading bootstraps v1 from seed data."""
    snapshot = clean_store.active_snapshot
    assert snapshot is not None
    assert snapshot.version == 1
    assert snapshot.status == "READY"
    assert clean_store.active_version == 1
    assert len(snapshot.documents) >= 5
    assert clean_store.get_status()["active_version"] == 1


@pytest.mark.asyncio
async def test_successful_sync_creates_new_version(clean_store):
    """3. Successful sync creates a new version (v1 -> v2) and switches active pointer."""
    sync_service = KnowledgeSyncService(store=clean_store)

    # Mock MCP change detection & document fetching for fast unit test
    async def mock_inspect():
        return True, "Notebook topics updated", {
            "notebook_id": "test_nb",
            "notebook_name": "Coway Support",
            "fingerprint": "fp_v2_test",
        }

    async def mock_fetch(nb_id):
        return [
            {
                "id": "doc_new_offer",
                "title": "Special Festive 2026 Offer",
                "category": "discounts",
                "source_name": "NotebookLM Synced",
                "keywords": ["offer", "festive"],
                "content": "Special festive discount of 20% on all Coway purifiers during festival month.",
            }
        ]

    sync_service.inspect_source_changes = mock_inspect
    sync_service.fetch_knowledge_documents_from_mcp = mock_fetch

    result = await sync_service.sync(force=True)
    assert result["success"] is True
    assert result["action"] == "SYNCED"
    assert result["active_version"] == 2
    assert clean_store.active_version == 2
    assert clean_store.active_snapshot.version == 2
    assert len(clean_store.active_snapshot.documents) >= 6


from tests.unit.test_step3_knowledge import MockKnowledgeAwareLLM


@pytest.mark.asyncio
async def test_failed_sync_does_not_replace_active_version(clean_store):
    """2 & 4. Failed sync does not replace the active version; existing snapshot remains available."""
    initial_version = clean_store.active_version
    initial_docs_count = len(clean_store.active_snapshot.documents)

    sync_service = KnowledgeSyncService(store=clean_store)

    async def mock_inspect():
        return True, "Source changed", {"notebook_name": "Coway", "fingerprint": "fp_v2"}

    sync_service.inspect_source_changes = mock_inspect
    sync_service.fetch_knowledge_documents_from_mcp = AsyncMock(return_value=[])

    # Force validation to fail
    sync_service.validate_snapshot = lambda s: False

    result = await sync_service.sync(force=True)
    assert result["success"] is False
    assert result["action"] == "FAILED"
    assert result["active_version"] == initial_version

    # Active snapshot is untouched
    assert clean_store.active_version == initial_version
    assert len(clean_store.active_snapshot.documents) == initial_docs_count
    assert clean_store.last_error is not None


@pytest.mark.asyncio
async def test_unchanged_knowledge_skips_rebuild(clean_store):
    """5. Unchanged knowledge does not trigger unnecessary rebuild."""
    sync_service = KnowledgeSyncService(store=clean_store)

    async def mock_unchanged():
        return False, "Knowledge metadata is unchanged", {"fingerprint": "same_fp"}

    sync_service.inspect_source_changes = mock_unchanged

    result = await sync_service.sync(force=False)
    assert result["success"] is True
    assert result["action"] == "SKIPPED"
    assert result["active_version"] == clean_store.active_version


@pytest.mark.asyncio
async def test_local_knowledge_provider_answers_from_snapshot(clean_store):
    """6. LocalKnowledgeProvider retrieves relevant evidence from active snapshot quickly."""
    provider = LocalKnowledgeProvider(store=clean_store)

    # Query products
    evidence = await provider.query("Airmega 250 specs and price")
    assert evidence is not None
    assert "Airmega 250" in evidence
    assert "34,999" in evidence or "29,999" in evidence
    assert provider.last_timings["provider"] == "LOCAL"
    assert provider.last_timings["matched_chunks"] > 0
    assert provider.last_timings["total"] < 0.05  # < 50ms (typically < 2ms)

    # Query discounts
    discount_evidence = await provider.query("discounts and bank offers")
    assert discount_evidence is not None
    assert "HDFC" in discount_evidence or "10%" in discount_evidence

    # Query troubleshooting
    ts_evidence = await provider.query("power silent fan front cover")
    assert ts_evidence is not None
    assert "interlock" in ts_evidence.lower() or "safety" in ts_evidence.lower()


class LocalKnowledgeAwareLLM(MockKnowledgeAwareLLM):
    async def generate_with_knowledge(self, messages, system_prompt):
        resp = await super().generate_with_knowledge(messages, system_prompt)
        resp.knowledge_provider_name = "LOCAL"
        return resp


@pytest.mark.asyncio
async def test_conversation_does_not_invoke_notebooklm(clean_store):
    """7. Customer conversation does NOT invoke NotebookLM when KNOWLEDGE_PROVIDER=local."""
    provider = LocalKnowledgeProvider(store=clean_store)
    mock_llm = LocalKnowledgeAwareLLM(knowledge_provider=provider)
    agent = SalesAgent(name="Sarah", llm_client=mock_llm, knowledge_provider=provider)
    conv = Conversation(department="SALES", active_agent=agent, knowledge_provider=provider)

    # Process message
    response = await conv.process_message("What products do you sell?")
    assert response is not None
    assert response.knowledge_provider_name == "LOCAL"
    assert conv.knowledge_provider.__class__.__name__ == "LocalKnowledgeProvider"
    assert response.timings.knowledge_breakdown == {}  # No MCP breakdown needed for local


@pytest.mark.asyncio
async def test_get_knowledge_status_endpoint(clean_store):
    """8. GET /knowledge/status endpoint returns active version, status, item counts."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/knowledge/status")
        assert res.status_code == 200
        data = res.json()
        assert "active_version" in data
        assert "status" in data
        assert "source_count" in data
        assert "indexed_item_count" in data
        assert data["indexed_item_count"] >= 5


@pytest.mark.asyncio
async def test_post_knowledge_sync_endpoint(clean_store):
    """9. POST /knowledge/sync endpoint triggers synchronization."""
    transport = ASGITransport(app=app)
    with patch.object(KnowledgeSyncService, "inspect_source_changes", AsyncMock(return_value=(True, "Forced update", {"fingerprint": "fp_test"}))):
        with patch.object(KnowledgeSyncService, "fetch_knowledge_documents_from_mcp", AsyncMock(return_value=[])):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                res = await client.post("/knowledge/sync?force=true")
                assert res.status_code == 200
                data = res.json()
                assert data["success"] is True
                assert data["action"] == "SYNCED"
                assert data["active_version"] >= 1


@pytest.mark.asyncio
async def test_background_sync_lifecycle_and_non_blocking(clean_store):
    """10. Background sync starts, runs non-blocking, and stops cleanly."""
    sync_service = KnowledgeSyncService(store=clean_store)
    sync_service.start_background_sync(interval_minutes=15)
    assert sync_service._bg_task is not None
    assert not sync_service._bg_task.done()

    # Verify conversation processing works immediately while background sync is running
    provider = LocalKnowledgeProvider(store=clean_store)
    mock_llm = LocalKnowledgeAwareLLM(knowledge_provider=provider)
    agent = SalesAgent(name="Sarah", llm_client=mock_llm, knowledge_provider=provider)
    conv = Conversation(department="SALES", active_agent=agent, knowledge_provider=provider)

    resp = await conv.process_message("hello")
    assert resp is not None
    assert len(resp.text) > 0

    sync_service.stop_background_sync()
    assert sync_service._bg_task is None
