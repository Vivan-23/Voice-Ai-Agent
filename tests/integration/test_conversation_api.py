"""Integration tests for FastAPI conversation REST endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.api.dependencies import get_conversation_service
from app.interfaces.knowledge import BaseKnowledgeService, KnowledgeQueryResult, KnowledgeItem
from app.services.conversation import ConversationService


class MockFastKnowledgeService(BaseKnowledgeService):
    """Fast in-memory knowledge service for API endpoint integration tests."""

    async def query(self, question: str, session_id=None) -> KnowledgeQueryResult:
        return KnowledgeQueryResult(
            query=question,
            answer="The Coway Airmega 150 features 3-stage filtration and Smart Auto Mode.",
            items=[KnowledgeItem(source_title="Coway Air Mega 150 Manual", content="Specs", confidence=1.0)],
            is_grounded=True,
        )

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def override_conversation_service():
    mock_service = ConversationService(knowledge_service=MockFastKnowledgeService())
    app.dependency_overrides[get_conversation_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_conversation_turn_endpoint(override_conversation_service):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Send conversation message
        payload = {
            "session_id": "api_test_session_1",
            "message": "How does the Airmega 150 work?",
            "caller_phone": "+919876543210",
            "customer_name": "API Tester",
        }
        res = await client.post("/api/v1/conversation/message", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == "api_test_session_1"
        assert data["agent_role"] in ["customer_service", "sales"]
        assert "message" in data
        assert isinstance(data["grounded"], bool)
        assert data["grounded"] is True

        # 2. Inspect session state
        state_res = await client.get(f"/api/v1/conversation/{payload['session_id']}/state")
        assert state_res.status_code == 200
        state_data = state_res.json()
        assert state_data["session_id"] == "api_test_session_1"
        assert state_data["turn_count"] >= 1

        # 3. Reset session state
        reset_res = await client.post(f"/api/v1/conversation/{payload['session_id']}/reset")
        assert reset_res.status_code == 200
        assert reset_res.json()["reset"] is True
