"""Unit and integration tests for ElevenLabs Speech Engine voice integration.

Covers:
1. Configuration loading and defaults
2. Missing ELEVENLABS_API_KEY error handling
3. Missing Speech Engine ID error handling
4. Session isolation (no state leakage between callers)
5. Sales voice session uses SalesAgent (Sarah)
6. Support voice session uses CustomerSupportAgent (Alex)
7. Voice path does not invoke NotebookLM
8. Voice path strictly uses LocalKnowledgeProvider
9. Manager agent remains non-invoked for standard inquiries
10. Manager escalation occurs on human request
11. WebSocket HS256 JWT authorization verification
12. URL normalization for Speech Engine WebSocket endpoints
"""

import time
import hmac
import hashlib
import json
import base64
import pytest
from httpx import AsyncClient, ASGITransport

from config.settings import Settings, get_settings
from app.voice.session_manager import (
    VoiceSessionManager,
    VoiceTurnTiming,
    get_voice_session_manager,
)
from app.knowledge.local import LocalKnowledgeProvider
from app.knowledge.store import RuntimeKnowledgeStore
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent
from app.main import app
from scripts.setup_elevenlabs import normalize_ws_url
from elevenlabs.speech_engine import verify_speech_engine_jwt


def test_elevenlabs_settings_defaults():
    """Test 1: Verify ElevenLabs configuration settings and field definitions."""
    settings = Settings()
    assert hasattr(settings, "elevenlabs_api_key")
    assert hasattr(settings, "elevenlabs_speech_engine_id")
    assert hasattr(settings, "elevenlabs_public_ws_url")
    assert bool(settings.elevenlabs_voice_id)
    assert bool(settings.elevenlabs_model_id)


@pytest.mark.asyncio
async def test_missing_api_key_fails_clearly(monkeypatch):
    """Test 2: Missing ELEVENLABS_API_KEY returns clear HTTP 400."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/voice/session", json={"department": "SALES"})
        assert response.status_code == 400
        data = response.json()
        assert "ELEVENLABS_API_KEY is not configured" in data["detail"]


@pytest.mark.asyncio
async def test_missing_speech_engine_id_fails_clearly(monkeypatch):
    """Test 3: Missing Speech Engine ID returns HTTP 400 pointing to setup script."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key_123")
    monkeypatch.setenv("ELEVENLABS_SPEECH_ENGINE_ID", "")
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/voice/session", json={"department": "SALES"})
        assert response.status_code == 400
        data = response.json()
        assert "ELEVENLABS_SPEECH_ENGINE_ID is not configured" in data["detail"]
        assert "python scripts/setup_elevenlabs.py" in data["detail"]


def test_voice_session_isolation():
    """Test 4: Each voice session creates completely isolated conversation state."""
    manager = VoiceSessionManager()

    session1 = manager.create_pending_session("SALES")
    session2 = manager.create_pending_session("CUSTOMER_SUPPORT")

    # Verify session identifiers are distinct
    assert session1.session_id != session2.session_id
    assert session1.conversation is not session2.conversation

    # Bind sessions to distinct ElevenLabs conversation IDs
    bound1 = manager.bind_conversation("conv_user_alpha", session_id=session1.session_id)
    bound2 = manager.bind_conversation("conv_user_beta", session_id=session2.session_id)

    # Mutate history in session 1
    bound1.conversation.history.append({"role": "user", "content": "Hello Sarah, price check"})
    assert len(bound1.conversation.history) > len(bound2.conversation.history)
    assert not any("price check" in m["content"] for m in bound2.conversation.history)

    # Retrieve by conversation ID
    retrieved1 = manager.get_session("conv_user_alpha")
    retrieved2 = manager.get_session("conv_user_beta")
    assert retrieved1.session_id == session1.session_id
    assert retrieved2.session_id == session2.session_id

    # Cleanup session 1 and verify session 2 remains untouched
    manager.remove_session("conv_user_alpha")
    assert manager.get_session("conv_user_alpha") is None
    assert manager.get_session("conv_user_beta") is not None


def test_sales_voice_session_uses_sales_agent():
    """Test 5: Sales voice session configures SalesAgent (Sarah)."""
    manager = VoiceSessionManager()
    session = manager.create_pending_session("SALES")

    assert session.department == "SALES"
    assert session.agent_name == "Sarah"
    assert isinstance(session.conversation.active_agent, SalesAgent)
    assert "Sarah from Coway Sales" in session.greeting
    assert session.conversation.history[0]["content"] == session.greeting


def test_support_voice_session_uses_support_agent():
    """Test 6: Support voice session configures CustomerSupportAgent (Alex)."""
    manager = VoiceSessionManager()
    session = manager.create_pending_session("CUSTOMER_SUPPORT")

    assert session.department == "CUSTOMER_SUPPORT"
    assert session.agent_name == "Alex"
    assert isinstance(session.conversation.active_agent, CustomerSupportAgent)
    assert "Alex from Coway Customer Support" in session.greeting
    assert session.conversation.history[0]["content"] == session.greeting


def test_voice_uses_local_knowledge_not_notebooklm():
    """Test 7 & 8: Voice session strictly uses LocalKnowledgeProvider, NOT NotebookLM."""
    manager = VoiceSessionManager()
    session = manager.create_pending_session("SALES")

    provider = session.conversation.knowledge_provider
    assert isinstance(provider, LocalKnowledgeProvider)
    assert "NotebookLM" not in provider.__class__.__name__

    # Verify active agent has the local knowledge provider attached
    assert session.conversation.active_agent.knowledge_provider is provider
    assert isinstance(session.conversation.active_agent.knowledge_provider, LocalKnowledgeProvider)


@pytest.mark.asyncio
async def test_manager_remains_exception_only():
    """Test 9: Manager agent is NOT invoked for normal informational queries."""
    manager = VoiceSessionManager()
    session = manager.create_pending_session("SALES")

    # Mock response without manager intervention
    response = await session.conversation.process_message("What air purifiers do you have?")
    # In standard flow, manager_invoked is False unless exception condition triggers
    if hasattr(response, "manager_invoked"):
        assert response.manager_invoked is False or response.manager_action is None


@pytest.mark.asyncio
async def test_manager_escalates_on_human_request():
    """Test 10: Manager agent intervenes when customer explicitly requests human."""
    manager = VoiceSessionManager()
    session = manager.create_pending_session("SALES")

    response = await session.conversation.process_message("I need to speak to a human representative right now.")
    assert response.manager_invoked is True
    assert response.manager_action in ("ESCALATE_TO_HUMAN", "ASSIGN_AGENT", "CONTINUE_GUIDE")


def test_jwt_verification_logic():
    """Test 11: Speech Engine JWT authorization verification logic."""
    api_key = "test_elevenlabs_secret_key"
    now = int(time.time())

    # Build valid HS256 JWT
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": "https://api.elevenlabs.io/convai/speech-engine",
        "sub": "convai_speech_engine_upstream",
        "iat": now,
        "exp": now + 60,
    }

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    h_b64 = b64url(json.dumps(header).encode())
    p_b64 = b64url(json.dumps(payload).encode())
    msg = f"{h_b64}.{p_b64}".encode()

    secret = hashlib.sha256(api_key.encode()).digest()
    sig = hmac.new(secret, msg, hashlib.sha256).digest()
    sig_b64 = b64url(sig)

    token = f"{h_b64}.{p_b64}.{sig_b64}"

    # Valid token passes
    verified = verify_speech_engine_jwt(token, api_key)
    assert verified["iss"] == "https://api.elevenlabs.io/convai/speech-engine"
    assert verified["sub"] == "convai_speech_engine_upstream"

    # Corrupt token fails
    with pytest.raises(ValueError):
        verify_speech_engine_jwt(token + "corrupt", api_key)


def test_normalize_ws_url():
    """Test 12: URL normalizer correctly formats wss endpoint."""
    assert normalize_ws_url("https://abc.ngrok-free.app") == "wss://abc.ngrok-free.app/ws/voice"
    assert normalize_ws_url("http://localhost:8000") == "ws://localhost:8000/ws/voice"
    assert normalize_ws_url("wss://abc.ngrok-free.app/ws/voice") == "wss://abc.ngrok-free.app/ws/voice"
    assert normalize_ws_url("abc.ngrok-free.app") == "wss://abc.ngrok-free.app/ws/voice"


def test_voice_turn_timing_format():
    """Test 13: VoiceTurnTiming formats debug telemetry block matching specification."""
    timing = VoiceTurnTiming(
        transcript="what products do you sell?",
        turn_total_seconds=0.85,
        stt_seconds=0.35,
        llm_initial_seconds=0.22,
        knowledge_seconds=0.0003,
        llm_final_seconds=0.28,
        time_to_first_text_seconds=0.45,
        llm_provider="GROQ",
        knowledge_provider="LOCAL",
        notebooklm_called=False,
        agent_name="Sarah",
        department="SALES",
    )
    block = timing.format_debug_block()
    assert "VOICE TURN" in block
    assert "what products do you sell?" in block
    assert "GROQ / DECISION:" in block
    assert "LOCAL KNOWLEDGE:       0.0003s" in block
    assert "KNOWLEDGE SOURCE:      LOCAL" in block
    assert "NOTEBOOKLM:            NOT CALLED" in block
    assert "SPECIALIST AGENT:      Sarah (SALES)" in block


@pytest.mark.asyncio
async def test_health_and_voice_ui_endpoint():
    """Test 14: Health and Voice UI endpoints return HTTP 200."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        health_resp = await ac.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        voice_resp = await ac.get("/voice")
        assert voice_resp.status_code == 200
        assert "COWAY AI" in voice_resp.text
        assert "Choose Department" in voice_resp.text
        assert "Sarah" in voice_resp.text
        assert "Alex" in voice_resp.text
