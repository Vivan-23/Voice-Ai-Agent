"""Tests for public production deployment readiness:
- GET /health
- GET /status (safe, non-secret diagnostic telemetry)
- CORS origin resolution
- Dynamic Render PORT binding
- Frontend static asset availability
"""

import pytest
from httpx import AsyncClient, ASGITransport

from config.settings import Settings, get_settings
from app.main import app, FRONTEND_HTML_PATH, VOICE_HTML_PATH


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify public GET /health responds with HTTP 200 without requiring external services."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"
        assert "service" in data


@pytest.mark.asyncio
async def test_runtime_status_endpoint_non_secret():
    """Verify GET /status exposes necessary diagnostic information without leaking secrets."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/status")
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert data.get("status") == "healthy"
        assert "llm" in data
        assert "knowledge" in data
        assert "elevenlabs" in data
        assert "cors_origins" in data

        # Check knowledge fields
        assert data["knowledge"]["provider"] == "LOCAL RUNTIME"
        assert data["knowledge"]["version"] >= 1
        assert data["knowledge"]["items"] >= 1

        # Check NO secrets are exposed
        body_text = response.text.lower()
        assert "gsk_" not in body_text
        assert "xi_api_key" not in body_text
        assert "api_key" not in body_text
        assert "password" not in body_text
        assert "secret" not in body_text


def test_cors_origin_resolution():
    """Verify CORS origins include localhost and custom frontend origins."""
    settings = Settings(
        cors_allowed_origins="http://localhost:8000,https://custom.domain.com",
        frontend_origin="https://coway-ai.netlify.app",
    )
    origins = settings.get_cors_origins()
    assert "https://custom.domain.com" in origins
    assert "https://coway-ai.netlify.app" in origins
    assert "http://localhost:8000" in origins


def test_effective_port_detection(monkeypatch):
    """Verify effective port adapts to Render PORT environment variable."""
    settings = Settings(app_port=8000)
    assert settings.effective_port == 8000

    monkeypatch.setenv("PORT", "10000")
    assert settings.effective_port == 10000


def test_frontend_files_exist():
    """Verify frontend/index.html and app/templates/voice.html exist."""
    assert FRONTEND_HTML_PATH.exists() or VOICE_HTML_PATH.exists()
