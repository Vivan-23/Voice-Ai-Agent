"""Application configuration powered by Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    app_name: str = "Voice AI Platform POC"
    app_env: str = Field(default="development", description="Runtime environment")
    app_debug: bool = Field(default=True, description="Enable debug mode")
    app_host: str = Field(default="0.0.0.0", description="Host to bind server")
    app_port: int = Field(default=8000, description="Port to bind server")
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARN, ERROR")

    # ElevenLabs configuration (STT & TTS ONLY)
    elevenlabs_api_key: Optional[str] = Field(
        default=None, description="API key for ElevenLabs voice services"
    )
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", description="Default voice ID for TTS synthesis"
    )
    elevenlabs_model_id: str = Field(
        default="eleven_turbo_v2_5", description="Default model ID for TTS synthesis"
    )
    elevenlabs_speech_engine_id: Optional[str] = Field(
        default=None, description="ElevenLabs Speech Engine ID"
    )
    elevenlabs_public_ws_url: Optional[str] = Field(
        default=None, description="Public WSS URL pointing to our voice WebSocket"
    )

    # NotebookLM MCP configuration
    notebooklm_mcp_command: str = Field(
        default="npx", description="Binary or command used to launch NotebookLM MCP"
    )
    notebooklm_mcp_args: List[str] = Field(
        default=["notebooklm-mcp@latest"],
        description="Arguments passed to launch the MCP server",
    )
    notebooklm_notebook_id: Optional[str] = Field(
        default=None, description="Active registered notebook ID"
    )
    notebooklm_notebook_url: Optional[str] = Field(
        default=None, description="Active NotebookLM URL"
    )

    # Knowledge System & Background Sync
    knowledge_provider: str = Field(
        default="local", description="Knowledge provider: 'local', 'notebooklm', 'mock'"
    )
    knowledge_sync_interval_minutes: int = Field(
        default=15, description="Interval in minutes for background knowledge sync check"
    )
    knowledge_snapshot_dir: str = Field(
        default="data/knowledge_snapshots", description="Directory for versioned runtime KB snapshots"
    )

    # Database configuration
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/voice_ai_db",
        description="Async PostgreSQL connection string",
    )
    db_echo: bool = Field(default=False, description="Log raw SQL statements")

    # LLM Settings (Groq primary)
    llm_provider: str = Field(
        default="groq", description="LLM provider: groq, gemini, openai"
    )
    llm_model: str = Field(
        default="openai/gpt-oss-20b", description="Model name for reasoning & agent dialogue"
    )
    groq_api_key: Optional[str] = Field(
        default=None, description="Groq API key"
    )
    llm_api_key: Optional[str] = Field(
        default=None, description="Generic LLM API key"
    )
    gemini_api_key: Optional[str] = Field(
        default=None, description="Google Gemini API key"
    )
    google_api_key: Optional[str] = Field(
        default=None, description="Google API key alias"
    )
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key"
    )
    llm_temperature: float = Field(
        default=0.2, description="Sampling temperature for grounded conversation"
    )

    # CORS & Deployment
    cors_allowed_origins: str = Field(
        default="http://localhost:8000,http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000,http://127.0.0.1:3000,http://127.0.0.1:5173",
        description="Comma-separated allowed CORS origins",
    )
    frontend_origin: Optional[str] = Field(
        default=None, description="Production Frontend Origin (e.g. Netlify URL)"
    )
    port: Optional[int] = Field(
        default=None, description="Render assigned PORT environment variable"
    )

    # Decision Engine (System One structured decision-making)
    decision_engine: str = Field(
        default="jev", description="Decision engine: 'jev' or 'fallback'"
    )
    jev_api_key: Optional[str] = Field(
        default=None, description="API key for TypeSafe AI JEV decision engine"
    )
    jev_api_url: str = Field(
        default="https://api.typesafe.ai/v1/decision", description="Endpoint for JEV decision API"
    )
    jev_timeout_seconds: float = Field(
        default=1.8, description="Max timeout in seconds for JEV decision API before fallback"
    )

    @property
    def effective_port(self) -> int:
        """Derive port from PORT environment variable (Render) or app_port."""
        import os
        if self.port:
            return self.port
        if "PORT" in os.environ:
            try:
                return int(os.environ["PORT"])
            except ValueError:
                pass
        return self.app_port

    def get_cors_origins(self) -> List[str]:
        """Compute list of unique allowed CORS origins."""
        origins = set()
        if self.cors_allowed_origins:
            for o in self.cors_allowed_origins.split(","):
                o = o.strip()
                if o:
                    origins.add(o)
        if self.frontend_origin:
            for o in self.frontend_origin.split(","):
                o = o.strip()
                if o:
                    origins.add(o)
        origins.update([
            "http://localhost:8000",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ])
        return sorted(list(origins))


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
