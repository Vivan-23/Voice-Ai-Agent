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

    # Database configuration
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/voice_ai_db",
        description="Async PostgreSQL connection string",
    )
    db_echo: bool = Field(default=False, description="Log raw SQL statements")

    # LLM Settings (for LangGraph orchestration)
    llm_provider: str = Field(
        default="gemini", description="LLM provider: gemini, openai, local"
    )
    llm_model: str = Field(
        default="gemini-2.5-flash", description="Model name for reasoning & agent dialogue"
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


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
