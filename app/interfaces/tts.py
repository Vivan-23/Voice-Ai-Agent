"""Text-to-Speech (TTS) Service Interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from app.schemas.call import AudioChunk


class BaseTTSService(ABC):
    """Abstract interface for pluggable Text-to-Speech synthesis providers."""

    @abstractmethod
    async def synthesize_stream(
        self, text_stream: AsyncGenerator[str, None], voice_id: Optional[str] = None
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream synthesized audio chunks in real-time as text tokens arrive."""
        yield  # type: ignore

    @abstractmethod
    async def synthesize_text(
        self, text: str, voice_id: Optional[str] = None
    ) -> AudioChunk:
        """Synthesize a complete string into a single audio chunk."""
        pass
