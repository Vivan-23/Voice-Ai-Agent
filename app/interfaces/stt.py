"""Speech-to-Text (STT) Service Interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from app.schemas.call import AudioChunk, TranscriptSegment


class BaseSTTService(ABC):
    """Abstract interface for pluggable Speech-to-Text providers."""

    @abstractmethod
    async def transcribe_stream(
        self, audio_stream: AsyncGenerator[AudioChunk, None]
    ) -> AsyncGenerator[TranscriptSegment, None]:
        """Transcribe an incoming real-time audio chunk stream into transcript segments."""
        yield  # type: ignore

    @abstractmethod
    async def transcribe_file(
        self, audio_bytes: bytes, content_type: str = "audio/wav"
    ) -> TranscriptSegment:
        """Transcribe a pre-recorded audio file or complete recording buffer."""
        pass
