"""Pydantic schemas representing calls, audio chunks, and transcripts."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class CallStatus(str, Enum):
    """Lifecycle status of a phone call."""
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    FAILED = "failed"


class AudioEncoding(str, Enum):
    """Audio encodings supported by telephony and streaming pipeline."""
    PCM_16K = "pcm_16000"
    PCM_8K = "pcm_8000"
    MULAW_8K = "mulaw_8000"
    MP3 = "mp3"
    WAV = "wav"


class AudioChunk(BaseModel):
    """A discrete binary slice of streamed audio with metadata."""
    data: bytes
    encoding: AudioEncoding = AudioEncoding.PCM_16K
    sample_rate: int = 16000
    sequence_id: int = 0
    is_final: bool = False


class SpeakerRole(str, Enum):
    """Speaker turn attribution."""
    CUSTOMER = "customer"
    AGENT = "agent"
    SYSTEM = "system"


class TranscriptSegment(BaseModel):
    """A transcription chunk representing spoken words."""
    speaker: SpeakerRole
    text: str
    confidence: float = 1.0
    start_time: float = 0.0
    end_time: float = 0.0
    is_final: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CallSession(BaseModel):
    """Active call session holding metadata and state."""
    call_id: str
    caller_phone: str
    recipient_phone: str
    status: CallStatus = CallStatus.INITIATED
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    transcript: List[TranscriptSegment] = Field(default_factory=list)
    active_agent: str = "router"
