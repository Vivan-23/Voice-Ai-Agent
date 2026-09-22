"""Database ORM models for Call Sessions and Transcripts."""

from datetime import datetime, timezone
from sqlalchemy import String, DateTime, JSON, ForeignKey, Text, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base


class CallRecord(Base):
    """Stores metadata and post-call analytics for a phone call."""
    __tablename__ = "call_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    customer_phone: Mapped[str] = mapped_column(String(32), index=True)
    recipient_phone: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="initiated")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recording_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    primary_intent: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolution_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    manager_interventions: Mapped[int] = mapped_column(Integer, default=0)
    structured_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    transcripts: Mapped[list["TranscriptRecord"]] = relationship(
        "TranscriptRecord", back_populates="call", cascade="all, delete-orphan"
    )


class TranscriptRecord(Base):
    """Stores granular conversational turns for auditing and RAG grounding."""
    __tablename__ = "transcript_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String(64), ForeignKey("call_records.id", ondelete="CASCADE"), index=True)
    speaker: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    start_time: Mapped[float] = mapped_column(Float, default=0.0)
    end_time: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    call: Mapped[CallRecord] = relationship("CallRecord", back_populates="transcripts")
