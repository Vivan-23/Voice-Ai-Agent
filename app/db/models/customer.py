"""Database ORM models for Customers and CRM sync records."""

from datetime import datetime, timezone
from sqlalchemy import String, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base


class CustomerRecord(Base):
    """Stores caller identity and CRM profile snapshot."""
    __tablename__ = "customer_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    phone_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    email: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    tags: Mapped[list | None] = mapped_column(JSON, default=list)
    custom_fields: Mapped[dict | None] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
