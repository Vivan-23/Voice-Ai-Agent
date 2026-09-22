"""Pydantic models for post-call background processing and CRM updates."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CustomerProfile(BaseModel):
    """Customer information model."""
    customer_id: Optional[str] = None
    phone_number: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionItem(BaseModel):
    """Post-call action item or follow-up."""
    description: str
    assignee_role: str = "support"
    due_date: Optional[str] = None
    completed: bool = False


class CallSummary(BaseModel):
    """Structured extraction produced from call recording and transcript."""
    call_id: str
    customer_phone: str
    primary_intent: str
    resolution_status: str  # "resolved", "escalated", "pending_followup"
    summary: str
    key_topics: List[str] = Field(default_factory=list)
    customer_sentiment: str = "neutral"  # "positive", "neutral", "negative", "frustrated"
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    action_items: List[ActionItem] = Field(default_factory=list)
    escalated_to_human: bool = False
    manager_interventions_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
