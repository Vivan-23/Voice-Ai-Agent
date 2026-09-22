"""Structured Decision Layer schemas for JEV & Fallback decision engine."""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DecisionAction(str, Enum):
    """Actions decided by JEV for the ongoing customer conversation."""
    CONTINUE = "CONTINUE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    RETRIEVE_KNOWLEDGE = "RETRIEVE_KNOWLEDGE"
    ANSWER_FROM_CONTEXT = "ANSWER_FROM_CONTEXT"
    RECOMMEND = "RECOMMEND"
    HANDOFF_TO_SPECIALIST = "HANDOFF_TO_SPECIALIST"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"
    END_CALL = "END_CALL"
    DEESCALATE_OR_MANAGER_REVIEW = "DEESCALATE_OR_MANAGER_REVIEW"
    ADVANCE_CONVERSATION = "ADVANCE_CONVERSATION"
    CONTINUE_TROUBLESHOOTING = "CONTINUE_TROUBLESHOOTING"
    CONTINUE_DISCOVERY = "CONTINUE_DISCOVERY"


class JevDecision(BaseModel):
    """Structured decision object emitted by JEV or the Fallback decision engine."""
    action: DecisionAction = Field(default=DecisionAction.CONTINUE, description="Action to take next")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Decision confidence")
    reason: str = Field(default="", description="High-level reasoning for the decision")
    interpretation: Optional[str] = Field(default=None, description="What the customer utterance means in context")
    customer_intent: Optional[str] = Field(default=None, description="Granular intent for this turn")
    conversation_stage: Optional[str] = Field(default=None, description="Current or target lifecycle stage")
    knowledge_required: bool = Field(default=False, description="Whether factual company knowledge is required")
    knowledge_query: Optional[str] = Field(default=None, description="Contextually grounded knowledge query if needed")
    state_updates: Dict[str, Any] = Field(default_factory=dict, description="Key-value state updates to apply")
    next_expected_input: Optional[str] = Field(default=None, description="Expected next answer or input topic")
    manager_required: bool = Field(default=False, description="Whether supervisor/manager intervention is needed")
    specialist_handoff_required: bool = Field(default=False, description="Whether handoff to another specialist is needed")
    target_specialist: Optional[str] = Field(default=None, description="Target specialist role on handoff")
    human_escalation_required: bool = Field(default=False, description="Whether immediate human transfer is required")
    customer_frustrated: bool = Field(default=False, description="Whether customer frustration was detected")
    engine_used: str = Field(default="FALLBACK", description="Engine that generated this decision (JEV or FALLBACK)")
