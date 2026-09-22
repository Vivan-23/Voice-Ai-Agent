"""Pydantic schemas representing conversational turns, messages, and API payloads."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.agent import AgentRole, UserIntent, CallLifecycleStage

# Alias UserIntent as IntentType for compatibility
IntentType = UserIntent


class ResolutionStatus(str, Enum):
    """Lifecycle status of a conversation issue."""
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED_HUMAN = "escalated_human"
    NEEDS_FOLLOWUP = "needs_followup"


class CustomerInfo(BaseModel):
    """Customer profile information attached to the conversational session."""
    customer_id: Optional[str] = None
    name: Optional[str] = "Customer"
    phone: Optional[str] = None
    email: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """A single message record in the conversation history."""
    role: str = Field(description="Role: user, assistant, system, manager")
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_role: Optional[AgentRole] = None
    intent: Optional[UserIntent] = None
    grounded: bool = True
    citations: List[str] = Field(default_factory=list)


class ConversationTurnRequest(BaseModel):
    """Request payload for sending a text message turn to the AI platform."""
    session_id: str = Field(description="Unique identifier for the customer conversation session")
    message: str = Field(description="User message text")
    caller_phone: Optional[str] = Field(default="+919876543210", description="Customer phone number if known")
    customer_name: Optional[str] = Field(default=None, description="Customer name if known")
    language: Optional[str] = Field(default=None, description="IVR selected language (e.g., English, Hindi)")
    department: Optional[str] = Field(default=None, description="IVR selected department (e.g., SALES, CUSTOMER_SERVICE)")


# Alias for request compatibility
ConversationMessageRequest = ConversationTurnRequest


class CitationItem(BaseModel):
    """Structured citation metadata returned to API consumers."""
    source_title: str
    excerpt: Optional[str] = None
    url: Optional[str] = None


class ConversationTurnResponse(BaseModel):
    """Clean API response returned to caller for a conversational turn."""
    session_id: str
    message: str
    agent_role: AgentRole
    intent: UserIntent
    stage: CallLifecycleStage = CallLifecycleStage.ACTIVE_RESOLUTION
    language: str = "English"
    department: str = "SALES"
    specialist_agent: AgentRole = AgentRole.SALES
    grounded: bool = True
    citations: List[str] = Field(default_factory=list)
    manager_intervention: bool = False
    manager_guidance: Optional[str] = None
    human_escalation_required: bool = False
    resolution_status: ResolutionStatus = ResolutionStatus.IN_PROGRESS
    turn_count: int = 1
    llm_used: bool = True
    llm_model: Optional[str] = None
    knowledge_retrieved: bool = False



# Alias for response compatibility
ConversationMessageResponse = ConversationTurnResponse


# Mapping from intent to appropriate primary specialist agent
INTENT_AGENT_MAP: Dict[UserIntent, AgentRole] = {
    UserIntent.PRODUCT_ENQUIRY: AgentRole.SALES,
    UserIntent.PRICING: AgentRole.SALES,
    UserIntent.OFFER: AgentRole.SALES,
    UserIntent.AVAILABILITY: AgentRole.SALES,
    UserIntent.PRODUCT_USAGE: AgentRole.CUSTOMER_SERVICE,
    UserIntent.TROUBLESHOOTING: AgentRole.CUSTOMER_SERVICE,
    UserIntent.WARRANTY: AgentRole.CUSTOMER_SERVICE,
    UserIntent.INSTALLATION: AgentRole.CUSTOMER_SERVICE,
    UserIntent.MAINTENANCE: AgentRole.CUSTOMER_SERVICE,
    UserIntent.ORDER_SUPPORT: AgentRole.CUSTOMER_SERVICE,
    UserIntent.REFUND_REPLACEMENT: AgentRole.CUSTOMER_SERVICE,
    UserIntent.COMPLAINT: AgentRole.CUSTOMER_SERVICE,
    UserIntent.UNKNOWN: AgentRole.CUSTOMER_SERVICE,
}
