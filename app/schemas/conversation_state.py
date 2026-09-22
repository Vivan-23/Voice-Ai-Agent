"""First-class ConversationState schema representing mental model of representative."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CustomerState(BaseModel):
    name: Optional[str] = "Customer"
    phone: Optional[str] = None


class ProductState(BaseModel):
    active: Optional[str] = None
    mentioned: List[str] = Field(default_factory=list)
    comparison: List[str] = Field(default_factory=list)


class IssueState(BaseModel):
    type: Optional[str] = None
    power_indicator: Optional[bool] = None
    fan_running: Optional[bool] = None
    current_step: Optional[str] = None
    step_failed: Optional[str] = None


class SalesContextState(BaseModel):
    room_type: Optional[str] = None
    room_size: Optional[str] = None
    recommended_model: Optional[str] = None


class ConversationFlowState(BaseModel):
    stage: str = "INTENT_DISCOVERY"
    last_agent_question: Optional[str] = None
    last_agent_instruction: Optional[str] = None
    awaiting_answer_to: Optional[str] = None
    last_customer_message: Optional[str] = None
    last_agent_message: Optional[str] = None


class KnowledgeContextState(BaseModel):
    last_query: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)


class EscalationState(BaseModel):
    manager_required: bool = False
    human_requested: bool = False
    frustrated: bool = False


class ConversationState(BaseModel):
    """Structured first-class object representing what a human agent keeps mentally."""
    call_id: str = "call_default"
    session_id: str = "session_default"
    language: str = "ENGLISH"
    department: str = "CUSTOMER_SERVICE"
    active_specialist: str = "CUSTOMER_SERVICE"
    goal: Optional[str] = None
    customer: CustomerState = Field(default_factory=CustomerState)
    product: ProductState = Field(default_factory=ProductState)
    issue: IssueState = Field(default_factory=IssueState)
    sales: SalesContextState = Field(default_factory=SalesContextState)
    conversation: ConversationFlowState = Field(default_factory=ConversationFlowState)
    knowledge: KnowledgeContextState = Field(default_factory=KnowledgeContextState)
    escalation: EscalationState = Field(default_factory=EscalationState)
    turn_count: int = 0

    def apply_updates(self, updates: Dict[str, Any]):
        """Apply deterministic updates from JevDecision to internal state fields."""
        for key, val in updates.items():
            if key == "active_product":
                self.product.active = val
                if val not in self.product.mentioned:
                    self.product.mentioned.append(val)
            elif key == "mentioned_products" and isinstance(val, list):
                for p in val:
                    if p not in self.product.mentioned:
                        self.product.mentioned.append(p)
            elif key == "comparison" and isinstance(val, list):
                self.product.comparison = val
            elif key == "power_indicator":
                self.issue.power_indicator = bool(val)
            elif key == "fan_running":
                self.issue.fan_running = bool(val)
            elif key == "current_step":
                self.issue.current_step = str(val)
            elif key == "step_failed":
                self.issue.step_failed = str(val)
            elif key == "room_type":
                self.sales.room_type = str(val)
            elif key == "room_size":
                self.sales.room_size = str(val)
            elif key == "recommended_model":
                self.sales.recommended_model = str(val)
            elif key == "goal":
                self.goal = str(val)
            elif key == "stage":
                self.conversation.stage = str(val)
            elif key == "awaiting_answer_to":
                self.conversation.awaiting_answer_to = str(val)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Produce a clean summary dictionary suitable for logging and prompt injection."""
        return self.model_dump()
