"""Customer Relationship Management (CRM) Service Interface."""

from abc import ABC, abstractmethod
from typing import Optional
from app.schemas.crm import CallSummary, CustomerProfile


class BaseCRMService(ABC):
    """Abstract interface for CRM providers (Local DB, Salesforce, HubSpot, Zoho, etc.)."""

    @abstractmethod
    async def get_customer(self, phone_number: str) -> Optional[CustomerProfile]:
        """Fetch customer profile by incoming caller phone number."""
        pass

    @abstractmethod
    async def upsert_customer(self, profile: CustomerProfile) -> CustomerProfile:
        """Create or update a customer record."""
        pass

    @abstractmethod
    async def record_call_interaction(
        self, call_summary: CallSummary
    ) -> str:
        """Persist structured post-call summary and interaction notes into CRM."""
        pass
