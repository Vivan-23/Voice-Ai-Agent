"""Deterministic Mock Knowledge Provider for unit testing and offline verification."""

from typing import Optional
from app.knowledge.base import BaseKnowledgeProvider


class MockKnowledgeProvider(BaseKnowledgeProvider):
    """Grounded in-memory factual knowledge provider simulating NotebookLM."""

    def __init__(self):
        self.catalog = {
            "products": (
                "Coway sells our signature Airmega air purifiers in India, including the Airmega 150 "
                "(covers up to 355 sqft at ₹16,999 with cartridge pre-filter and Green Anti-Flu HEPA) "
                "and the Airmega 250 (covers up to 600 sqft at ₹34,999 with multi-directional airflow and custom auto modes)."
            ),
            "discounts": (
                "For Coway Airmega models (including Airmega 150 and Airmega 250), there are currently "
                "seasonal promotional discounts of up to 10% instant discount on select bank cards."
            ),
            "airmega 250": (
                "The Coway Airmega 250 is listed at ₹34,999. It covers rooms up to 600 square feet with "
                "multi-directional airflow, real-time air quality indicator lights, and washable pre-filter."
            ),
            "airmega 150": (
                "The Coway Airmega 150 covers bedrooms up to 355 square feet at ₹16,999 featuring an easy-clean "
                "cartridge pre-filter, Green Anti-Flu True HEPA, and quiet night mode."
            ),
            "price": (
                "The Airmega 150 is listed at ₹16,999 and the Airmega 250 is listed at ₹34,999."
            ),
            "warranty": (
                "Coway air purifiers include a 1-year product warranty and a 5-year motor warranty with free service visits for manufacturing defects."
            ),
            "troubleshooting": (
                "For Coway air purifiers not powering on or silent fans: Ensure the unit is plugged into a working socket. "
                "If power is on but the fan is silent, the front cover safety interlock switch may be open. Remove the front panel, "
                "ensure the pre-filter is locked securely, and firmly snap the front panel closed."
            ),
            "fan": (
                "If the fan is not spinning while power is on, check the front panel safety switch by re-seating the pre-filter and firmly snapping the front panel shut."
            ),
        }

    async def query(self, question: str) -> Optional[str]:
        q_lower = question.lower()

        if "xyz" in q_lower or "unknown" in q_lower or "unrelated" in q_lower:
            return None

        if any(w in q_lower for w in ["discount", "offer", "deal", "promo"]):
            return self.catalog["discounts"]
        if any(w in q_lower for w in ["fan", "not spinning", "silent", "interlock", "panel"]):
            return self.catalog["fan"]
        if any(w in q_lower for w in ["troubleshoot", "not working", "broken", "issue", "won't turn on"]):
            return self.catalog["troubleshooting"]
        if "250" in q_lower or "second" in q_lower:
            return self.catalog["airmega 250"]
        if "150" in q_lower:
            return self.catalog["airmega 150"]
        if any(w in q_lower for w in ["price", "cost", "how much", "rate"]):
            return self.catalog["price"]
        if any(w in q_lower for w in ["warranty", "guarantee"]):
            return self.catalog["warranty"]
        if any(w in q_lower for w in ["product", "products", "sell", "lineup", "models"]):
            return self.catalog["products"]

        return None
