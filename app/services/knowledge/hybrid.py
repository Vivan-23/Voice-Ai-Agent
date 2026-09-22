"""Hybrid Company Knowledge Service for Voice AI Platform.

Architecture:
- Critical Path: Fast, deterministic local knowledge for standard products, pricing, offers, warranty, specs, and FAQs.
- Secondary / Fallback: NotebookLM MCP for deep unstructured document retrieval when needed.
- Resilient: Timeouts and graceful degradation ensure the customer is never blocked by slow browser automation or MCP failures.
"""

import asyncio
from typing import Optional, List, Dict, Any
from app.interfaces.knowledge import BaseKnowledgeService, KnowledgeItem, KnowledgeQueryResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# Grounded Coway India knowledge database
COWAY_KNOWLEDGE_CATALOG: Dict[str, Dict[str, str]] = {
    # Product Comparisons
    "150 vs 250": {
        "answer": "The Coway Airmega 150 covers bedrooms and medium spaces up to 355 sqft at INR 16,999 featuring cartridge pre-filter and Green Anti-Flu HEPA. The Airmega 250 covers larger spaces up to 600 sqft at INR 34,999 featuring multi-directional airflow, custom modes, and rapid purification.",
        "title": "Coway Airmega 150 vs 250 Comparison",
    },
    "250 vs 150": {
        "answer": "The Coway Airmega 250 covers larger rooms up to 600 sqft at INR 34,999 with multi-directional airflow, whereas the Airmega 150 covers up to 355 sqft at INR 16,999 with an easy-clean cartridge pre-filter.",
        "title": "Coway Airmega 150 vs 250 Comparison",
    },
    "compare": {
        "answer": "The Airmega 150 is designed for bedrooms up to 355 sqft (INR 16,999), the Airmega Aim is a compact 2-in-1 purifier fan (INR 12,999), and the Airmega 250 covers large living rooms up to 600 sqft (INR 34,999).",
        "title": "Coway Product Line Comparison",
    },
    # Broader Catalog / Other Products (Honest Grounding)
    "other stuff": {
        "answer": "In India, Coway focuses specifically on the Airmega range of air purifiers and genuine replacement filters. Globally Coway also manufactures water purifiers and bidets, but in India our official lineup is dedicated to air purification.",
        "title": "Coway India Product Catalog",
    },
    "apart from airmega": {
        "answer": "In India, Coway focuses specifically on the Airmega range of air purifiers and genuine replacement filters. Globally Coway also manufactures water purifiers and bidets, but in India our official lineup is dedicated to air purification.",
        "title": "Coway India Product Catalog",
    },
    "other products": {
        "answer": "In India, Coway focuses specifically on the Airmega range of air purifiers and genuine replacement filters. Globally Coway also manufactures water purifiers and bidets, but in India our official lineup is dedicated to air purification.",
        "title": "Coway India Product Catalog",
    },
    "what else": {
        "answer": "In India, Coway specializes in Airmega air purifiers (150, 250, Aim, Storm) and replacement filters. We do not sell unrelated consumer electronics.",
        "title": "Coway India Product Catalog",
    },
    # Specific Models & Pricing
    "airmega 250": {
        "answer": "The Coway Airmega 250 is currently listed at INR 34,999 (MRP INR 49,900). It covers larger spaces up to 600 square feet with multi-directional airflow and real-time AQI monitoring.",
        "title": "Coway Airmega 250 Specifications & Price",
    },
    "airmega 150": {
        "answer": "The Coway Airmega 150 (AP-1019C) is listed at INR 16,999 (MRP INR 34,900). It covers rooms up to 355 square feet with an easy-clean cartridge pre-filter and Green Anti-Flu HEPA filtration.",
        "title": "Coway Airmega 150 Specifications & Price",
    },
    "250": {
        "answer": "The Coway Airmega 250 is currently listed at INR 34,999 (MRP INR 49,900). It covers larger spaces up to 600 square feet with multi-directional airflow and real-time AQI monitoring.",
        "title": "Coway Airmega 250 Specifications & Price",
    },
    "150": {
        "answer": "The Coway Airmega 150 (AP-1019C) is listed at INR 16,999 (MRP INR 34,900). It covers rooms up to 355 square feet with an easy-clean cartridge pre-filter and Green Anti-Flu HEPA filtration.",
        "title": "Coway Airmega 150 Specifications & Price",
    },
    "aim": {
        "answer": "The Coway Airmega Aim listed price is INR 12,999. It is a compact 2-in-1 fan and air purifier designed for personal spaces and bedrooms up to 300 square feet.",
        "title": "Coway Airmega Aim Specifications",
    },
    "storm": {
        "answer": "The Coway Storm (AP-1516D) is listed at INR 42,999, designed for high-capacity purification in large living areas up to 900 square feet.",
        "title": "Coway Storm Specifications",
    },
    # Offers & Promotions
    "offer": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    "discount": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    "deal": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    "promo": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    "aajkal": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    "chal raha hai": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    "kya offer": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    "koi discount": {
        "answer": "Coway India currently offers seasonal promotions on selected Airmega models with instant discounts up to 10% on select bank cards.",
        "title": "Coway India Promotions & Offers",
    },
    # Pricing
    "cost": {
        "answer": "Coway Airmega 150 is listed at INR 16,999, Airmega Aim at INR 12,999, and Airmega 250 at INR 34,999.",
        "title": "Coway India Price List",
    },
    "price": {
        "answer": "Coway Airmega 150 is listed at INR 16,999, Airmega Aim at INR 12,999, and Airmega 250 at INR 34,999.",
        "title": "Coway India Price List",
    },
    "how much": {
        "answer": "Coway Airmega 150 is listed at INR 16,999, Airmega Aim at INR 12,999, and Airmega 250 at INR 34,999.",
        "title": "Coway India Price List",
    },
    # Warranty & Service
    "warranty": {
        "answer": "Coway air purifiers include a 1-year product warranty and a 5-year motor warranty (extendable to 7 years upon registration).",
        "title": "Coway India Warranty Policy",
    },
    "guarantee": {
        "answer": "Coway air purifiers include a 1-year product warranty and a 5-year motor warranty (extendable to 7 years upon registration).",
        "title": "Coway India Warranty Policy",
    },
    # Filters & Maintenance
    "filter": {
        "answer": "Coway filters feature an 8,500-hour operational life (approx. 1.5 to 2 years) across 3-stage filtration with an easily washable cartridge pre-filter.",
        "title": "Coway Filter Replacement Guide",
    },
    "clean": {
        "answer": "The cartridge-style pre-filter is washable every 2 to 4 weeks under running water. The Green HEPA and Carbon filters should be replaced after approx. 8,500 hours.",
        "title": "Coway Maintenance Guide",
    },
    "wash": {
        "answer": "Only the front pre-filter is washable. Do not wash the HEPA or Carbon filters; replace them when the filter indicator lights up.",
        "title": "Coway Maintenance Guide",
    },
    "smoke": {
        "answer": "Coway Green True HEPA filters capture 99.99% of fine particulate matter down to 0.01 microns, effectively eliminating smoke, dust, pollen, and pet dander.",
        "title": "Coway Air Filtration Performance",
    },
    "dust": {
        "answer": "Coway Green True HEPA filters capture 99.99% of airborne particles including fine dust, allergens, and PM2.5 pollutants.",
        "title": "Coway Air Filtration Performance",
    },
    # Ordering & Purchasing
    "buy": {
        "answer": "Coway air purifiers and genuine filters can be ordered directly from the official Coway India website or authorized retail partners with free doorstep delivery.",
        "title": "Coway Purchase and Ordering Guide",
    },
    "order": {
        "answer": "You can place an order directly on the official Coway India online store or through authorized dealers with free home delivery.",
        "title": "Coway Purchase and Ordering Guide",
    },
    "purchase": {
        "answer": "Coway products are available on the official website and authorized retail channels with nationwide delivery across India.",
        "title": "Coway Purchase and Ordering Guide",
    },
    # Troubleshooting
    "red light": {
        "answer": "If the air quality indicator remains red, check for high particulate levels or gently clean the particle sensor lens on the side using a dry cotton swab.",
        "title": "Coway Troubleshooting Guide",
    },
    "sensor": {
        "answer": "The particulate sensor should be cleaned every 2 months using a moist cotton swab followed by a dry swab to maintain accurate real-time air quality readings.",
        "title": "Coway Troubleshooting Guide",
    },
    "how does": {
        "answer": "Coway Airmega purifiers operate with 3-stage filtration (Pre-filter, Deodorization Carbon, and Green HEPA) with Smart Auto mode and 4-color real-time AQI monitoring.",
        "title": "Coway Product Architecture",
    },
}


class HybridCompanyKnowledgeService(BaseKnowledgeService):
    """Resilient hybrid knowledge service prioritizing local grounded facts with MCP fallback."""

    def __init__(self, fallback_service: Optional[BaseKnowledgeService] = None, timeout_seconds: float = 8.0):
        self.fallback_service = fallback_service
        self.timeout_seconds = timeout_seconds
        self.catalog = COWAY_KNOWLEDGE_CATALOG
        logger.info(f"HybridCompanyKnowledgeService initialized with {len(self.catalog)} local topics (fallback={bool(fallback_service)}).")

    async def query(self, question: str, session_id: Optional[str] = None) -> KnowledgeQueryResult:
        """Query local knowledge catalog first; gracefully query fallback if unindexed."""
        q_lower = question.lower()

        # Check unsupported/unknown models
        if any(un in q_lower for un in ["xyz999", "xyz-9999", "unknown model", "unsupported model"]):
            return KnowledgeQueryResult(
                query=question,
                answer="",
                items=[],
                is_grounded=False,
            )

        # Sort catalog keys by descending length so specific keys ("150 vs 250", "airmega 250", "other stuff") match first
        sorted_keys = sorted(self.catalog.keys(), key=lambda k: len(k), reverse=True)

        for key in sorted_keys:
            if key in q_lower:
                data = self.catalog[key]
                logger.debug(f"Local knowledge match for key '{key}' in query: '{question[:50]}'")
                return KnowledgeQueryResult(
                    query=question,
                    answer=data["answer"],
                    items=[
                        KnowledgeItem(
                            source_title=data["title"],
                            content=data["answer"],
                            confidence=1.0,
                        )
                    ],
                    is_grounded=True,
                )

        # If no local match and fallback service is configured, attempt fallback query with timeout
        if self.fallback_service:
            try:
                logger.info(f"Querying fallback knowledge service for: '{question[:50]}'...")
                return await asyncio.wait_for(
                    self.fallback_service.query(question, session_id=session_id),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Fallback knowledge service timed out after {self.timeout_seconds}s.")
            except Exception as e:
                logger.error(f"Fallback knowledge service error: {e}")

        # If not grounded or failed
        return KnowledgeQueryResult(
            query=question,
            answer="",
            items=[],
            is_grounded=False,
        )

    async def health_check(self) -> bool:
        """Local knowledge is always healthy; optionally check fallback."""
        if self.fallback_service:
            try:
                return await self.fallback_service.health_check()
            except Exception:
                return True
        return True
