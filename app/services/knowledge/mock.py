"""Deterministic Mock Knowledge Service for tests and offline/local verification."""

from typing import Optional
from app.interfaces.knowledge import BaseKnowledgeService, KnowledgeItem, KnowledgeQueryResult


class MockCompanyKnowledgeService(BaseKnowledgeService):
    """Deterministic Mock Knowledge Service loaded with verified Coway India facts."""

    def __init__(self):
        self.catalog = {
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
            "airmega 250": {
                "answer": "The Coway Airmega 250 is listed at INR 34,999. It covers rooms up to 600 square feet.",
                "title": "Coway India Price List",
            },
            "airmega 150": {
                "answer": "The Coway Airmega 150 features cartridge-style easy-clean filters, 303 m3/h CADR, and smart auto modes.",
                "title": "Coway Airmega 150 Specifications",
            },
            "250": {
                "answer": "The Coway Airmega 250 is listed at INR 34,999. It covers rooms up to 600 square feet.",
                "title": "Coway India Price List",
            },
            "how much": {
                "answer": "The Coway Airmega 150 listed price is INR 16,999 (MRP INR 34,900), and the Airmega 250 is listed at INR 34,999.",
                "title": "Coway India Price List",
            },
            "cost": {
                "answer": "The Coway Airmega 150 listed price is INR 16,999 (MRP INR 34,900), and the Airmega 250 is listed at INR 34,999.",
                "title": "Coway India Price List",
            },
            "price": {
                "answer": "The Coway Airmega 150 listed price is INR 16,999 (MRP INR 34,900), and the Airmega 250 is listed at INR 34,999.",
                "title": "Coway India Price List",
            },
            "discount": {
                "answer": "Coway India currently offers seasonal promotions on the Airmega 150 with instant discounts up to 10% on select bank cards.",
                "title": "Coway India Promotions & Offers",
            },
            "offer": {
                "answer": "Coway India currently offers seasonal promotions on the Airmega 150 with instant discounts up to 10% on select bank cards.",
                "title": "Coway India Promotions & Offers",
            },
            "aajkal": {
                "answer": "Coway India currently offers seasonal promotions on the Airmega 150 with instant discounts up to 10% on select bank cards.",
                "title": "Coway India Promotions & Offers",
            },
            "chal raha hai": {
                "answer": "Coway India currently offers seasonal promotions on the Airmega 150 with instant discounts up to 10% on select bank cards.",
                "title": "Coway India Promotions & Offers",
            },
            "kya offer": {
                "answer": "Coway India currently offers seasonal promotions on the Airmega 150 with instant discounts up to 10% on select bank cards.",
                "title": "Coway India Promotions & Offers",
            },
            "koi offer": {
                "answer": "Coway India currently offers seasonal promotions on the Airmega 150 with instant discounts up to 10% on select bank cards.",
                "title": "Coway India Promotions & Offers",
            },
            "warranty": {
                "answer": "Coway air purifiers include a 1-year product warranty and a 5-year motor warranty (extendable to 7 years upon registration).",
                "title": "Coway India Warranty Policy",
            },
            "how does": {
                "answer": "The Coway Airmega 150 features cartridge-style easy-clean filters, 303 m3/h CADR, and smart auto modes.",
                "title": "Coway Airmega 150 Specifications",
            },
            "airmega": {
                "answer": "Coway Airmega is our premium line of air purifiers with Green True HEPA filtration, designed for bedrooms, living rooms, and large spaces.",
                "title": "Coway Airmega Product Range",
            },
            "troubleshoot": {
                "answer": "For Coway air purifiers, ensure the front panel is securely snapped into place and the pre-filter is clean. Check if the power indicator is illuminated.",
                "title": "Coway Standard Troubleshooting Guide",
            },
            "filter": {
                "answer": "Filters feature an 8,500-hour operational life (approx. 1.5 to 2 years) across 3-stage filtration.",
                "title": "Coway Filter Replacement Guide",
            },
            "smoke": {
                "answer": "Coway Green True HEPA filters capture 99.99% of fine particulate matter down to 0.01 microns, effectively eliminating smoke, dust, pollen, and pet dander.",
                "title": "Coway Air Filtration Performance",
            },
            "dust": {
                "answer": "Coway Green True HEPA filters capture 99.99% of airborne particles including fine dust, allergens, and PM2.5 pollutants.",
                "title": "Coway Air Filtration Performance",
            },
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
        }

    async def query(self, question: str, session_id: Optional[str] = None) -> KnowledgeQueryResult:
        q_lower = question.lower()

        # Unknown / unsupported models
        if "xyz999" in q_lower or "xyz-9999" in q_lower or "unknown" in q_lower:
            return KnowledgeQueryResult(
                query=question,
                answer="",
                items=[],
                is_grounded=False,
            )

        # Match specific topics by longest key first
        sorted_keys = sorted(self.catalog.keys(), key=lambda k: len(k), reverse=True)
        for key in sorted_keys:
            if key in q_lower:
                data = self.catalog[key]
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

        return KnowledgeQueryResult(
            query=question,
            answer="",
            items=[],
            is_grounded=False,
        )

    async def health_check(self) -> bool:
        return True
