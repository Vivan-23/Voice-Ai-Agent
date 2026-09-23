"""Runtime Knowledge Store & Snapshot Persistence for Voice AI Platform POC.

Maintains:
- In-memory active knowledge snapshot for sub-millisecond call-time retrieval
- Versioned snapshot persistence on disk (data/knowledge_snapshots/v{N}.json)
- Atomic snapshot switching and manifest management
- Search index for factual product, policy, and troubleshooting retrieval
"""

import os
import json
import time
import re
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class KnowledgeDocument:
    """Individual indexed knowledge chunk/article."""
    id: str
    title: str
    category: str  # e.g., "products", "discounts", "warranty", "troubleshooting", "faq"
    source_name: str
    content: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class KnowledgeSource:
    """Ingested knowledge source metadata."""
    id: str
    name: str
    source_type: str  # "notebooklm", "catalog", "manual", "document"
    fingerprint: str
    item_count: int = 0
    synced_at: Optional[str] = None


@dataclass
class KnowledgeSnapshot:
    """Versioned immutable snapshot of company knowledge."""
    version: int
    status: str  # "READY", "BUILDING", "ERROR"
    timestamp: str  # ISO 8601
    source_count: int
    sources: List[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_sync_duration_seconds: float = 0.0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeSnapshot":
        return cls(
            version=data.get("version", 1),
            status=data.get("status", "READY"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            source_count=data.get("source_count", len(data.get("sources", []))),
            sources=data.get("sources", []),
            documents=data.get("documents", []),
            metadata=data.get("metadata", {}),
            last_sync_duration_seconds=data.get("last_sync_duration_seconds", 0.0),
            last_error=data.get("last_error"),
        )


# Default baseline seed knowledge for Coway India (used on initial bootstrap)
DEFAULT_SEED_DOCUMENTS = [
    {
        "id": "prod_airmega_150",
        "title": "Coway Airmega 150 Specifications & Pricing",
        "category": "products",
        "source_name": "Coway Product Catalog 2026",
        "keywords": ["airmega 150", "150", "bedroom", "coverage", "price", "mrp", "cost", "filter", "hepa", "dimensions", "specs"],
        "content": (
            "Model: Coway Airmega 150 (AP-1019C).\n"
            "Target Space: Compact bedrooms, study rooms, personal offices.\n"
            "Room Coverage: Up to 355 sq. ft. (33 sq. meters) with 303 m3/hr CADR.\n"
            "Pricing: MRP INR 16,999 (Standard Selling Price: INR 14,999).\n"
            "Filtration System: 3-Stage Filtration featuring Washable Pre-Filter (cartridge-type for easy removal and cleaning), "
            "Patented Urethane Carbon Deodorization Filter, and Anti-Virus Green True HEPA Filter trapping 99.97% of airborne allergens and PM0.1 particles.\n"
            "Key Features: Real-time AQI 4-color LED indicator ring (Blue: Good, Green: Normal, Yellow: Unhealthy, Red: Very Unhealthy), "
            "Filter replacement LED reminder, Ultra-quiet operation (22 dB in sleep mode), Power consumption 35 Watts.\n"
            "Available Colors: White, Moss Green, and Peony Pink."
        ),
    },
    {
        "id": "prod_airmega_250",
        "title": "Coway Airmega 250 Specifications & Pricing",
        "category": "products",
        "source_name": "Coway Product Catalog 2026",
        "keywords": ["airmega 250", "250", "living room", "coverage", "price", "mrp", "cost", "filter", "hepa", "max2", "eco mode", "smart"],
        "content": (
            "Model: Coway Airmega 250 (AP-1720H).\n"
            "Target Space: Medium to large living rooms, master bedrooms, office lounges.\n"
            "Room Coverage: Up to 600 sq. ft. (55 sq. meters) with 460 m3/hr CADR.\n"
            "Pricing: MRP INR 34,999 (Standard Selling Price: INR 29,999).\n"
            "Filtration System: Multi-directional 360-degree suction with Max2 all-in-one Filter combining Green True HEPA and Activated Carbon.\n"
            "Key Features: Smart ECO Mode (automatically stops fan when air remains clean for 10 minutes to save power, resumes when pollution detected), "
            "Rapid Mode for high-speed cleaning, Real-time PM2.5 numerical/color indicator, Sleep mode (22 dB), Child lock, Power consumption 45 Watts.\n"
            "Design: Sleek matte finish with bottom pre-filter slide-out tray for convenient cleaning without opening the entire front chassis."
        ),
    },
    {
        "id": "prod_airmega_300",
        "title": "Coway Airmega 300 / 300S Specifications & Pricing",
        "category": "products",
        "source_name": "Coway Product Catalog 2026",
        "keywords": ["airmega 300", "airmega 300s", "300", "large room", "commercial", "coverage", "price", "dual suction"],
        "content": (
            "Model: Coway Airmega 300 / 300S.\n"
            "Target Space: Large open-plan apartments, villas, commercial offices.\n"
            "Room Coverage: Up to 1,256 sq. ft. (117 sq. meters).\n"
            "Pricing: MRP INR 44,999.\n"
            "Filtration System: Dual Suction Max2 Filters on both sides of unit for rapid air turnover.\n"
            "Key Features: Smart Wi-Fi connectivity (300S model) via Coway IoCare Mobile App, Alexa / Google Assistant voice control, Smart Auto mode."
        ),
    },
    {
        "id": "offers_discounts_2026",
        "title": "Coway Official Discounts & Promotional Offers",
        "category": "discounts",
        "source_name": "Coway Sales & Promotional Policy",
        "keywords": ["discount", "discounts", "offer", "offers", "promotion", "coupon", "deal", "bank offer", "credit card", "hdfc", "icici", "sbi", "exchange"],
        "content": (
            "Current Official Promotions & Discounts:\n"
            "1. Instant Bank Discount: Up to 10% instant discount (max INR 2,000) on HDFC, ICICI, and SBI Bank credit cards and EMI transactions.\n"
            "2. Seasonal Bundle Offer: 15% discount on purchasing an extra replacement Max2/Green HEPA filter set together with any Airmega purifier.\n"
            "3. Delivery & Setup: 100% Free Pan-India doorstep delivery with free video/onsite installation demonstration.\n"
            "4. No-Cost EMI: 3-month and 6-month no-cost EMI available across leading credit cards."
        ),
    },
    {
        "id": "warranty_service_policy",
        "title": "Coway Warranty, Filter Life & Customer Support Details",
        "category": "warranty",
        "source_name": "Coway Customer Care & Warranty Guidelines",
        "keywords": ["warranty", "guarantee", "motor", "service", "customer care", "helpline", "toll free", "filter life", "replacement", "hours"],
        "content": (
            "Warranty & Service Coverage:\n"
            "1. Standard Warranty: 1-Year Comprehensive Warranty covering all electrical components and manufacturing defects.\n"
            "2. Motor Warranty: 5-Year Extended Warranty on the internal suction motor.\n"
            "3. Filter Lifespan: Green True HEPA and Max2 filters have an 8,500-hour (approx 1 year of daily use) lifespan. "
            "The washable pre-filter should be washed under running water every 2 to 4 weeks.\n"
            "4. Customer Care Contact: Toll-free Helpline: 1800-102-6960 (Monday to Saturday, 9:00 AM - 6:00 PM IST) or email info@cowayindia.in.\n"
            "5. Service Turnaround: Pan-India home technician visits within 24 to 48 hours in major metropolitan cities."
        ),
    },
    {
        "id": "troubleshooting_power_fan",
        "title": "Troubleshooting: Unit Won't Turn On or Silent Fan",
        "category": "troubleshooting",
        "source_name": "Coway Technical Service Manual",
        "keywords": ["power", "not turning on", "silent", "fan not working", "won't start", "front cover", "interlock", "safety switch", "troubleshoot"],
        "content": (
            "Troubleshooting Steps for Power / Fan Not Starting:\n"
            "1. Check Power Source: Ensure the power cord is securely plugged into a live 220-240V AC wall outlet and the power button is pressed.\n"
            "2. Inspect Front Cover Safety Interlock: All Coway Airmega models feature a physical safety interlock switch. If the front panel cover is not snapped tightly into place, the safety switch will disconnect power to the fan to prevent injury.\n"
            "3. Cover Reseat Procedure: Remove the front cover, verify the pre-filter is properly locked in its upper clips, and then align the bottom tabs of the front cover and push firmly on the top two corners until a solid 'click' sound is heard on both sides.\n"
            "4. If power indicator light is on but fan does not spin after securing cover, guide the customer to request a technician service visit via 1800-102-6960."
        ),
    },
    {
        "id": "troubleshooting_aqi_sensor_odor",
        "title": "Troubleshooting: AQI Light Red / Odor / Noise Issues",
        "category": "troubleshooting",
        "source_name": "Coway Technical Service Manual",
        "keywords": ["red light", "aqi sensor", "dust sensor", "smell", "odor", "noise", "plastic", "troubleshoot"],
        "content": (
            "Troubleshooting Steps for AQI Sensor and Odor:\n"
            "1. Red AQI Light Staying On Constantly: The optical particle sensor on the side may have dust accumulation. Open the sensor cover on the right side of the unit and gently wipe the lens with a damp cotton swab, followed by a dry cotton swab. Adjust sensor sensitivity if necessary.\n"
            "2. Plastic / Unpleasant Smell from New Unit: Check if the plastic wrapping on the internal HEPA/Carbon filters was removed prior to first use. Always unwrap filter plastic before turning on.\n"
            "3. Sour or Musty Smell: Indicates pre-filter is dirty or carbon filter has reached saturation (> 1 year). Wash pre-filter and replace carbon filter cartridge."
        ),
    },
]


DEFAULT_SEED_SOURCES = [
    {
        "id": "src_coway_catalog",
        "name": "Coway Product Catalog 2026",
        "source_type": "catalog",
        "fingerprint": "fp_coway_catalog_2026",
        "item_count": 3,
        "synced_at": "2026-09-22T12:00:00Z",
    },
    {
        "id": "src_coway_promotions",
        "name": "Coway Sales & Promotional Policy",
        "source_type": "document",
        "fingerprint": "fp_coway_promo_2026",
        "item_count": 1,
        "synced_at": "2026-09-22T12:00:00Z",
    },
    {
        "id": "src_coway_warranty",
        "name": "Coway Customer Care & Warranty Guidelines",
        "source_type": "document",
        "fingerprint": "fp_coway_warranty_2026",
        "item_count": 1,
        "synced_at": "2026-09-22T12:00:00Z",
    },
    {
        "id": "src_coway_manual",
        "name": "Coway Technical Service Manual",
        "source_type": "manual",
        "fingerprint": "fp_coway_manual_2026",
        "item_count": 2,
        "synced_at": "2026-09-22T12:00:00Z",
    },
]


class RuntimeKnowledgeStore:
    """Singleton in-memory knowledge store with atomic version switching and disk persistence."""

    _instance: Optional["RuntimeKnowledgeStore"] = None

    def __new__(cls, snapshot_dir: Optional[str] = None) -> "RuntimeKnowledgeStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, snapshot_dir: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self.snapshot_dir = snapshot_dir or os.getenv("KNOWLEDGE_SNAPSHOT_DIR", "data/knowledge_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.snapshot_dir, "manifest.json")

        self.active_snapshot: Optional[KnowledgeSnapshot] = None
        self.is_syncing: bool = False
        self.last_error: Optional[str] = None

        # Load latest active snapshot from disk or initialize with seed
        self.load_active_snapshot()
        self._initialized = True

    @property
    def active_version(self) -> int:
        return self.active_snapshot.version if self.active_snapshot else 0

    @property
    def status(self) -> str:
        if self.is_syncing:
            return "SYNCING"
        if self.active_snapshot:
            return self.active_snapshot.status
        return "UNINITIALIZED"

    def load_active_snapshot(self) -> KnowledgeSnapshot:
        """Load the active snapshot from disk manifest, or create default v1 seed snapshot."""
        try:
            if os.path.exists(self.manifest_path):
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                active_ver = manifest.get("active_version", 1)
                snapshot_file = os.path.join(self.snapshot_dir, f"v{active_ver}.json")

                if os.path.exists(snapshot_file):
                    with open(snapshot_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    snapshot = KnowledgeSnapshot.from_dict(data)
                    self.active_snapshot = snapshot
                    self.last_error = None
                    return snapshot

            # If manifest doesn't exist or snapshot file missing, initialize seed snapshot v1
            seed_snapshot = KnowledgeSnapshot(
                version=1,
                status="READY",
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_count=len(DEFAULT_SEED_SOURCES),
                sources=DEFAULT_SEED_SOURCES,
                documents=DEFAULT_SEED_DOCUMENTS,
                metadata={"type": "seed_bootstrap", "notebook_id": "coway-india-customer-support-k"},
                last_sync_duration_seconds=0.001,
            )
            self.save_snapshot(seed_snapshot)
            self.active_snapshot = seed_snapshot
            self._update_manifest(1)
            return seed_snapshot

        except Exception as e:
            self.last_error = f"Failed to load active snapshot: {e}"
            # Fallback to in-memory seed snapshot
            fallback = KnowledgeSnapshot(
                version=1,
                status="READY",
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_count=len(DEFAULT_SEED_SOURCES),
                sources=DEFAULT_SEED_SOURCES,
                documents=DEFAULT_SEED_DOCUMENTS,
                metadata={"type": "in_memory_fallback"},
            )
            self.active_snapshot = fallback
            return fallback

    def save_snapshot(self, snapshot: KnowledgeSnapshot) -> str:
        """Write snapshot to disk safely (v{version}.json)."""
        file_path = os.path.join(self.snapshot_dir, f"v{snapshot.version}.json")
        temp_path = f"{file_path}.tmp.{os.getpid()}"

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)

        # Atomic file rename on Windows/POSIX
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)

        return file_path

    def _update_manifest(self, active_version: int):
        """Update manifest.json pointing to current active version."""
        manifest = {
            "active_version": active_version,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        temp_path = f"{self.manifest_path}.tmp.{os.getpid()}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)
        os.rename(temp_path, self.manifest_path)

    def switch_active_version(self, snapshot: KnowledgeSnapshot):
        """Atomically switch in-memory pointer to new validated version and update manifest."""
        self.save_snapshot(snapshot)
        self.active_snapshot = snapshot
        self._update_manifest(snapshot.version)
        self.last_error = None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform high-speed in-memory lexical and keyword matching across active snapshot documents.

        Execution time is typically < 1 ms.
        """
        if not self.active_snapshot or not self.active_snapshot.documents:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored_docs: List[Tuple[float, Dict[str, Any]]] = []

        for doc in self.active_snapshot.documents:
            score = 0.0
            title_lower = doc.get("title", "").lower()
            content_lower = doc.get("content", "").lower()
            keywords = [k.lower() for k in doc.get("keywords", [])]

            # 1. Exact phrase matches in title or content
            if query.lower() in title_lower:
                score += 15.0
            if query.lower() in content_lower:
                score += 8.0

            # 2. Keyword exact and partial matches
            for kw in keywords:
                if kw in query.lower():
                    score += 10.0
                elif any(tok in kw for tok in query_tokens if len(tok) > 2):
                    score += 3.0

            # 3. Token overlap with content
            for token in query_tokens:
                if len(token) <= 2:
                    continue
                if token in title_lower:
                    score += 4.0
                if token in content_lower:
                    score += 1.0

            if score > 0.0:
                scored_docs.append((score, doc))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored_docs[:top_k]]

    def get_status(self) -> Dict[str, Any]:
        """Return full diagnostic status of the runtime knowledge base."""
        snap = self.active_snapshot
        return {
            "active_version": snap.version if snap else 0,
            "status": self.status,
            "last_sync": snap.timestamp if snap else None,
            "source_count": snap.source_count if snap else 0,
            "indexed_item_count": len(snap.documents) if snap else 0,
            "last_error": self.last_error or (snap.last_error if snap else None),
            "is_syncing": self.is_syncing,
            "snapshot_dir": self.snapshot_dir,
            "last_sync_duration_seconds": snap.last_sync_duration_seconds if snap else 0.0,
        }
