"""Local In-Memory Knowledge Provider for fast conversational retrieval.

Fetches grounded company knowledge directly from the in-memory RuntimeKnowledgeStore
in sub-millisecond execution time, completely eliminating synchronous NotebookLM browser delays
from live customer calls.
"""

import time
from typing import Optional, Dict, Any, List
from app.knowledge.base import BaseKnowledgeProvider
from app.knowledge.store import RuntimeKnowledgeStore


class LocalKnowledgeProvider(BaseKnowledgeProvider):
    """Fast, in-memory knowledge provider reading from active KnowledgeSnapshot."""

    def __init__(self, store: Optional[RuntimeKnowledgeStore] = None):
        self.store = store or RuntimeKnowledgeStore()
        self.last_timings: Dict[str, Any] = {}

    async def query(self, question: str) -> Optional[str]:
        """Query the in-memory knowledge store and return formatted factual evidence.

        Execution is completely in-memory and typically completes in under 5 milliseconds.
        """
        t_start = time.perf_counter()

        try:
            matched_docs = self.store.search(question, top_k=3)
            retrieval_sec = time.perf_counter() - t_start

            if not matched_docs:
                self.last_timings = {
                    "provider": "LOCAL",
                    "retrieval_sec": retrieval_sec,
                    "matched_chunks": 0,
                    "active_version": self.store.active_version,
                    "total": retrieval_sec,
                }
                return None

            # Format grounded evidence blocks
            evidence_parts: List[str] = []
            for i, doc in enumerate(matched_docs, 1):
                title = doc.get("title", f"Document {i}")
                source = doc.get("source_name", "Official Company Knowledge")
                content = doc.get("content", "").strip()
                evidence_parts.append(f"[{i}] {title} (Source: {source}):\n{content}")

            formatted_evidence = "\n\n".join(evidence_parts)

            t_total = time.perf_counter() - t_start
            self.last_timings = {
                "provider": "LOCAL",
                "retrieval_sec": retrieval_sec,
                "matched_chunks": len(matched_docs),
                "active_version": self.store.active_version,
                "total": t_total,
            }

            return formatted_evidence

        except Exception as ex:
            self.last_timings = {
                "provider": "LOCAL",
                "error": str(ex),
                "total": time.perf_counter() - t_start,
            }
            return None
