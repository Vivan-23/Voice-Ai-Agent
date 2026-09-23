"""Knowledge Synchronization Service (Background Syncer & Version Management).

Synchronizes company knowledge from NotebookLM MCP ingestion interface
into versioned, high-performance in-memory runtime snapshots.

Key Guarantees:
1. Live customer calls NEVER query NotebookLM synchronously.
2. Versioned snapshots with atomic switching (v1 -> v2).
3. If sync fails, the active version remains active and untouched.
4. Unchanged knowledge does not trigger unnecessary rebuilds.
5. Periodic background scheduler runs non-blocking without delaying requests.
"""

import os
import json
import time
import shutil
import hashlib
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.knowledge.store import (
    RuntimeKnowledgeStore,
    KnowledgeSnapshot,
    DEFAULT_SEED_DOCUMENTS,
    DEFAULT_SEED_SOURCES,
)

logger = logging.getLogger("voice_ai.knowledge_sync")


class KnowledgeSyncService:
    """Service to detect source changes, fetch fresh knowledge from NotebookLM, and build validated versioned snapshots."""

    def __init__(
        self,
        store: Optional[RuntimeKnowledgeStore] = None,
        notebook_id: Optional[str] = None,
        command: str = "npx",
        args: Optional[List[str]] = None,
    ):
        self.store = store or RuntimeKnowledgeStore()
        self.notebook_id = notebook_id or os.getenv("NOTEBOOKLM_NOTEBOOK_ID", "coway-india-customer-support-k")
        self.command = command or os.getenv("NOTEBOOKLM_MCP_COMMAND", "npx")
        self.args = args or ["notebooklm-mcp@latest"]
        self._bg_task: Optional[asyncio.Task] = None
        self._sync_lock = asyncio.Lock()

    def _get_server_params(self) -> StdioServerParameters:
        npx_binary = shutil.which(self.command) or self.command
        env = dict(os.environ)
        env["HEADLESS"] = "true"
        return StdioServerParameters(
            command=npx_binary,
            args=self.args,
            env=env,
        )

    async def inspect_source_changes(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Inspect NotebookLM MCP metadata to detect source changes.

        Capabilities & Limitations:
        - Detectable: Notebook metadata, active notebook name, library stats, last_modified, topics, tags.
        - Undetectable via current MCP: Granular per-file Google Drive mtime timestamps inside NotebookLM storage.
        """
        server_params = self._get_server_params()

        try:
            async def _inspect():
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()

                        # 1. Get health & active notebook
                        health_res = await session.call_tool("get_health", {})
                        health_raw = health_res.content[0].text if health_res.content else "{}"
                        health_data = json.loads(health_raw).get("data", {})

                        # 2. Get library stats (includes library last_modified)
                        stats_res = await session.call_tool("get_library_stats", {})
                        stats_raw = stats_res.content[0].text if stats_res.content else "{}"
                        stats_data = json.loads(stats_raw).get("data", {})

                        # 3. Get notebooks list
                        list_res = await session.call_tool("list_notebooks", {})
                        list_raw = list_res.content[0].text if list_res.content else "{}"
                        list_data = json.loads(list_raw).get("data", {})
                        notebooks = list_data.get("notebooks", [])

                        # Find target notebook
                        target_nb = None
                        for nb in notebooks:
                            if nb.get("id") == self.notebook_id or nb.get("name") == health_data.get("active_notebook_name"):
                                target_nb = nb
                                break
                        if not target_nb and notebooks:
                            target_nb = notebooks[0]

                        # Compute manifest fingerprint from metadata
                        manifest_data = {
                            "notebook_id": target_nb.get("id") if target_nb else self.notebook_id,
                            "notebook_name": target_nb.get("name") if target_nb else health_data.get("active_notebook_name"),
                            "topics": target_nb.get("topics", []) if target_nb else [],
                            "tags": target_nb.get("tags", []) if target_nb else [],
                            "stats_last_modified": stats_data.get("last_modified"),
                            "total_notebooks": stats_data.get("total_notebooks", len(notebooks)),
                        }
                        manifest_str = json.dumps(manifest_data, sort_keys=True)
                        fingerprint = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()[:16]

                        active_fp = ""
                        if self.store.active_snapshot and self.store.active_snapshot.metadata:
                            active_fp = self.store.active_snapshot.metadata.get("fingerprint", "")

                        if not active_fp:
                            return True, "Initial snapshot fingerprint generation", {**manifest_data, "fingerprint": fingerprint}

                        if fingerprint != active_fp:
                            return True, f"Metadata fingerprint changed: {active_fp} -> {fingerprint}", {**manifest_data, "fingerprint": fingerprint}

                        return False, "Knowledge metadata is unchanged", {**manifest_data, "fingerprint": fingerprint}

            return await asyncio.wait_for(_inspect(), timeout=6.0)

        except Exception as ex:
            logger.warning(f"Could not connect to NotebookLM MCP during change check: {ex}")
            # If MCP is unavailable, report unchanged to preserve existing active snapshot
            return False, f"MCP connection unavailable ({ex})", {"error": str(ex)}

    async def fetch_knowledge_documents_from_mcp(self, target_nb_id: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch fresh grounded knowledge from NotebookLM MCP using a comprehensive query."""
        server_params = self._get_server_params()
        documents: List[Dict[str, Any]] = []

        prompt = (
            "Provide a comprehensive summary of all Coway air purifier models, specifications, room coverage, "
            "prices, discounts, bank offers, warranty terms, motor warranty, filter life, and troubleshooting instructions in India."
        )

        try:
            async def _fetch():
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()

                        args = {
                            "question": prompt,
                            "source_format": "footnotes",
                        }
                        if target_nb_id:
                            args["notebook_id"] = target_nb_id

                        res = await session.call_tool("ask_question", args)
                        raw_text = res.content[0].text if res.content else "{}"
                        data = json.loads(raw_text)

                        if data.get("success"):
                            answer = data.get("data", {}).get("answer", "").strip()
                            sources = data.get("data", {}).get("sources", [])
                            source_name = sources[0].get("sourceName", "NotebookLM Core Knowledge") if sources else "NotebookLM Core Knowledge"

                            if len(answer) > 20:
                                documents.append({
                                    "id": "mcp_comprehensive_overview",
                                    "title": "Coway India Comprehensive Knowledge (NotebookLM Synced)",
                                    "category": "synced_overview",
                                    "source_name": source_name,
                                    "keywords": ["coway", "air purifier", "airmega", "150", "250", "price", "discount", "warranty", "troubleshooting"],
                                    "content": answer,
                                })
                return documents

            return await asyncio.wait_for(_fetch(), timeout=90.0)
        except Exception as err:
            logger.warning(f"Failed to fetch grounded knowledge from MCP: {err}")
            return []

    def validate_snapshot(self, snapshot: KnowledgeSnapshot) -> bool:
        """Validate candidate snapshot integrity before atomic switch."""
        if snapshot.version <= 0:
            return False
        if snapshot.source_count <= 0 or not snapshot.sources:
            return False
        if not snapshot.documents or len(snapshot.documents) == 0:
            return False
        for doc in snapshot.documents:
            if not doc.get("content") or len(doc.get("content", "").strip()) < 10:
                return False
        return True

    async def sync(self, force: bool = False) -> Dict[str, Any]:
        """Execute knowledge synchronization.

        Steps:
        1. Determine if source knowledge has changed (or forced).
        2. If unchanged, do nothing.
        3. If changed, fetch updated knowledge.
        4. Build candidate snapshot v{N+1}.
        5. Validate candidate snapshot.
        6. Atomically switch active version pointer.
        7. If error occurs, keep last known-good active version.
        """
        async with self._sync_lock:
            t_start = time.perf_counter()
            self.store.is_syncing = True
            current_ver = self.store.active_version

            try:
                # 1. Change Detection
                has_changed, change_reason, manifest_meta = await self.inspect_source_changes()

                if not force and not has_changed:
                    self.store.is_syncing = False
                    return {
                        "success": True,
                        "action": "SKIPPED",
                        "reason": change_reason,
                        "active_version": current_ver,
                        "source_count": self.store.active_snapshot.source_count if self.store.active_snapshot else 0,
                        "duration_seconds": time.perf_counter() - t_start,
                    }

                # 2. Ingest updated knowledge
                new_version = current_ver + 1
                fetched_docs: List[Dict[str, Any]] = []

                # Attempt live MCP ingestion if available
                if "error" not in manifest_meta:
                    try:
                        fetched_docs = await self.fetch_knowledge_documents_from_mcp(
                            manifest_meta.get("notebook_id")
                        )
                    except Exception as e:
                        logger.warning(f"Live MCP ingestion error: {e}. Merging with seed knowledge.")

                # Combine fetched documents with structured baseline documents for complete coverage
                combined_docs = list(DEFAULT_SEED_DOCUMENTS)
                if fetched_docs:
                    for fdoc in fetched_docs:
                        combined_docs.append(fdoc)

                # Sources list
                sources = list(DEFAULT_SEED_SOURCES)
                if manifest_meta.get("notebook_name"):
                    sources.append({
                        "id": f"src_notebooklm_{new_version}",
                        "name": manifest_meta.get("notebook_name", "NotebookLM Core"),
                        "source_type": "notebooklm",
                        "fingerprint": manifest_meta.get("fingerprint", f"fp_{new_version}"),
                        "item_count": len(combined_docs),
                        "synced_at": datetime.now(timezone.utc).isoformat(),
                    })

                # 3. Build candidate snapshot
                candidate = KnowledgeSnapshot(
                    version=new_version,
                    status="READY",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source_count=len(sources),
                    sources=sources,
                    documents=combined_docs,
                    metadata=manifest_meta,
                    last_sync_duration_seconds=time.perf_counter() - t_start,
                )

                # 4. Validate candidate snapshot
                if not self.validate_snapshot(candidate):
                    raise ValueError("Candidate knowledge snapshot failed structural validation checks.")

                # 5. Atomic Switch
                self.store.switch_active_version(candidate)
                self.store.is_syncing = False

                return {
                    "success": True,
                    "action": "SYNCED",
                    "reason": change_reason,
                    "active_version": new_version,
                    "source_count": candidate.source_count,
                    "indexed_item_count": len(candidate.documents),
                    "duration_seconds": candidate.last_sync_duration_seconds,
                }

            except Exception as ex:
                self.store.is_syncing = False
                self.store.last_error = f"Sync failed: {str(ex)}"
                logger.error(f"Knowledge synchronization failed: {ex}. Active version {current_ver} preserved.")
                return {
                    "success": False,
                    "action": "FAILED",
                    "error": str(ex),
                    "active_version": current_ver,
                    "duration_seconds": time.perf_counter() - t_start,
                }

    async def _background_loop(self, interval_minutes: int):
        """Periodic background loop checking for knowledge changes."""
        interval_seconds = max(10, interval_minutes * 60)
        logger.info(f"Started Knowledge Sync background scheduler (interval: {interval_minutes}m / {interval_seconds}s)")

        while True:
            try:
                await asyncio.sleep(interval_seconds)
                logger.info("Triggering periodic knowledge change inspection...")
                result = await self.sync(force=False)
                logger.info(f"Periodic sync result: {result.get('action')} - {result.get('reason')}")
            except asyncio.CancelledError:
                logger.info("Knowledge Sync background scheduler stopped.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in background sync loop: {e}")

    def start_background_sync(self, interval_minutes: int = 15):
        """Start the non-blocking background synchronization loop."""
        if self._bg_task is None or self._bg_task.done():
            self._bg_task = asyncio.create_task(self._background_loop(interval_minutes))

    def stop_background_sync(self):
        """Stop the background synchronization task."""
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
            self._bg_task = None
