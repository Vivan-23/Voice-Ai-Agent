"""NotebookLM MCP Knowledge Provider Implementation.

Connects to the local NotebookLM MCP server over stdio transport to retrieve
factual, grounded responses and citations from user-uploaded company knowledge bases.
Shares the persistent authentication state, cookies, and local library seamlessly.
"""

import asyncio
import json
import os
import shutil
from typing import Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.interfaces.knowledge import (
    BaseKnowledgeService,
    KnowledgeItem,
    KnowledgeQueryResult,
)
from app.core.exceptions import KnowledgeRetrievalException
from app.core.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


class NotebookLMMCPKnowledgeService(BaseKnowledgeService):
    """Knowledge service grounded on NotebookLM via Model Context Protocol (MCP)."""

    def __init__(
        self,
        notebook_id: Optional[str] = None,
        notebook_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.notebook_id = notebook_id or settings.notebooklm_notebook_id
        self.notebook_url = notebook_url or settings.notebooklm_notebook_url

    def _get_server_params(self) -> StdioServerParameters:
        """Construct server parameters resolving npx executable across Windows/POSIX."""
        settings = get_settings()
        npx_binary = shutil.which(settings.notebooklm_mcp_command) or settings.notebooklm_mcp_command

        env = dict(os.environ)
        env["HEADLESS"] = "true"

        return StdioServerParameters(
            command=npx_binary,
            args=settings.notebooklm_mcp_args,
            env=env,
        )

    async def query(
        self, question: str, session_id: Optional[str] = None
    ) -> KnowledgeQueryResult:
        """Query NotebookLM MCP via ask_question tool."""
        logger.info(f"Querying NotebookLM knowledge base: '{question[:60]}...'")
        server_params = self._get_server_params()

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    args = {
                        "question": question,
                        "source_format": "footnotes",
                    }
                    if self.notebook_id:
                        args["notebook_id"] = self.notebook_id
                    if session_id:
                        args["session_id"] = session_id

                    result = await session.call_tool("ask_question", args)
                    raw_text = result.content[0].text if result.content else "{}"
                    data = json.loads(raw_text)

                    if not data.get("success"):
                        error_msg = data.get("error", "Unknown error from NotebookLM MCP")
                        raise KnowledgeRetrievalException(error_msg)

                    payload = data.get("data", {})
                    raw_answer = payload.get("answer", "")
                    
                    # Clean out NotebookLM MCP banner and internal Thoughts
                    clean_ans = re.sub(r"^\[AI-GENERATED[^\]]*\]\s*", "", raw_answer, flags=re.DOTALL)
                    if "Thoughts" in clean_ans:
                        parts = re.split(r"(?:^|\n)Thoughts\s*\n.*?\n\n", clean_ans, flags=re.DOTALL)
                        clean_ans = parts[-1] if parts else clean_ans
                    answer = clean_ans.strip()

                    raw_sources = payload.get("sources", [])

                    items: List[KnowledgeItem] = []
                    for s in raw_sources:
                        items.append(
                            KnowledgeItem(
                                source_title=s.get("sourceName", "Document"),
                                content=s.get("sourceText", ""),
                                confidence=1.0,
                                url=payload.get("notebook_url"),
                            )
                        )

                    return KnowledgeQueryResult(
                        query=question,
                        answer=answer,
                        items=items,
                        is_grounded=True,
                    )
        except Exception as e:
            err_details = [f"{type(e).__name__}: {e}"]
            if hasattr(e, "exceptions"):
                for sub_e in e.exceptions:
                    err_details.append(f"  -> SubException: {type(sub_e).__name__}: {sub_e}")
            if getattr(e, "__cause__", None):
                err_details.append(f"  -> Cause: {type(e.__cause__).__name__}: {e.__cause__}")
            if getattr(e, "__context__", None):
                err_details.append(f"  -> Context: {type(e.__context__).__name__}: {e.__context__}")

            full_err_msg = "\n".join(err_details)
            logger.error(f"Error querying NotebookLM MCP:\n{full_err_msg}", exc_info=True)
            raise KnowledgeRetrievalException(f"Failed to query knowledge base: {e}")

    async def get_notebooks(self) -> list:
        """Retrieve registered notebooks from local MCP library."""
        server_params = self._get_server_params()
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                res = await session.call_tool("list_notebooks", {})
                raw_text = res.content[0].text if res.content else "{}"
                data = json.loads(raw_text)
                return data.get("data", {}).get("notebooks", [])

    async def health_check(self) -> bool:
        """Verify MCP server process health and authentication."""
        server_params = self._get_server_params()
        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    res = await session.call_tool("get_health", {})
                    raw_text = res.content[0].text if res.content else "{}"
                    data = json.loads(raw_text)
                    return bool(data.get("data", {}).get("authenticated", False))
        except Exception as e:
            logger.error(f"NotebookLM MCP health check failed: {e}")
            return False
