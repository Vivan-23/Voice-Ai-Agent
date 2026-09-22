"""NotebookLM Knowledge Provider Implementation via Model Context Protocol (MCP).

Includes granular latency instrumentation to measure:
- MCP process/stdio startup
- Session initialization
- ask_question tool execution & response wait
- Response parsing & normalization
"""

import os
import json
import re
import time
import shutil
from typing import Optional, Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.knowledge.base import BaseKnowledgeProvider


class NotebookLMKnowledgeProvider(BaseKnowledgeProvider):
    """Knowledge provider querying NotebookLM via MCP stdio transport with timing telemetry."""

    def __init__(
        self,
        notebook_id: Optional[str] = None,
        command: str = "npx",
        args: Optional[list] = None,
    ):
        self.notebook_id = notebook_id or os.getenv("NOTEBOOKLM_NOTEBOOK_ID", "coway-india-customer-support-k")
        self.command = command or os.getenv("NOTEBOOKLM_MCP_COMMAND", "npx")
        self.args = args or ["notebooklm-mcp@latest"]
        self.last_timings: Dict[str, float] = {}

    def _get_server_params(self) -> StdioServerParameters:
        npx_binary = shutil.which(self.command) or self.command
        env = dict(os.environ)
        env["HEADLESS"] = "true"
        return StdioServerParameters(
            command=npx_binary,
            args=self.args,
            env=env,
        )

    async def query(self, question: str) -> Optional[str]:
        """Query NotebookLM MCP tool 'ask_question' with granular timing instrumentation."""
        t_total_start = time.perf_counter()
        server_params = self._get_server_params()

        try:
            # 1. Measure MCP process / stdio startup
            t_conn_start = time.perf_counter()
            async with stdio_client(server_params) as (read_stream, write_stream):
                t_conn_end = time.perf_counter()
                mcp_startup_sec = t_conn_end - t_conn_start

                # 2. Measure MCP session initialization
                t_init_start = time.perf_counter()
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    t_init_end = time.perf_counter()
                    session_init_sec = t_init_end - t_init_start

                    args = {
                        "question": question,
                        "source_format": "footnotes",
                    }
                    if self.notebook_id:
                        args["notebook_id"] = self.notebook_id

                    # 3. Measure NotebookLM ask_question call (browser launch, typing, streaming response wait)
                    t_call_start = time.perf_counter()
                    result = await session.call_tool("ask_question", args)
                    t_call_end = time.perf_counter()
                    ask_question_sec = t_call_end - t_call_start

                    # 4. Measure response parsing and citation normalization
                    t_parse_start = time.perf_counter()
                    raw_text = result.content[0].text if result.content else "{}"
                    data = json.loads(raw_text)

                    if not data.get("success"):
                        self.last_timings = {
                            "mcp_startup": mcp_startup_sec,
                            "session_init": session_init_sec,
                            "ask_question_wait": ask_question_sec,
                            "parsing": time.perf_counter() - t_parse_start,
                            "total": time.perf_counter() - t_total_start,
                        }
                        return None

                    payload = data.get("data", {})
                    raw_answer = payload.get("answer", "")

                    # Clean out AI-Generated banner and Thoughts blocks
                    clean_ans = re.sub(r"^\[AI-GENERATED[^\]]*\]\s*", "", raw_answer, flags=re.DOTALL)
                    if "Thoughts" in clean_ans:
                        parts = re.split(r"(?:^|\n)Thoughts\s*\n.*?\n\n", clean_ans, flags=re.DOTALL)
                        clean_ans = parts[-1] if parts else clean_ans

                    answer = clean_ans.strip()
                    t_parse_end = time.perf_counter()
                    parsing_sec = t_parse_end - t_parse_start

                    total_sec = time.perf_counter() - t_total_start
                    self.last_timings = {
                        "mcp_startup": mcp_startup_sec,
                        "session_init": session_init_sec,
                        "ask_question_wait": ask_question_sec,
                        "parsing": parsing_sec,
                        "total": total_sec,
                    }

                    return answer if len(answer) > 5 else None

        except Exception as ex:
            self.last_timings = {
                "error": str(ex),
                "total": time.perf_counter() - t_total_start,
            }
            return None
