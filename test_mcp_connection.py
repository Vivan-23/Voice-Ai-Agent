"""Integration test script to verify NotebookLM MCP integration end-to-end from Python."""

import asyncio
import os
import shutil
import sys
import json

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_end_to_end_verification():
    print("=" * 70)
    print("NOTEBOOKLM MCP END-TO-END PYTHON INTEGRATION TEST")
    print("=" * 70)

    # 1. Resolve executable
    npx_cmd = shutil.which("npx") or "npx"
    print(f"\n[Step 1] Transport Setup")
    print(f"  Command: {npx_cmd}")
    print(f"  Transport: stdio (JSON-RPC over stdin/stdout)")

    env = dict(os.environ)
    env["HEADLESS"] = "true"

    server_params = StdioServerParameters(
        command=npx_cmd,
        args=["notebooklm-mcp@latest"],
        env=env,
    )

    # 2. Connect via stdio client
    print("\n[Step 2] Connecting to NotebookLM MCP Server...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("  [OK] MCP ClientSession initialized successfully")

            # 3. Call get_health
            print("\n[Step 3] Calling 'get_health'...")
            health_res = await session.call_tool("get_health", {})
            health_text = health_res.content[0].text
            health_data = json.loads(health_text)
            auth_status = health_data.get("data", {}).get("authenticated")
            active_nb = health_data.get("data", {}).get("active_notebook_name")
            print(f"  [OK] Health Status: {health_data.get('data', {}).get('status')}")
            print(f"  [OK] Authenticated: {auth_status}")
            print(f"  [OK] Active Notebook: {active_nb}")

            # 4. Call list_notebooks
            print("\n[Step 4] Calling 'list_notebooks'...")
            list_res = await session.call_tool("list_notebooks", {})
            list_text = list_res.content[0].text
            list_data = json.loads(list_text)
            notebooks = list_data.get("data", {}).get("notebooks", [])
            print(f"  [OK] Found {len(notebooks)} notebook(s) in local library:")
            for nb in notebooks:
                print(f"    - ID: {nb.get('id')}")
                print(f"      Name: {nb.get('name')}")
                print(f"      Topics: {', '.join(nb.get('topics', []))}")

            target_notebook_id = notebooks[0].get("id") if notebooks else None

            # 5. Call ask_question
            question = "What products and services does this company provide?"
            print(f"\n[Step 5] Calling 'ask_question' on notebook '{target_notebook_id}'...")
            print(f"  Query: \"{question}\"")
            print("  Waiting for Gemini 2.5 grounded synthesis via NotebookLM...")

            ask_res = await session.call_tool(
                "ask_question",
                {
                    "notebook_id": target_notebook_id,
                    "question": question,
                    "source_format": "footnotes",
                },
            )

            ask_text = ask_res.content[0].text
            ask_data = json.loads(ask_text)

            if not ask_data.get("success"):
                print(f"  [ERROR] querying notebook: {ask_data.get('error')}")
                return False

            data = ask_data.get("data", {})
            answer = data.get("answer", "")
            sources = data.get("sources", [])

            print("\n" + "=" * 70)
            print("RETURNED GROUNDED ANSWER:")
            print("=" * 70)
            print(answer[:1500] + ("..." if len(answer) > 1500 else ""))

            print("\n" + "=" * 70)
            print(f"EXTRACTED SOURCES / CITATIONS ({len(sources)} sources):")
            print("=" * 70)
            for s in sources[:5]:
                print(f"  {s.get('marker')} {s.get('sourceName')}")
                preview = s.get('sourceText', '').replace('\n', ' ')[:120]
                print(f"     \"{preview}...\"\n")

            print("=" * 70)
            print("END-TO-END VERIFICATION RESULT: SUCCESS [OK]")
            print("=" * 70)
            return True


if __name__ == "__main__":
    success = asyncio.run(run_end_to_end_verification())
    if not success:
        sys.exit(1)
