"""Call async MCP tools from inside the synchronous EnergyPlus callback.

The EnergyPlus callback is a plain synchronous function, but the MCP client
session is asyncio-based and must stay open for the life of the run. So the
session lives on its own event loop in a background thread, and the callback
submits coroutines to it with run_coroutine_threadsafe.

This keeps the agent talking to a real MCP server over stdio rather than
importing the tool functions directly.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"


class MCPBridge:
    """Synchronous facade over an MCP stdio session."""

    def __init__(self, server_path: Path | None = None, startup_timeout: float = 45.0):
        self.server_path = Path(server_path or SERVER_PATH)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._session: ClientSession | None = None
        self._error: BaseException | None = None
        self.tool_names: list[str] = []
        self.call_count = 0

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            raise RuntimeError(f"MCP server did not start: {self._error}")
        if self._error:
            raise RuntimeError(f"MCP server failed: {self._error}")

    # -- background loop -------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:       # noqa: BLE001 - surfaced to caller
            self._error = exc
            self._ready.set()
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        params = StdioServerParameters(
            command=sys.executable, args=[str(self.server_path)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.tool_names = sorted(t.name for t in tools.tools)
                self._session = session
                self._ready.set()
                # Hold the session open until close() is called.
                while not self._stop.is_set():
                    await asyncio.sleep(0.05)

    # -- sync API --------------------------------------------------------
    @staticmethod
    def _payload(result) -> dict[str, Any]:
        if getattr(result, "structuredContent", None):
            sc = result.structuredContent
            return sc.get("result", sc) if isinstance(sc, dict) else sc
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"ok": False, "raw": text}
        return {"ok": False, "error": "empty tool response"}

    def call(self, name: str, args: dict[str, Any] | None = None,
             timeout: float = 30.0) -> dict[str, Any]:
        """Invoke an MCP tool and return its parsed JSON payload."""
        if self._session is None:
            return {"ok": False, "error": "MCP session not ready"}
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, args or {}), self._loop)
        try:
            result = future.result(timeout)
        except Exception as exc:                        # noqa: BLE001
            return {"ok": False, "error": f"tool {name} failed: {exc}"}
        self.call_count += 1
        return self._payload(result)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def __enter__(self) -> "MCPBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    with MCPBridge() as bridge:
        print("tools:", bridge.tool_names)
        print("zones:", json.dumps(bridge.call("list_zones"))[:200])
        print("state:", json.dumps(bridge.call("get_building_state"))[:200])
