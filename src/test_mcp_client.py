"""Standalone MCP client exercising all five tools over stdio.

Also starts a real EnergyPlus run in a background thread and calls the tools
while it is still going, to prove the tool layer works against a live
simulation rather than only against a finished one.

Run: python src/test_mcp_client.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config as cfg
import shared_state
from controllers import RuleBasedController
from eplus_runner import EnergyPlusRunner

SERVER = Path(__file__).resolve().parent / "mcp_server.py"
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def payload(result) -> dict:
    """FastMCP returns content blocks; pull the JSON out of the first one."""
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    return {}


def start_background_sim() -> threading.Thread:
    """Run a full simulation in a thread so tools can be called mid-run."""
    def _run():
        runner = EnergyPlusRunner(
            controller=RuleBasedController(),
            outdir=cfg.OUT / "mcp_live",
            verbose=False,
        )
        runner.run()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


async def main() -> int:
    shared_state.StateStore.clear()

    print("=== starting background EnergyPlus run ===")
    sim = start_background_sim()
    # Give the simulation a moment to get past warmup and publish state.
    for _ in range(100):
        if shared_state.read_state():
            break
        time.sleep(0.05)
    live = shared_state.read_state() is not None
    print(f"  simulation publishing state: {live}, thread alive: {sim.is_alive()}")

    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n=== list_tools ===")
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            for t in tools.tools:
                first_line = (t.description or "").strip().splitlines()[0]
                print(f"  {t.name:24} {first_line}")
            check("all five tools registered", len(names) == 5, ", ".join(names))

            print("\n=== get_building_state ===")
            state = payload(await session.call_tool("get_building_state", {}))
            check("returns live state", state.get("ok") is True,
                  f"sim_time={state.get('sim_time')} step={state.get('step')}")
            if state.get("ok"):
                temps = state["zone_temps_c"]
                sane = all(10 < v < 40 for v in temps.values())
                check("zone temps sane", sane, json.dumps(temps))

            print("\n=== list_zones ===")
            zones = payload(await session.call_tool("list_zones", {}))
            check("five conditioned zones", len(zones.get("zones", [])) == 5,
                  f"total area {zones.get('total_floor_area_m2')} m2")

            print("\n=== get_energy_summary ===")
            summary = payload(await session.call_tool(
                "get_energy_summary", {"window_hours": 4}))
            check("energy summary returned", summary.get("ok") is True,
                  f"kwh={summary.get('kwh')} peak_kw={summary.get('peak_kw')}")

            print("\n=== set_setpoints (valid) ===")
            good = payload(await session.call_tool(
                "set_setpoints",
                {"heating_c": 20.0, "cooling_c": 25.0, "reason": "mild afternoon"}))
            check("valid setpoints accepted", good.get("ok") is True,
                  json.dumps(good.get("applied")))

            print("\n=== set_setpoints (out of range) ===")
            bad = payload(await session.call_tool(
                "set_setpoints", {"heating_c": 35.0, "cooling_c": 5.0}))
            check("out-of-range rejected", bad.get("ok") is False,
                  (bad.get("violations") or [""])[0])
            check("rejection is actionable",
                  bool(bad.get("limits")) and bool(bad.get("would_clamp_to")),
                  json.dumps(bad.get("would_clamp_to")))

            print("\n=== set_setpoints (bad zone) ===")
            badzone = payload(await session.call_tool(
                "set_setpoints",
                {"heating_c": 21.0, "cooling_c": 24.0, "zone": "KITCHEN"}))
            check("unknown zone rejected", badzone.get("ok") is False,
                  str(badzone.get("error"))[:70])

            print("\n=== read_simulation_errors ===")
            errs = payload(await session.call_tool(
                "read_simulation_errors", {"max_lines": 15}))
            check("error log compressed", errs.get("ok") is True,
                  f"{errs.get('lines_before')} -> {errs.get('lines_after')} lines, "
                  f"{errs.get('compression_ratio')}x fewer tokens")
            check("output stays short", (errs.get("lines_after") or 999) < 40,
                  f"{errs.get('lines_after')} lines")

    sim.join(timeout=120)
    print(f"\nbackground simulation finished: {not sim.is_alive()}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("all MCP tool tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
