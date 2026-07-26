"""MCP server exposing the building to the supervisory agent.

Five tools over stdio. The server runs as its own process and talks to the
live EnergyPlus simulation through the atomic JSON files in shared_state.py,
so tool calls work while a simulation is mid-run.

Run standalone:  python src/mcp_server.py
Test client:     python src/test_mcp_client.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

import config as cfg
import shared_state
from log_tools import compress_err_file
from policy import clamp_pair

mcp = FastMCP("eco-loop-building")

# Zone floor areas from the 5ZoneAirCooled geometry, m^2.
ZONE_AREAS = {
    "SPACE1-1": 99.16,
    "SPACE2-1": 42.73,
    "SPACE3-1": 96.48,
    "SPACE4-1": 42.73,
    "SPACE5-1": 182.49,
}

NO_SIM = ("No simulation state available. Start a run with "
          "`python src/run_experiment.py --mode agent` first.")


@mcp.tool()
def get_building_state() -> dict[str, Any]:
    """Current sensor readings and the setpoint policy now in force.

    Returns zone air temperatures and relative humidity, outdoor drybulb,
    occupancy, energy used so far, and the active heating/cooling setpoints.
    """
    state = shared_state.read_state()
    if not state:
        return {"ok": False, "error": NO_SIM}

    return {
        "ok": True,
        "sim_time": state["sim_time"],
        "step": state["step"],
        "outdoor_temp_c": state["outdoor_temp"],
        "zone_temps_c": state["zone_temps"],
        "zone_rh_pct": state["zone_rh"],
        "occupied": state["occupied"],
        "occupancy_fraction": state["occupancy"],
        "cumulative_kwh": state["cumulative_kwh"],
        "cumulative_hvac_kwh": state["cumulative_hvac_kwh"],
        "current_policy": state["current_policy"],
        "comfort_band_c": [cfg.COMFORT_LOW, cfg.COMFORT_HIGH],
    }


@mcp.tool()
def get_energy_summary(window_hours: float = 4.0) -> dict[str, Any]:
    """Energy used over a trailing window, with peak demand and baseline delta.

    Args:
        window_hours: length of the trailing window in simulated hours.
    """
    state = shared_state.read_state()
    if not state:
        return {"ok": False, "error": NO_SIM}
    if window_hours <= 0:
        return {"ok": False, "error": "window_hours must be positive"}

    summary = shared_state.summarise_energy(state, window_hours)
    summary["ok"] = True
    return summary


@mcp.tool()
def set_setpoints(heating_c: float, cooling_c: float,
                  zone: str = "all", reason: str = "") -> dict[str, Any]:
    """Request new thermostat setpoints. Validated and clamped before use.

    Rejects values outside the safety envelope rather than silently applying
    them: heating must be within [18, 22] C, cooling within [23, 27] C, and
    the deadband at least 1.5 C.

    Args:
        heating_c: requested heating setpoint, degrees C.
        cooling_c: requested cooling setpoint, degrees C.
        zone: a zone name, or "all" to apply building-wide.
        reason: short justification, recorded in the decision log.
    """
    if zone != "all" and zone not in cfg.ZONES:
        return {"ok": False,
                "error": f"unknown zone {zone!r}. Valid zones: "
                         f"{', '.join(cfg.ZONES)}, or 'all'."}

    clamped_h, clamped_c, violations = clamp_pair(heating_c, cooling_c)

    # Context-aware check: the 18-27 C envelope is safe for an empty building,
    # but while people are present the setpoints must also keep zones inside
    # the comfort band. Rejecting here (rather than silently clamping) gives
    # the agent a specific error it can correct against.
    live = shared_state.read_state() or {}
    if live.get("occupied"):
        if clamped_c > cfg.COMFORT_HIGH:
            violations.append(
                f"building is OCCUPIED: cooling_c {clamped_c} exceeds the "
                f"comfort limit {cfg.COMFORT_HIGH} C")
        if clamped_h < cfg.COMFORT_LOW:
            violations.append(
                f"building is OCCUPIED: heating_c {clamped_h} is below the "
                f"comfort limit {cfg.COMFORT_LOW} C")
    elif live:
        # Mirror image: with nobody in the building, any setpoint tight enough
        # to start equipment is wasted electricity.
        if clamped_c < cfg.SETBACK_COOLING_MIN:
            violations.append(
                f"building is EMPTY: cooling_c {clamped_c} is below "
                f"{cfg.SETBACK_COOLING_MIN} C, which would run the AC for "
                f"nobody. Raise it.")
        if clamped_h > cfg.SETBACK_HEATING_MAX:
            violations.append(
                f"building is EMPTY: heating_c {clamped_h} is above "
                f"{cfg.SETBACK_HEATING_MAX} C, which would run the heater for "
                f"nobody. Lower it.")

    if violations:
        # Surface the specific problem so the agent can self-correct.
        return {
            "ok": False,
            "error": "setpoints outside the safety envelope",
            "violations": violations,
            "requested": {"heating_c": heating_c, "cooling_c": cooling_c},
            "would_clamp_to": {"heating_c": clamped_h, "cooling_c": clamped_c},
            "limits": {
                "heating_c": [cfg.HEATING_MIN, cfg.HEATING_MAX],
                "cooling_c": [cfg.COOLING_MIN, cfg.COOLING_MAX],
                "min_deadband_c": cfg.MIN_DEADBAND,
                "occupied_comfort_band_c": [cfg.COMFORT_LOW, cfg.COMFORT_HIGH],
                "currently_occupied": bool(live.get("occupied")),
            },
        }

    shared_state.write_pending_policy({
        "heating_c": clamped_h,
        "cooling_c": clamped_c,
        "zone": zone,
        "reason": reason[:300],
    })
    return {
        "ok": True,
        "applied": {"heating_c": clamped_h, "cooling_c": clamped_c, "zone": zone},
        "deadband_c": round(clamped_c - clamped_h, 2),
    }


@mcp.tool()
def read_simulation_errors(max_lines: int = 20) -> dict[str, Any]:
    """Severity-filtered, deduplicated summary of the EnergyPlus error log.

    Raw .err files repeat the same warning once per timestep. This filters to
    Warning/Severe/Fatal, collapses repeats into counts, and keeps the top
    entries by severity, so the caller never receives a raw log.

    Args:
        max_lines: maximum number of distinct issues to return.
    """
    candidates = sorted(
        cfg.OUT.glob("*/eplusout.err"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"ok": False, "error": "no eplusout.err found under out/"}

    result = compress_err_file(candidates[0], max_lines=max_lines)
    result["source"] = str(candidates[0].relative_to(cfg.ROOT))
    return result


@mcp.tool()
def list_zones() -> dict[str, Any]:
    """Conditioned zone names and floor areas."""
    state = shared_state.read_state()
    temps = (state or {}).get("zone_temps", {})
    return {
        "ok": True,
        "zones": [
            {
                "name": z,
                "floor_area_m2": ZONE_AREAS.get(z),
                "current_temp_c": temps.get(z),
            }
            for z in cfg.ZONES
        ],
        "excluded": {
            cfg.PLENUM: "unconditioned return plenum, not under thermostat control"
        },
        "total_floor_area_m2": round(sum(ZONE_AREAS.values()), 2),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
