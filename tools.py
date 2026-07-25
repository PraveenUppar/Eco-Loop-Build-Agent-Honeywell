"""
Mock thermal model + agentic tools for the Eco-Loop closed-loop control demo.

This module has two jobs:

1. `MockBuilding` - a cheap single-zone RC-style thermal simulator that stands
   in for EnergyPlus during Hour 0-7 of the build (see PLAN.md). It's fast
   enough to iterate on the LLM agent logic without paying EnergyPlus's
   startup/step cost, and it exposes the same shape of data (zone temp,
   outdoor temp, occupancy, PMV, energy) that the real EnergyPlus path will.

2. The tool functions the LLM calls: `get_zone_state`, `get_forecast`,
   `set_zone_setpoint`. These are the only way the agent can affect the
   simulation - it never writes to simulation state directly.

Comfort band and setpoint limits are enforced here, not trusted from the
LLM. `set_zone_setpoint` clamps any out-of-band request rather than
rejecting the whole tool call, so a single bad LLM output can't push the
zone outside safe limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Hard safety limits - the LLM can never move the setpoint outside this
# range, regardless of what it asks for. This is what Hour 3-7's
# "confirm the comfort clamp actually rejects out-of-band requests" step
# is checking.
MIN_SETPOINT_C = 18.0
MAX_SETPOINT_C = 26.0

# Comfort band used for the dashboard's "did AI violate comfort" check.
# Narrower than the hard clamp - the clamp is a safety rail, this is the
# actual comfort target the agent should be reasoning about.
COMFORT_MIN_C = 20.0
COMFORT_MAX_C = 24.0


@dataclass
class ZoneState:
    zone_temp_c: float = 21.0
    outdoor_temp_c: float = 15.0
    setpoint_c: float = 21.0
    occupied: bool = False
    energy_kwh_step: float = 0.0
    pmv: float = 0.0
    minute_of_day: int = 0
    day: int = 0


class MockBuilding:
    """
    Single-zone lumped-capacitance thermal model.

    Zone temperature drifts toward a weighted blend of outdoor temp (heat
    leakage through the envelope) and the HVAC setpoint (conditioning
    effort), with thermal mass giving it inertia. Energy consumed each step
    is proportional to how hard the HVAC has to work to close the gap
    between current temp and setpoint - this is what lets a smarter
    setpoint schedule actually show up as a kWh saving.
    """

    def __init__(self, step_minutes: int = 15, envelope_loss: float = 0.08,
                 hvac_gain: float = 0.4, thermal_mass: float = 0.5):
        self.step_minutes = step_minutes
        self.envelope_loss = envelope_loss  # how fast zone tracks outdoor temp
        self.hvac_gain = hvac_gain          # how fast zone tracks setpoint
        self.thermal_mass = thermal_mass    # inertia (0-1, higher = slower to change)
        self.state = ZoneState()

    def _outdoor_temp(self, day: int, minute_of_day: int) -> float:
        """Simple diurnal sine wave: cool at 4am, peak mid-afternoon."""
        hour = minute_of_day / 60.0
        base = 14.0 + 2.0 * day * 0.0  # flat across days for now, easy to extend
        swing = 8.0 * math.sin(math.pi * (hour - 9) / 12)
        return round(base + swing, 2)

    def _occupied(self, minute_of_day: int) -> bool:
        hour = minute_of_day / 60.0
        return 8.0 <= hour < 18.0

    def _pmv(self, zone_temp_c: float, occupied: bool) -> float:
        """
        Simplified PMV proxy (not full ASHRAE 55 PMV - this is a mock
        model). Centers on 22C as neutral, scaled so +-3C from neutral
        is roughly +-1.5 PMV. Only meaningful while occupied; unoccupied
        zones report 0 (no one there to feel discomfort).
        """
        if not occupied:
            return 0.0
        return round((zone_temp_c - 22.0) / 2.0, 2)

    def step(self, requested_setpoint_c: float) -> ZoneState:
        s = self.state
        setpoint = clamp_setpoint(requested_setpoint_c)

        s.outdoor_temp_c = self._outdoor_temp(s.day, s.minute_of_day)
        s.occupied = self._occupied(s.minute_of_day)

        # Drift zone temp toward outdoor (envelope loss) and setpoint (HVAC),
        # damped by thermal mass.
        drift = (self.envelope_loss * (s.outdoor_temp_c - s.zone_temp_c) +
                 self.hvac_gain * (setpoint - s.zone_temp_c))
        s.zone_temp_c = round(s.zone_temp_c + drift * (1 - self.thermal_mass), 3)

        # Energy proportional to conditioning effort this step (kWh).
        effort = abs(setpoint - s.zone_temp_c) + abs(drift)
        s.energy_kwh_step = round(effort * self.hvac_gain * (self.step_minutes / 60.0), 4)

        s.setpoint_c = setpoint
        s.pmv = self._pmv(s.zone_temp_c, s.occupied)

        s.minute_of_day += self.step_minutes
        if s.minute_of_day >= 24 * 60:
            s.minute_of_day = 0
            s.day += 1

        return s


def clamp_setpoint(requested_c: float) -> float:
    """Hard safety rail - never let a setpoint leave [MIN_SETPOINT_C, MAX_SETPOINT_C]."""
    return max(MIN_SETPOINT_C, min(MAX_SETPOINT_C, requested_c))


# --------------------------------------------------------------------------
# Tool definitions exposed to the LLM (Ollama/OpenAI tool-calling schema).
# The functions below are what actually executes when the model emits a
# tool call; TOOL_SCHEMAS is what gets sent to the model so it knows they
# exist.
# --------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_zone_state",
            "description": (
                "Get the current state of the building zone: temperature, "
                "outdoor temperature, occupancy, current setpoint, and PMV "
                "thermal comfort index."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_zone_setpoint",
            "description": (
                f"Set the HVAC setpoint temperature for the zone, in Celsius. "
                f"Requests outside [{MIN_SETPOINT_C}, {MAX_SETPOINT_C}] are "
                f"clamped to that range automatically - it is safe to call "
                f"this even if you're unsure of the exact bound."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "setpoint_c": {
                        "type": "number",
                        "description": "Desired setpoint temperature in Celsius.",
                    }
                },
                "required": ["setpoint_c"],
            },
        },
    },
]


class ToolExecutor:
    """Binds the tool schemas above to a running MockBuilding instance."""

    def __init__(self, building: MockBuilding):
        self.building = building
        self._pending_setpoint = building.state.setpoint_c

    def get_zone_state(self) -> dict:
        s = self.building.state
        return {
            "zone_temp_c": s.zone_temp_c,
            "outdoor_temp_c": s.outdoor_temp_c,
            "setpoint_c": s.setpoint_c,
            "occupied": s.occupied,
            "pmv": s.pmv,
            "comfort_band_c": [COMFORT_MIN_C, COMFORT_MAX_C],
        }

    def set_zone_setpoint(self, setpoint_c: float) -> dict:
        clamped = clamp_setpoint(setpoint_c)
        was_clamped = clamped != setpoint_c
        self._pending_setpoint = clamped
        return {
            "accepted_setpoint_c": clamped,
            "was_clamped": was_clamped,
            "requested_setpoint_c": setpoint_c,
        }

    def dispatch(self, name: str, arguments: dict) -> dict:
        if name == "get_zone_state":
            return self.get_zone_state()
        if name == "set_zone_setpoint":
            return self.set_zone_setpoint(**arguments)
        raise ValueError(f"Unknown tool: {name}")

    def consume_pending_setpoint(self) -> float:
        """Called once per loop step after the agent has finished reasoning."""
        return self._pending_setpoint
