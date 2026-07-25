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

import grid

# Hard safety limits - the LLM can never move the setpoint outside this
# range, regardless of what it asks for. This is what Hour 3-7's
# "confirm the comfort clamp actually rejects out-of-band requests" step
# is checking.
#
# The floor is freeze protection, not a comfort number: it exists to stop
# any strategy - or any model output - from letting an empty building
# approach pipe-freezing conditions. It sits below the deepest setback the
# controller will ever deliberately command (16.0C, see
# supervisor.DEEP_SETBACK_C) precisely so that the rail and the strategy
# are independent. A rail set at the strategy's own value silently stops
# being a rail.
MIN_SETPOINT_C = 15.0
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
    cost_step: float = 0.0
    carbon_g_step: float = 0.0
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
                 hvac_gain: float = 0.4, thermal_mass: float = 0.5,
                 hvac_capacity: float = 3.0,
                 outdoor_base_c: float = 4.0, outdoor_swing_c: float = 5.0):
        self.step_minutes = step_minutes
        # Winter-ish default (-1C to 9C over the day). The earlier mild
        # profile (6C to 22C) put outdoor temperature ABOVE the indoor
        # target every afternoon, so the heating was already off during
        # the evening price peak - which left load shifting nothing to
        # shift and made the whole grid-signal question moot.
        self.outdoor_base_c = outdoor_base_c
        self.outdoor_swing_c = outdoor_swing_c
        self.envelope_loss = envelope_loss  # how fast zone tracks outdoor temp
        self.hvac_gain = hvac_gain          # how fast zone closes the setpoint gap
        self.thermal_mass = thermal_mass    # inertia (0-1, higher = slower to change)
        self.hvac_capacity = hvac_capacity  # max heat per step (limits warm-up rate)
        self.state = ZoneState()

    def _outdoor_temp(self, day: int, minute_of_day: int) -> float:
        """Simple diurnal sine wave: cool at 4am, peak mid-afternoon."""
        hour = minute_of_day / 60.0
        swing = self.outdoor_swing_c * math.sin(math.pi * (hour - 9) / 12)
        return round(self.outdoor_base_c + swing, 2)

    def _occupied(self, minute_of_day: int) -> bool:
        hour = minute_of_day / 60.0
        return 8.0 <= hour < 18.0

    def minutes_until_occupancy_change(self) -> int:
        """
        Minutes until the zone's occupancy status next flips. Exposed so
        the agent can anticipate an upcoming occupied period and preheat
        in advance - without this, the agent can only react after
        occupancy already started, which causes a cold-start comfort
        violation (observed empirically, see BUILD_LOG.md).

        Derives occupancy from `minute_of_day` via `_occupied()` rather
        than reading `state.occupied`: that flag is only refreshed inside
        `step()`, so at the moment a control decision is made it still
        describes the *previous* step. Trusting it made this function
        report 1440 minutes at exactly 08:00 - the instant occupancy
        begins - which ordered a setback right as people arrived.
        """
        minute_of_day = self.state.minute_of_day
        if self._occupied(minute_of_day):
            return (18 * 60) - minute_of_day
        target = 8 * 60
        if minute_of_day >= target:
            target += 24 * 60
        return target - minute_of_day

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

        # Heat the HVAC injects this step. The first term replaces heat lost
        # through the envelope (what it takes to HOLD the current
        # temperature); the second closes any remaining gap up to the
        # setpoint. Capped by hvac_capacity, so warming a cold zone takes
        # real time and preheating has to start early enough to matter.
        #
        # Structuring it this way (rather than a bare proportional term)
        # means the zone actually converges ON the setpoint. A plain
        # proportional controller leaves steady-state droop - setpoint 21C
        # settled at 19.2C - which quietly broke every comfort comparison,
        # since the baseline's nominal 21C was really delivering 19.2C.
        envelope_gap = s.zone_temp_c - s.outdoor_temp_c
        hold = self.envelope_loss * envelope_gap
        close_gap = self.hvac_gain * (setpoint - s.zone_temp_c)
        heat_input = max(0.0, min(self.hvac_capacity, hold + close_gap))

        drift = self.envelope_loss * (s.outdoor_temp_c - s.zone_temp_c) + heat_input
        s.zone_temp_c = round(s.zone_temp_c + drift * (1 - self.thermal_mass), 3)

        # Energy this step (kWh), heating-only degree-day model.
        #
        # Two components, and the first one matters:
        #
        #   standing_loss - heat continuously escaping through the envelope,
        #     proportional to how far the zone is held above outdoor. The
        #     HVAC must keep replacing this just to HOLD a temperature.
        #   ramp - extra heat to climb toward a setpoint above current temp.
        #
        # An earlier version charged only `abs(setpoint - zone_temp)`, which
        # made holding `setpoint == zone_temp` almost free. That is not how
        # buildings work, and it was exploitable: the LLM discovered it and
        # started echoing the current zone temperature back as its setpoint
        # to zero out its energy score. With standing loss included, a lower
        # setpoint genuinely costs less (smaller indoor-outdoor gap), which
        # is what makes unoccupied setback a real saving rather than an
        # artifact.
        s.energy_kwh_step = round(heat_input * (self.step_minutes / 60.0), 4)

        # Same kWh costs different amounts and emits different CO2
        # depending on when it's drawn - this is what makes load shifting
        # worth anything.
        s.cost_step = round(s.energy_kwh_step * grid.price_per_kwh(s.minute_of_day), 5)
        s.carbon_g_step = round(
            s.energy_kwh_step * grid.carbon_intensity(s.minute_of_day), 2
        )

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
                "outdoor temperature, occupancy, current setpoint, PMV "
                "thermal comfort index, and minutes_until_occupancy_change "
                "(minutes until the zone flips from unoccupied to occupied, "
                "or vice versa - use this to preheat before occupancy starts "
                "instead of waiting until people have already arrived)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grid_forecast",
            "description": (
                "Get electricity price and grid carbon intensity for the "
                "next few hours. Power is cheapest and cleanest overnight "
                "and around midday, and most expensive and most carbon "
                "intensive during the evening peak. Use this to decide "
                "WHEN to heat: heating early, while power is cheap, stores "
                "warmth in the building's thermal mass and lets you coast "
                "through the expensive peak without running the HVAC."
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
            # Derived from the clock, not `s.occupied`, for the same
            # staleness reason as minutes_until_occupancy_change: the
            # stored flag lags by one step at the moment a decision is
            # made, so at 08:00 it still says the zone is empty.
            "occupied": self.building._occupied(s.minute_of_day),
            "pmv": s.pmv,
            "comfort_band_c": [COMFORT_MIN_C, COMFORT_MAX_C],
            "minutes_until_occupancy_change": self.building.minutes_until_occupancy_change(),
            "price_per_kwh_now": grid.price_per_kwh(s.minute_of_day),
            "carbon_g_per_kwh_now": grid.carbon_intensity(s.minute_of_day),
            "minutes_until_price_peak": grid.minutes_until_peak(s.minute_of_day),
        }

    def get_grid_forecast(self) -> dict:
        m = self.building.state.minute_of_day
        return {
            "now": {
                "price_per_kwh": grid.price_per_kwh(m),
                "carbon_g_per_kwh": grid.carbon_intensity(m),
                "is_peak": grid.is_peak(m),
            },
            "minutes_until_price_peak": grid.minutes_until_peak(m),
            "forecast": grid.forecast(m, hours=6),
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
        if name == "get_grid_forecast":
            return self.get_grid_forecast()
        if name == "set_zone_setpoint":
            return self.set_zone_setpoint(**arguments)
        raise ValueError(f"Unknown tool: {name}")

    def begin_cycle(self) -> None:
        """
        Clear the pending setpoint before the agent starts reasoning.

        Without this, a cycle in which the model never successfully called
        `set_zone_setpoint` silently reused the previous value and reported
        it as the agent's proposal - so non-decisions were logged as
        decisions. It made the agent look like it was choosing 21.0 (the
        initial default) fifteen times when it had in fact chosen nothing.
        """
        self._pending_setpoint = None

    def consume_pending_setpoint(self) -> float | None:
        """
        The setpoint the agent asked for this cycle, or None if it never
        made a valid call. None is meaningful - `supervisor.supervise()`
        treats it as "agent produced no setpoint" and applies a
        situation-appropriate fallback instead of a stale number.
        """
        return self._pending_setpoint


# --------------------------------------------------------------------------
# Building-level tools (real EnergyPlus path).
#
# The single-zone tools above hand the agent a raw thermostat to write to.
# For a five-zone building that scales badly in both directions: the agent
# would need five numeric decisions per cycle, and each one would be an
# independent chance for a small model to emit something unusable.
#
# These tools instead expose the building as one aggregate and take one
# named strategy back. Per-zone setpoints are then derived deterministically
# by supervisor.apply_mode(), which also enforces comfort per zone on every
# timestep. The agent decides policy; the supervisor handles physics.
# --------------------------------------------------------------------------

BUILDING_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_building_state",
            "description": (
                "Get the current state of the whole building: how many zones "
                "are occupied, the coldest and average zone temperature, "
                "outdoor temperature, and four pre-evaluated conditions - "
                "building_occupied, occupancy_imminent, peak_now, "
                "peak_approaching - each already computed as true or false."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grid_forecast",
            "description": (
                "Get electricity price and grid carbon intensity for the next "
                "few hours. Power is cheapest and cleanest overnight and around "
                "midday, and most expensive and most carbon intensive during "
                "the evening peak. Heating early, while power is cheap, stores "
                "warmth in the building's thermal mass and lets the building "
                "coast through the expensive peak with the heating idle."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class BuildingToolExecutor:
    """
    Serves building-level state read out of the running EnergyPlus model.

    `update()` is called by the run loop with the values it just read from
    the simulation, immediately before the agent is invoked - so a
    `get_building_state` call always reflects the live simulation rather
    than a stale or synthetic snapshot.
    """

    def __init__(self):
        self._state: dict = {}
        self.calls: list[str] = []

    def update(self, state: dict) -> None:
        self._state = dict(state)

    def get_building_state(self) -> dict:
        return dict(self._state)

    def get_grid_forecast(self) -> dict:
        minute = int(self._state.get("minute_of_day", 0))
        return {
            "now": {
                "price_per_kwh": grid.price_per_kwh(minute),
                "carbon_g_per_kwh": grid.carbon_intensity(minute),
                "is_peak": grid.is_peak(minute),
            },
            "minutes_until_price_peak": grid.minutes_until_peak(minute),
            "forecast": grid.forecast(minute, hours=6),
        }

    def begin_cycle(self) -> None:
        self.calls = []

    def dispatch(self, name: str, arguments: dict) -> dict:
        self.calls.append(name)
        if name == "get_building_state":
            return self.get_building_state()
        if name == "get_grid_forecast":
            return self.get_grid_forecast()
        raise ValueError(f"Unknown tool: {name}")


class LiveStateExecutor(ToolExecutor):
    """
    ToolExecutor whose `get_zone_state` serves externally-supplied state.

    Used for the real EnergyPlus modes. Previously those built a
    `ToolExecutor(MockBuilding())`, so when the LLM called
    `get_zone_state` it received a mock building that was never stepped -
    frozen at its initial values. The agent was reasoning about a
    stationary toy while EnergyPlus ran separately, which defeats the
    point of the closed loop.

    The run loop calls `update(state)` with the values it just read out of
    EnergyPlus, immediately before invoking the agent.
    """

    def __init__(self, building: MockBuilding):
        super().__init__(building)
        self._live: dict | None = None

    def update(self, state: dict) -> None:
        self._live = dict(state)

    def get_zone_state(self) -> dict:
        if self._live is None:
            return super().get_zone_state()
        return dict(self._live)
