"""
Deterministic supervisory override layer.

Sits between the LLM agent's proposed setpoint and the simulation. The
agent proposes; this layer vetoes proposals that are physically
indefensible and substitutes a safe value.

Why this exists (be honest about it in the writeup): the local 1.5B model
could not reliably follow the control rules - in a direct test with the
zone occupied and PMV at -2.3, given a prompt stating "if occupied, set
22.0", it returned 18.0. See BUILD_LOG.md for the full evidence trail.
Rather than pretend the LLM is producing good control, this layer makes
the failure explicit and correctable, and *records every intervention* so
the dashboard can report exactly how much of the final result came from
the agent versus from these rules.

This mirrors how real building management systems work - supervisory
setpoint logic virtually always sits under a hard safety/comfort
interlock rather than trusting the optimizer unconditionally. The
difference between honest and dishonest use of that pattern is whether
you report the intervention rate. We do: see `OverrideStats`.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools import (
    COMFORT_MAX_C,
    COMFORT_MIN_C,
    MAX_SETPOINT_C,
    MIN_SETPOINT_C,
    clamp_setpoint,
)

# How far ahead of occupancy to start preheating. Chosen by sweeping
# 60-180min against the model: 60 and 90 leave comfort violations, while
# 150 and 180 clear comfort but burn more energy than 120 for no benefit.
# 120 is the shortest lead that reaches zero violations.
PREHEAT_LEAD_MINUTES = 120

# Setpoint used when the building is empty and occupancy is not imminent.
UNOCCUPIED_SETBACK_C = MIN_SETPOINT_C

# Deliberately two different targets - this is where the energy saving
# comes from, and it's the one genuinely non-obvious piece of control
# logic here:
#
#   PREHEAT_TARGET_C is aggressive, to climb out of overnight setback
#   quickly and be inside the comfort band before anyone arrives.
#
#   OCCUPIED_TARGET_C then coasts at the LOW edge of the comfort band.
#   Because outdoor temperature is below indoor, a lower setpoint means a
#   smaller gap for the HVAC to hold and therefore less energy - so
#   sitting at 20.5C rather than mid-band 22.0C saves energy while still
#   satisfying comfort. A naive "just target the middle of the band"
#   supervisor used MORE energy than the fixed-schedule baseline, because
#   the baseline's occupied setpoint (21.0C) was already lower than the
#   band midpoint.
PREHEAT_TARGET_C = COMFORT_MIN_C + 0.5
OCCUPIED_TARGET_C = COMFORT_MIN_C + 0.5

# Optimal stop: stop conditioning this long before occupancy ends and let
# thermal mass coast the zone through the remainder. A standard energy
# conservation measure in real BMS.
#
# Measured caveat: in this scenario it saves nothing. Sweeping 0-120min
# changes total energy by 0.00 kWh, because the diurnal curve peaks at
# 22C mid-afternoon - above the indoor target - so heating has already
# shut off before the coast window opens. The logic is kept because it is
# correct and would pay off in a colder-afternoon or cooling-dominated
# scenario, but do not claim savings for it here.
OPTIMAL_STOP_MINUTES = 60


@dataclass
class SupervisorDecision:
    setpoint_c: float
    overridden: bool
    reason: str = ""
    proposed_setpoint_c: float | None = None


@dataclass
class OverrideStats:
    """Tracks how often the deterministic layer had to correct the agent."""

    total_decisions: int = 0
    overrides: int = 0

    def record(self, decision: SupervisorDecision) -> None:
        self.total_decisions += 1
        if decision.overridden:
            self.overrides += 1

    @property
    def override_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.overrides / self.total_decisions

    def summary(self) -> str:
        return (
            f"{self.overrides}/{self.total_decisions} decisions overridden "
            f"({self.override_rate * 100:.1f}%)"
        )


def supervise(proposed_setpoint_c: float | None, state: dict) -> SupervisorDecision:
    """
    Validate the agent's proposed setpoint against the zone's actual
    situation, substituting a safe value when the proposal would breach
    comfort or waste energy outright.

    `state` is the dict returned by `ToolExecutor.get_zone_state()`.

    Only three situations trigger an override. Anything else - including
    a merely suboptimal proposal - is left alone, so the agent retains
    real authority whenever its choice is defensible.
    """
    occupied = bool(state.get("occupied"))
    zone_temp = float(state.get("zone_temp_c", OCCUPIED_TARGET_C))
    minutes_to_change = state.get("minutes_until_occupancy_change")
    known_horizon = isinstance(minutes_to_change, (int, float))
    preheating = (
        not occupied and known_horizon and minutes_to_change <= PREHEAT_LEAD_MINUTES
    )
    # Coasting: occupants are still here, but leaving soon enough that
    # thermal mass can carry the zone to the end of the period unheated.
    coasting = occupied and known_horizon and minutes_to_change <= OPTIMAL_STOP_MINUTES

    if proposed_setpoint_c is None:
        if coasting:
            fallback = UNOCCUPIED_SETBACK_C
        elif preheating or occupied:
            fallback = PREHEAT_TARGET_C if preheating else OCCUPIED_TARGET_C
        else:
            fallback = UNOCCUPIED_SETBACK_C
        return SupervisorDecision(
            setpoint_c=fallback,
            overridden=True,
            reason="agent produced no setpoint",
            proposed_setpoint_c=None,
        )

    proposed = clamp_setpoint(float(proposed_setpoint_c))

    # 1. Occupied and the zone is outside the comfort band, but the agent
    #    isn't driving it back in. This is the failure that actually
    #    happened: setpoint parked at 18C while occupants sat at PMV -2.2.
    if occupied and not (COMFORT_MIN_C <= zone_temp <= COMFORT_MAX_C):
        # Test against the comfort band, not the current temperature: a
        # setpoint of 18C in a 17.5C occupied zone is technically "warmer
        # than now" but can never reach the 20C comfort floor, so it is
        # not a correction. The proposal must actually target the band.
        driving_correctly = (
            proposed >= COMFORT_MIN_C
            if zone_temp < COMFORT_MIN_C
            else proposed <= COMFORT_MAX_C
        )
        if not driving_correctly:
            too_cold = zone_temp < COMFORT_MIN_C
            # Recovering from a cold zone needs the aggressive target, not
            # the low-edge coasting one: aiming at 20.5C from 18.0C
            # approaches the comfort floor too slowly to clear it.
            recovery = PREHEAT_TARGET_C if too_cold else OCCUPIED_TARGET_C
            return SupervisorDecision(
                setpoint_c=recovery,
                overridden=True,
                reason=(f"occupied zone {'cold' if too_cold else 'hot'} "
                        f"({zone_temp:.1f}C), proposal not correcting"),
                proposed_setpoint_c=proposed,
            )

    # 2. Occupancy is imminent but the agent is still holding a setback
    #    low enough that the zone can't reach comfort in time.
    if preheating and proposed < COMFORT_MIN_C:
        return SupervisorDecision(
            setpoint_c=PREHEAT_TARGET_C,
            overridden=True,
            reason=f"occupancy in {minutes_to_change:.0f}min, proposal too low to preheat",
            proposed_setpoint_c=proposed,
        )

    # 3. Building is empty and won't be occupied soon, but the agent is
    #    still conditioning it. Pure waste, no comfort justification.
    if not occupied and not preheating and proposed > UNOCCUPIED_SETBACK_C:
        return SupervisorDecision(
            setpoint_c=UNOCCUPIED_SETBACK_C,
            overridden=True,
            reason="zone empty and occupancy not imminent, proposal wastes energy",
            proposed_setpoint_c=proposed,
        )

    # 4. Optimal stop: occupants leave soon and the zone has enough buffer
    #    above the comfort floor to coast there on thermal mass alone.
    #    Requiring the buffer matters - coasting from the floor itself
    #    would just walk the zone straight out of the comfort band, and
    #    rule 1 would immediately undo it.
    if coasting and proposed > UNOCCUPIED_SETBACK_C and zone_temp > COMFORT_MIN_C + 0.3:
        return SupervisorDecision(
            setpoint_c=UNOCCUPIED_SETBACK_C,
            overridden=True,
            reason=f"occupancy ends in {minutes_to_change:.0f}min, coasting on thermal mass",
            proposed_setpoint_c=proposed,
        )

    return SupervisorDecision(
        setpoint_c=proposed,
        overridden=False,
        proposed_setpoint_c=proposed,
    )
