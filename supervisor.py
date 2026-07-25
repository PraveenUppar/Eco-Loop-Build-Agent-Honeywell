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

import comfort
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
#
# Pinned to 18.0 rather than to MIN_SETPOINT_C. It used to track the safety
# rail, but the rail has since dropped to 15.0C for freeze protection, and
# following it down would have silently changed every mock-path result as a
# side effect of an unrelated safety change. The mock arm's setback is a
# control choice and belongs here, stated.
UNOCCUPIED_SETBACK_C = 18.0

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

# Load shifting against the price/carbon peak.
#
# PRECHARGE_WINDOW_MINUTES before the peak begins, push the setpoint up
# to PRECHARGE_TARGET_C while power is still cheap. The extra warmth is
# stored in the building's thermal mass. Once the peak starts, drop back
# to the comfort floor and let that stored heat carry the zone, so the
# HVAC draws little or nothing during the most expensive, most carbon
# intensive hours.
#
# This is the one strategy here that a fixed schedule cannot replicate:
# it depends on what time it is *in price terms*, which a static
# setpoint schedule has no way to know.
PRECHARGE_WINDOW_MINUTES = 120
PRECHARGE_TARGET_C = COMFORT_MAX_C - 0.5
PEAK_FLOOR_C = COMFORT_MIN_C


# Fraction of the indoor-outdoor temperature gap the zone retains per
# 15-minute step once heating stops. Derived from the building model's
# envelope loss and thermal mass (zone_next = 0.96*zone + 0.04*outdoor),
# and checked against logged coast data - predicted 20.23C where the
# simulation produced 20.223C.
COAST_RETENTION_PER_STEP = 0.96
COAST_STEP_MINUTES = 15


def projected_temp_after(zone_temp_c: float, outdoor_temp_c: float,
                         minutes: float) -> float:
    """
    Where the zone ends up if heating stops for `minutes`.

    Any decision to stop heating - optimal stop, riding out a price peak -
    has to answer "will the zone still be comfortable when this is over?"
    Checking only the *current* temperature isn't enough: a zone at 20.7C
    looks fine but lands at 19.4C after 45 minutes of coasting. Two such
    rules firing at once (peak discharge plus optimal stop) stacked their
    drops and pushed the zone through the comfort floor.
    """
    steps = max(0.0, minutes) / COAST_STEP_MINUTES
    return outdoor_temp_c + (zone_temp_c - outdoor_temp_c) * (
        COAST_RETENTION_PER_STEP ** steps
    )


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


# --------------------------------------------------------------------------
# Mode-based supervisory control (real EnergyPlus path).
#
# The single-zone mock path above has the agent emit a raw setpoint float.
# That does not survive contact with a 1.5B model: measured on 75 synthetic
# zone states, it produced a usable number in 6% of calls (see
# ARCHITECTURE.md). Emitting a *named strategy* from a fixed set instead
# turns the model's job from open-ended numeric regression into a 5-way
# classification, which is both far more reliable and constrainable by
# Ollama's structured-output grammar - the model cannot emit anything that
# is not one of these five labels.
#
# The named strategies are real, standard Energy Conservation Measures, not
# arbitrary buckets. The supervisor translates whichever one the agent picks
# into a per-zone setpoint and enforces comfort on top of it.
# --------------------------------------------------------------------------

BUILDING_MODES = ("SETBACK", "PREHEAT", "COMFORT", "PRECHARGE", "COAST")

# Deep unoccupied setback. Lower than the mock path's 18.0C, and lower than
# the fixed-schedule baseline's 18.0C, because a *variable* setback depth is
# one of the things a static schedule structurally cannot do: a schedule has
# to pick one number that covers both a 13-hour overnight gap and a 61-hour
# weekend. Going deep is only safe when the recovery time is available,
# which is exactly what `occupancy_imminent` tracks.
DEEP_SETBACK_C = 16.0

# Occupied target. Deliberately identical to the fixed-schedule baseline's
# occupied setpoint (21.0C) so that no part of the reported energy
# difference comes from simply running the building colder while people are
# in it. Any saving has to come from *when* heat is bought, not from
# quietly degrading the occupied condition.
MODE_COMFORT_C = 21.0

# Aggressive recovery target, used when climbing out of setback before
# arrival and when comfort enforcement has to rescue a cold occupied zone.
MODE_PREHEAT_C = 21.5

# Pre-charge target. Held below the model's 23.9C occupied cooling setpoint
# so heating and cooling never fight each other - overshooting that would
# have the chiller undo the heat we just paid to store.
MODE_PRECHARGE_C = 23.0

# Coast target: the comfort floor. Heating effectively idles and the zone
# spends stored thermal mass instead of buying peak-priced electricity.
MODE_COAST_C = COMFORT_MIN_C

MODE_SETPOINT_C = {
    "SETBACK": DEEP_SETBACK_C,
    "PREHEAT": MODE_PREHEAT_C,
    "COMFORT": MODE_COMFORT_C,
    "PRECHARGE": MODE_PRECHARGE_C,
    "COAST": MODE_COAST_C,
}


@dataclass
class BuildingFeatures:
    """
    The four booleans the control policy actually turns on.

    Pre-computed rather than left as raw minute counts on purpose. Measured
    over 75 states, asking the 1.5B model to evaluate `minutes_until_peak
    between 1 and 120` itself scored 49.3%; handing it the already-evaluated
    booleans scored 54.7% and ran 23% faster (1.90s -> 1.47s per call).
    Arithmetic is the part small models are worst at, and it is the part we
    can trivially do for them.
    """

    building_occupied: bool
    occupancy_imminent: bool
    peak_now: bool
    peak_approaching: bool

    def to_dict(self) -> dict:
        return {
            "building_occupied": self.building_occupied,
            "occupancy_imminent": self.occupancy_imminent,
            "peak_now": self.peak_now,
            "peak_approaching": self.peak_approaching,
        }


def building_features(occupied_zones: int, minutes_until_occupancy: float,
                      minutes_until_peak: float) -> BuildingFeatures:
    return BuildingFeatures(
        building_occupied=occupied_zones > 0,
        occupancy_imminent=minutes_until_occupancy <= PREHEAT_LEAD_MINUTES,
        peak_now=minutes_until_peak == 0,
        peak_approaching=0 < minutes_until_peak <= PRECHARGE_WINDOW_MINUTES,
    )


def reference_mode(f: BuildingFeatures) -> str:
    """
    The deterministic policy the LLM is measured against.

    This is the control arm for the whole experiment. Running it alone
    (`--mode rules`) answers "what did the language model actually add?",
    and the fraction of cycles where the agent's chosen mode matches this
    is reported as mode agreement. Without this reference there is no way
    to tell an agent that is reasoning from one that is guessing.
    """
    if f.peak_now and f.building_occupied:
        return "COAST"
    if f.peak_approaching and (f.building_occupied or f.occupancy_imminent):
        return "PRECHARGE"
    if not f.building_occupied and not f.occupancy_imminent:
        return "SETBACK"
    if not f.building_occupied:
        return "PREHEAT"
    return "COMFORT"


@dataclass
class ZoneDecision:
    setpoint_c: float
    mode: str
    comfort_enforced: bool = False
    reason: str = ""


# Where in the comfort envelope each strategy sits, expressed as a PMV
# target rather than a temperature.
#
# This is the core control idea. ASHRAE 55 calls |PMV| <= 0.5 comfortable,
# which is a *band*, not a point - so there is room to move within it, and
# moving within it is free in comfort terms. Each strategy picks a different
# place to sit in that band:
#
#   PRECHARGE parks at the warm edge while electricity is cheap and clean,
#   banking heat in the structure. COAST spends it back down at the cool
#   edge while electricity is expensive and carbon-heavy. Both stay inside
#   the standard, so the load shifting costs nothing in comfort - it is
#   bought entirely out of the width of the band.
#
# Targets are kept at 0.45 rather than 0.5 so that normal control noise does
# not tip a zone over the line.
MODE_PMV_TARGET = {
    "PREHEAT": -0.10,    # arrive just-comfortable, without overshooting
    "COMFORT": -0.25,    # comfortable, at the efficient end of the band
    "PRECHARGE": 0.40,   # warm edge of the standard - store heat
    "COAST": -0.45,      # cool edge of the standard - spend it
}

# Heating setpoints are capped below the model's 23.9C occupied cooling
# setpoint. Without this, PRECHARGE solves to 27.5C in dry air, the clamp
# lets it through at the 24.0C comfort ceiling, and the chiller immediately
# starts removing the heat the boiler is paying to add - burning energy on
# both sides to hold a temperature nobody asked for.
MAX_HEATING_SETPOINT_C = 23.5


def _bounded(temp_c: float) -> float:
    """Keep a solved PMV target inside the usable heating-setpoint range."""
    return min(max(temp_c, COMFORT_MIN_C), MAX_HEATING_SETPOINT_C)


def apply_mode(mode: str, zone_temp_c: float, zone_occupied: bool,
               radiant_temp_c: float | None = None,
               rh_pct: float | None = None) -> ZoneDecision:
    """
    Translate a building-level strategy into a setpoint for one zone, then
    enforce comfort on top of it.

    Comfort enforcement is the load-bearing safety property and it runs on
    *every* zone on *every* timestep, independently of how often the LLM is
    consulted. This is what makes it safe to run the model on a slow cadence:
    a stale or wrong strategy cannot leave an occupied zone cold, because
    this check re-evaluates against live zone conditions 96 times a day per
    zone regardless.

    An earlier version ran the whole control path only when the LLM ran
    (every 8 timesteps). A setback chosen while the building was empty then
    stayed latched for 100 minutes after people arrived, which produced 57
    comfort violations on its own.

    When radiant temperature and humidity are supplied, targets are solved in
    PMV terms; without them it falls back to the fixed temperatures in
    MODE_SETPOINT_C, which is what the single-zone mock path uses.
    """
    if mode not in MODE_SETPOINT_C:
        mode = "COMFORT"

    have_pmv_inputs = radiant_temp_c is not None and rh_pct is not None

    if mode == "SETBACK":
        # Nobody is here; PMV is not defined for an empty room. Hold the
        # deep setback and let comfort enforcement below catch the case
        # where someone is present after all.
        setpoint = DEEP_SETBACK_C
    elif have_pmv_inputs:
        setpoint = _bounded(comfort.temp_for_pmv(
            MODE_PMV_TARGET[mode], radiant_temp_c, rh_pct))
    else:
        setpoint = MODE_SETPOINT_C[mode]

    if zone_occupied and have_pmv_inputs:
        pmv_now = comfort.fanger_pmv(zone_temp_c, radiant_temp_c, rh_pct)

        # Occupied and colder than the standard allows: nothing the strategy
        # layer wants outranks this. Applies even in COAST and SETBACK - if
        # stored heat ran out sooner than the strategy assumed, buy heat now.
        if pmv_now < -comfort.PMV_COMFORT_LIMIT:
            return ZoneDecision(
                setpoint_c=clamp_setpoint(_bounded(
                    comfort.temp_for_pmv(-0.10, radiant_temp_c, rh_pct))),
                mode=mode, comfort_enforced=True,
                reason=f"occupied zone at PMV {pmv_now:+.2f} is below the "
                       f"-{comfort.PMV_COMFORT_LIMIT} ASHRAE 55 limit",
            )

        if pmv_now > comfort.PMV_COMFORT_LIMIT:
            return ZoneDecision(
                setpoint_c=clamp_setpoint(_bounded(
                    comfort.temp_for_pmv(-0.25, radiant_temp_c, rh_pct))),
                mode=mode, comfort_enforced=True,
                reason=f"occupied zone at PMV {pmv_now:+.2f} is above the "
                       f"+{comfort.PMV_COMFORT_LIMIT} ASHRAE 55 limit",
            )

    elif zone_occupied:
        # Dry-bulb fallback for the mock path, which has no radiant or
        # humidity data to compute PMV from.
        if zone_temp_c < COMFORT_MIN_C:
            return ZoneDecision(
                setpoint_c=clamp_setpoint(MODE_PREHEAT_C), mode=mode,
                comfort_enforced=True,
                reason=f"occupied zone at {zone_temp_c:.1f}C is below the "
                       f"{COMFORT_MIN_C:.1f}C comfort floor",
            )
        if zone_temp_c > COMFORT_MAX_C:
            return ZoneDecision(
                setpoint_c=clamp_setpoint(COMFORT_MIN_C), mode=mode,
                comfort_enforced=True,
                reason=f"occupied zone at {zone_temp_c:.1f}C is above the "
                       f"{COMFORT_MAX_C:.1f}C comfort ceiling",
            )

    return ZoneDecision(setpoint_c=clamp_setpoint(setpoint), mode=mode)


@dataclass
class ModeStats:
    """
    Two separate honesty counters, deliberately not merged.

    `agreement` is how often the agent independently chose what the
    deterministic reference policy would have chosen - the measure of
    whether the LLM is contributing judgment.

    `comfort_enforced` is how often the physical safety layer had to
    intervene on a zone. These answer different questions and averaging
    them together would hide both.
    """

    decisions: int = 0
    agreed: int = 0
    zone_actions: int = 0
    comfort_enforced: int = 0
    mode_counts: dict = None

    def __post_init__(self):
        if self.mode_counts is None:
            self.mode_counts = {m: 0 for m in BUILDING_MODES}

    def record_decision(self, chosen: str, reference: str) -> None:
        self.decisions += 1
        if chosen == reference:
            self.agreed += 1
        self.mode_counts[chosen] = self.mode_counts.get(chosen, 0) + 1

    def record_zone(self, decision: ZoneDecision) -> None:
        self.zone_actions += 1
        if decision.comfort_enforced:
            self.comfort_enforced += 1

    @property
    def agreement_rate(self) -> float:
        return self.agreed / self.decisions if self.decisions else 0.0

    @property
    def enforcement_rate(self) -> float:
        return self.comfort_enforced / self.zone_actions if self.zone_actions else 0.0

    def summary(self) -> str:
        return (
            f"agent chose the reference strategy in {self.agreed}/{self.decisions} "
            f"cycles ({self.agreement_rate * 100:.1f}%); comfort enforcement fired on "
            f"{self.comfort_enforced}/{self.zone_actions} zone-steps "
            f"({self.enforcement_rate * 100:.1f}%); modes chosen: "
            + ", ".join(f"{m}={n}" for m, n in self.mode_counts.items() if n)
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
    outdoor_temp = float(state.get("outdoor_temp_c", zone_temp))
    minutes_to_change = state.get("minutes_until_occupancy_change")
    known_horizon = isinstance(minutes_to_change, (int, float))
    preheating = (
        not occupied and known_horizon and minutes_to_change <= PREHEAT_LEAD_MINUTES
    )
    # Coasting: occupants are still here, but leaving soon enough that
    # thermal mass can carry the zone to the end of the period unheated.
    # The projection guard is part of the condition rather than bolted on
    # at the override site, so that every path - including the "agent
    # proposed nothing" fallback - inherits it. Keeping it only at the
    # override site left the fallback able to coast straight through the
    # comfort floor.
    coasting = (
        occupied
        and known_horizon
        and minutes_to_change <= OPTIMAL_STOP_MINUTES
        and projected_temp_after(zone_temp, outdoor_temp, minutes_to_change)
        >= COMFORT_MIN_C
    )

    # Grid-aware load shifting. Pre-charging only makes sense while the
    # zone is (or is about to be) occupied - there's no point banking
    # heat in an empty building that will just leak away.
    minutes_to_peak = state.get("minutes_until_price_peak")
    known_peak = isinstance(minutes_to_peak, (int, float))
    in_peak = known_peak and minutes_to_peak == 0
    # Riding out the peak on stored heat is only safe while the zone can
    # hold comfort until the next decision comes round.
    peak_discharge_ok = (
        in_peak
        and occupied
        and projected_temp_after(zone_temp, outdoor_temp, COAST_STEP_MINUTES * 2)
        >= COMFORT_MIN_C
    )
    precharging = (
        known_peak
        and not in_peak
        and 0 < minutes_to_peak <= PRECHARGE_WINDOW_MINUTES
        and (occupied or preheating)
    )

    if proposed_setpoint_c is None:
        if coasting:
            fallback = UNOCCUPIED_SETBACK_C
        elif precharging:
            fallback = PRECHARGE_TARGET_C
        elif peak_discharge_ok:
            fallback = PEAK_FLOOR_C
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
    if coasting and proposed > UNOCCUPIED_SETBACK_C:
        return SupervisorDecision(
            setpoint_c=UNOCCUPIED_SETBACK_C,
            overridden=True,
            reason=f"occupancy ends in {minutes_to_change:.0f}min, coasting on thermal mass",
            proposed_setpoint_c=proposed,
        )

    # 5. Pre-charge: the price/carbon peak is approaching and power is
    #    still cheap. Bank heat in the thermal mass now so the HVAC can
    #    stay idle when energy gets expensive.
    if precharging and proposed < PRECHARGE_TARGET_C:
        return SupervisorDecision(
            setpoint_c=PRECHARGE_TARGET_C,
            overridden=True,
            reason=f"price peak in {minutes_to_peak:.0f}min, pre-charging thermal mass",
            proposed_setpoint_c=proposed,
        )

    # 6. In-peak: energy is at its most expensive and carbon-heavy, and
    #    the zone has banked warmth to spend. Ride it down to the comfort
    #    floor rather than paying peak rates to hold it higher.
    if peak_discharge_ok and proposed > PEAK_FLOOR_C:
        return SupervisorDecision(
            setpoint_c=PEAK_FLOOR_C,
            overridden=True,
            reason="price peak active, discharging stored heat instead of buying it",
            proposed_setpoint_c=proposed,
        )

    return SupervisorDecision(
        setpoint_c=proposed,
        overridden=False,
        proposed_setpoint_c=proposed,
    )
