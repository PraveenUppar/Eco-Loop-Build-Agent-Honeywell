"""
Orchestrates the closed-loop run.

Mock modes (fast iteration, no EnergyPlus needed):
  mock       - rule-based fixed schedule driving the mock thermal model.
  mock-rules - supervisor rules alone, no LLM. The mock control arm.
  mock-ai    - LLM agent driving the mock thermal model.

Real EnergyPlus modes (PyEnergyPlus API, all five conditioned zones):
  discover   - list the zones an IDF exposes and whether each is controllable.
  native     - observe only. The building runs on its own thermostat
               schedules, untouched. This is the true as-designed reference.
  baseline   - fixed 21C occupied / 18C setback schedule, written every
               timestep. A conventional BMS schedule.
  rules      - the deterministic strategy policy with no LLM in the loop.
               The control arm: whatever `ai` achieves beyond this is what
               the language model actually contributed.
  ai         - the LLM picks a building strategy; the supervisor turns it
               into per-zone setpoints and enforces comfort every timestep.

The real modes write two CSVs per run:
  logs/<mode>.csv        one row per timestep, building-level energy/cost/CO2
  logs/<mode>_zones.csv  one row per timestep per zone, temperature/PMV
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import comfort
import grid
from tools import (COMFORT_MAX_C, COMFORT_MIN_C, LiveStateExecutor, MockBuilding,
                   ToolExecutor)

LOG_COLUMNS = [
    "step", "day", "minute_of_day", "zone_temp_c", "outdoor_temp_c",
    "setpoint_c", "occupied", "pmv", "energy_kwh_step", "cost_step",
    "carbon_g_step", "latency_s",
    "agent_ran", "agent_proposed_c", "overridden", "override_reason", "reasoning",
]

BUILDING_LOG_COLUMNS = [
    "step", "day", "day_of_week", "minute_of_day", "outdoor_temp_c",
    "site_kwh_step", "hvac_kwh_step", "elec_kwh_step", "gas_kwh_step",
    "cost_step", "carbon_g_step", "price_per_kwh", "is_peak",
    "occupied_zones", "mean_zone_temp_c", "min_zone_temp_c",
    "worst_pmv", "zones_pmv_violation", "zones_band_violation",
    "mode", "reference_mode", "mode_agrees", "agent_ran", "latency_s",
    "state_injected", "comfort_enforced_zones", "reason",
]

ZONE_LOG_COLUMNS = [
    "step", "day", "minute_of_day", "zone", "zone_temp_c", "radiant_temp_c",
    "rh_pct", "setpoint_c", "occupied", "pmv", "ppd", "comfort_enforced",
]


def rule_based_setpoint(occupied: bool) -> float:
    """Fixed schedule baseline: 21C occupied, 18C setback unoccupied."""
    return 21.0 if occupied else 18.0


def open_log(mode: str, columns=None, suffix: str = ""):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"{mode}{suffix}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=columns or LOG_COLUMNS)
    writer.writeheader()
    return f, writer, path


def run_mock(hours: int, step_minutes: int) -> str:
    building = MockBuilding(step_minutes=step_minutes)
    steps = int(hours * 60 / step_minutes)

    f, writer, path = open_log("mock")
    try:
        for i in range(steps):
            setpoint = rule_based_setpoint(building._occupied(building.state.minute_of_day))
            s = building.step(setpoint)
            writer.writerow({
                "step": i, "day": s.day, "minute_of_day": s.minute_of_day,
                "zone_temp_c": s.zone_temp_c, "outdoor_temp_c": s.outdoor_temp_c,
                "setpoint_c": s.setpoint_c, "occupied": s.occupied, "pmv": s.pmv,
                "energy_kwh_step": s.energy_kwh_step, "cost_step": s.cost_step,
                "carbon_g_step": s.carbon_g_step, "latency_s": 0.0, "reasoning": "",
            })
    finally:
        f.close()
    print(f"[mock] {steps} steps logged to {path}")
    return path


def run_mock_rules(hours: int, step_minutes: int) -> str:
    """
    Supervisor rules alone, no LLM in the loop.

    This is the experimental control arm: it answers "how much does the
    language model actually add over the deterministic rules it's paired
    with?" Comparing mock-ai against this - rather than only against the
    fixed-schedule baseline - is what keeps the LLM's contribution
    honest, since `supervise(None, ...)` takes exactly the same rule
    branches the supervisor would use to override the agent.
    """
    from supervisor import supervise

    building = MockBuilding(step_minutes=step_minutes)
    executor = ToolExecutor(building)
    steps = int(hours * 60 / step_minutes)

    f, writer, path = open_log("mock-rules")
    try:
        for i in range(steps):
            setpoint = supervise(None, executor.get_zone_state()).setpoint_c
            s = building.step(setpoint)
            writer.writerow({
                "step": i, "day": s.day, "minute_of_day": s.minute_of_day,
                "zone_temp_c": s.zone_temp_c, "outdoor_temp_c": s.outdoor_temp_c,
                "setpoint_c": s.setpoint_c, "occupied": s.occupied, "pmv": s.pmv,
                "energy_kwh_step": s.energy_kwh_step, "cost_step": s.cost_step,
                "carbon_g_step": s.carbon_g_step, "latency_s": 0.0,
                "agent_ran": False, "agent_proposed_c": "", "overridden": False,
                "override_reason": "", "reasoning": "",
            })
    finally:
        f.close()
    print(f"[mock-rules] {steps} steps logged to {path}")
    return path


def run_mock_ai(hours: int, step_minutes: int, model: str, agent_every_n_steps: int,
                use_supervisor: bool = True) -> str:
    from llm_agent import HVACAgent  # deferred import: needs ollama installed
    from supervisor import OverrideStats, supervise

    building = MockBuilding(step_minutes=step_minutes)
    executor = ToolExecutor(building)
    agent = HVACAgent(executor, model=model)
    steps = int(hours * 60 / step_minutes)
    stats = OverrideStats()

    f, writer, path = open_log("mock-ai")
    current_setpoint = building.state.setpoint_c
    reasoning = ""
    latency = 0.0
    overridden = False
    override_reason = ""
    agent_proposed = current_setpoint
    try:
        for i in range(steps):
            agent_ran = i % agent_every_n_steps == 0
            if agent_ran:
                result = agent.run_step()
                agent_proposed = result.final_setpoint_c
                reasoning = result.reasoning
                latency = result.latency_s

                if use_supervisor:
                    decision = supervise(agent_proposed, executor.get_zone_state())
                    stats.record(decision)
                    current_setpoint = decision.setpoint_c
                    overridden = decision.overridden
                    override_reason = decision.reason
                else:
                    current_setpoint = agent_proposed
                    overridden, override_reason = False, ""

                flag = f" [OVERRIDE: {override_reason}]" if overridden else ""
                print(f"  step {i}: agent {agent_proposed}C -> applied "
                      f"{current_setpoint}C ({latency}s){flag}")

            s = building.step(current_setpoint)
            writer.writerow({
                "step": i, "day": s.day, "minute_of_day": s.minute_of_day,
                "zone_temp_c": s.zone_temp_c, "outdoor_temp_c": s.outdoor_temp_c,
                "setpoint_c": s.setpoint_c, "occupied": s.occupied, "pmv": s.pmv,
                "energy_kwh_step": s.energy_kwh_step, "cost_step": s.cost_step,
                "carbon_g_step": s.carbon_g_step, "latency_s": latency,
                "agent_ran": agent_ran, "agent_proposed_c": agent_proposed,
                "overridden": overridden, "override_reason": override_reason,
                "reasoning": reasoning,
            })
    finally:
        f.close()
    print(f"[mock-ai] {steps} steps logged to {path}")
    if use_supervisor:
        print(f"[mock-ai] supervisor: {stats.summary()}")
    return path


# The 5ZoneAirCooled model's OCCUPY-1 schedule: weekdays 08:00-19:00,
# weekends and holidays empty all day. Verified against the IDF, not
# assumed. Mirrored here so the controller can anticipate occupancy, which
# a bare "is anyone here right now" reading cannot support - preheating
# requires knowing when people will arrive.
OCC_START_MIN, OCC_END_MIN = 8 * 60, 19 * 60


def _is_weekend(day_of_week: int) -> bool:
    """EnergyPlus day_of_week is 1=Sunday .. 7=Saturday."""
    return day_of_week in (1, 7)


def _minutes_to_occupancy_change(minute_of_day: int, day_of_week: int) -> int:
    """
    Minutes until the zone next becomes occupied (or empties).

    Weekend-aware: on a Friday evening the next occupancy is Monday
    morning, not Saturday. Treating every day as a workday made the
    controller preheat an empty building through both weekend mornings.
    """
    day_min = 24 * 60
    if not _is_weekend(day_of_week) and OCC_START_MIN <= minute_of_day < OCC_END_MIN:
        return OCC_END_MIN - minute_of_day

    # Not occupied - find the next weekday 08:00.
    if not _is_weekend(day_of_week) and minute_of_day < OCC_START_MIN:
        return OCC_START_MIN - minute_of_day

    minutes = day_min - minute_of_day  # to midnight tonight
    d = day_of_week
    for _ in range(7):
        d = 1 if d == 7 else d + 1
        if not _is_weekend(d):
            return minutes + OCC_START_MIN
        minutes += day_min
    return minutes + OCC_START_MIN


def _import_energyplus():
    try:
        from pyenergyplus.api import EnergyPlusAPI
        return EnergyPlusAPI
    except ImportError:
        print(
            "pyenergyplus not importable.\n\n"
            "Install EnergyPlus from https://energyplus.net/downloads, then point\n"
            "PYTHONPATH at the install directory so its Python package is visible:\n\n"
            "  Windows (PowerShell):\n"
            '    $env:PYTHONPATH = "C:\\EnergyPlusV26-1-0"\n'
            "  Linux/macOS:\n"
            "    export PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-26-1-0\n\n"
            "Then re-run. Use --mode discover first to find your zone names.",
            file=sys.stderr,
        )
        sys.exit(1)


# EnergyPlus Constant::KindOfSim. The weather-file run period is 3;
# 1/2/4/5 are design-day and sizing runs.
#
# This gate is not cosmetic. Zone/system/plant sizing calculations run the
# design days internally even when SimulationControl says "Run Simulation
# for Sizing Periods: No", and the timestep callbacks fire during them. An
# earlier version logged those rows: 576 of 1248 rows (46%) came from
# sizing, all stamped day_of_year=1 with sub-timestep timestamps, and they
# were being summed into the energy totals. Every number produced before
# this filter existed was measured partly against design-day weather.
KIND_OF_SIM_RUN_PERIOD_WEATHER = 3

# Site energy meters, in Joules per timestep. Total site energy is
# electricity plus gas.
#
# The earlier code metered `Heating:EnergyTransfer`, which is heat delivered
# into the zones - not energy bought. With a boiler at 0.8 efficiency those
# differ by 25% before any distribution losses, and if the plant had been a
# heat pump they would have differed by a factor of three in the other
# direction. "Percentage reduction in total kWh consumed" has to be measured
# at the meter.
# `Electricity:Facility` is NOT reachable through the API in EnergyPlus 26.1
# even though it appears in the .mtd meter dictionary - get_meter_handle
# returns -1 for it while end-use meters resolve normally. Nothing errors and
# api_error_flag stays false; the meter just silently reads as zero, which
# reported a whole week of the building's electricity as 0.0 kWh. The
# purchased-electricity meter is the one that actually resolves, and it is
# the more correct choice anyway: it is metered at the point of purchase,
# which is what a kWh-reduction claim is about.
SITE_METERS = {"elec": "ElectricityPurchased:Facility", "gas": "NaturalGas:Facility"}

# Used if the primary purchased-electricity meter is unavailable. These three
# partition site electricity by scope and are always present.
SITE_ELEC_FALLBACK = ["Electricity:Building", "Electricity:HVAC", "Electricity:Plant"]

# HVAC-only subtotal, reported alongside site energy. Lights and plug loads
# are a large fixed load this controller cannot touch, so quoting only
# whole-building percentages would understate the control result, while
# quoting only HVAC would overstate the building result. Both are reported.
HVAC_METERS = [
    "Heating:Electricity", "Heating:NaturalGas", "Cooling:Electricity",
    "Fans:Electricity", "Pumps:Electricity", "HeatRejection:Electricity",
]

JOULES_PER_KWH = 3_600_000.0


def _zones_from_idf(idf_path: str) -> list[str]:
    """
    Zone names that have a thermostat, read straight from the IDF.

    Needed because output variables must be requested *before* the
    simulation starts, while the API can only enumerate zones once it is
    already running - so the zone list cannot come from the simulation
    itself on the same run.
    """
    with open(idf_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    zones = []
    for block in re.findall(r"ZoneControl:Thermostat,.*?;", text, flags=re.S):
        fields = [ln.split("!")[0].strip().rstrip(",;")
                  for ln in block.splitlines()[1:] if ln.strip()]
        if len(fields) >= 2 and fields[1]:
            zones.append(fields[1])
    return zones


def discover_energyplus_handles(idf_path: str, epw_path: str) -> None:
    """
    Print the zones an IDF exposes and whether each is actually controllable.

    Zone names are not guessable and differ between example files, so rather
    than reading them out of the IDF by hand, run a short simulation and ask
    the API what exists.
    """
    EnergyPlusAPI = _import_energyplus()
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    reported = {"done": False}

    print(f"\nZones with a thermostat, per the IDF text: "
          f"{', '.join(_zones_from_idf(idf_path)) or '(none found)'}")

    def on_ready(s):
        if reported["done"] or not api.exchange.api_data_fully_ready(s):
            return
        reported["done"] = True

        # Probe each zone for the handle this controller actually needs. A
        # zone without a thermostat (plenums, unconditioned spaces) resolves
        # to -1 here, which is exactly the failure that otherwise shows up
        # much later as an empty log.
        print("\n=== Zones in this model ===")
        print(f"{'zone':<24} setpoint actuator")
        controllable = []
        for name in api.exchange.get_object_names(s, "Zone"):
            ok = api.exchange.get_actuator_handle(
                s, "Zone Temperature Control", "Heating Setpoint", name) != -1
            if ok:
                controllable.append(name)
            print(f"{name:<24} {'yes' if ok else 'NO (no thermostat)'}")

        if controllable:
            print(f"\nControllable zones: {', '.join(controllable)}")
            print('Run with:  --zones all      (or --zones "%s")\n'
                  % ",".join(controllable))
        else:
            print("\nNo zone in this model exposes a heating setpoint actuator.\n"
                  "Pick an IDF with thermostats (e.g. 5ZoneAirCooled.idf).\n")
        api.runtime.stop_simulation(s)

    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(state, on_ready)
    # -D runs design days only - enough to enumerate objects without
    # simulating a full year.
    api.runtime.run_energyplus(
        state, ["-w", epw_path, "-d", "out/discover", "-D", idf_path]
    )


def run_real_energyplus(mode: str, idf_path: str, epw_path: str, model: str,
                        zones: list[str], agent_every_n_steps: int = 4) -> str:
    """
    Real EnergyPlus integration via the PyEnergyPlus API, all zones.

    Four EnergyPlus-specific details this gets right, each of which is a way
    the integration otherwise produces plausible-looking but wrong numbers:

    1. Output variables must be requested BEFORE the run starts, or the
       handle lookup returns -1 at runtime.
    2. Handles cannot be resolved until `api_data_fully_ready()` is true,
       so resolution happens lazily inside the callback rather than up front.
    3. Sizing and warmup timesteps must be excluded - see
       KIND_OF_SIM_RUN_PERIOD_WEATHER above.
    4. Control and measurement happen in *different* callbacks. The setpoint
       is written at the start of a zone timestep, before HVAC is simulated,
       so it governs the timestep about to run; the meters are read at the
       end of that same timestep, so the energy logged is the energy that
       setpoint actually caused. Doing both at the start reports each
       timestep's energy against the previous timestep's setpoint.
    """
    EnergyPlusAPI = _import_energyplus()
    from supervisor import ModeStats, apply_mode, building_features, reference_mode

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    stats = ModeStats()

    agent, agent_tools = None, None
    if mode == "ai":
        from llm_agent import StrategyAgent
        from tools import BuildingToolExecutor
        agent_tools = BuildingToolExecutor()
        agent = StrategyAgent(agent_tools, model=model)

    # Must be requested before run_energyplus(), not inside the callback.
    api.exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature",
                                  "Environment")
    for z in zones:
        api.exchange.request_variable(state, "Zone Mean Air Temperature", z)
        # Radiant temperature and humidity are needed for real Fanger PMV.
        api.exchange.request_variable(state, "Zone Mean Radiant Temperature", z)
        api.exchange.request_variable(state, "Zone Air Relative Humidity", z)
        # Read occupancy from the model instead of assuming a clock schedule:
        # this building is empty all weekend, and a clock assumption both
        # heats it on Saturday and counts comfort violations there, where
        # nobody is present to be uncomfortable.
        api.exchange.request_variable(state, "Zone People Occupant Count", z)

    bf, bwriter, bpath = open_log(mode, BUILDING_LOG_COLUMNS)
    zf, zwriter, zpath = open_log(mode, ZONE_LOG_COLUMNS, suffix="_zones")

    h = {"outdoor": None, "zones": {}, "site": {}, "hvac": {}, "resolved": False}
    counters = {"steps": 0, "skipped": 0, "agent_calls": 0}
    held = {"mode": None, "reason": "", "latency": 0.0, "state_injected": False}
    pending = {}

    def usable(s) -> bool:
        """True only for real, post-warmup weather-run-period timesteps."""
        return (api.exchange.api_data_fully_ready(s)
                and api.exchange.kind_of_sim(s) == KIND_OF_SIM_RUN_PERIOD_WEATHER
                and not api.exchange.warmup_flag(s))

    def resolve(s) -> bool:
        h["outdoor"] = api.exchange.get_variable_handle(
            s, "Site Outdoor Air Drybulb Temperature", "Environment")
        for z in zones:
            h["zones"][z] = {
                "temp": api.exchange.get_variable_handle(s, "Zone Mean Air Temperature", z),
                "mrt": api.exchange.get_variable_handle(s, "Zone Mean Radiant Temperature", z),
                "rh": api.exchange.get_variable_handle(s, "Zone Air Relative Humidity", z),
                "people": api.exchange.get_variable_handle(s, "Zone People Occupant Count", z),
                "heat_sp": api.exchange.get_actuator_handle(
                    s, "Zone Temperature Control", "Heating Setpoint", z),
            }
        for key, name in SITE_METERS.items():
            h["site"][key] = api.exchange.get_meter_handle(s, name)
        if h["site"].get("elec", -1) == -1:
            h["elec_parts"] = [hd for hd in
                               (api.exchange.get_meter_handle(s, n)
                                for n in SITE_ELEC_FALLBACK) if hd != -1]
            print(f"[{mode}] {SITE_METERS['elec']} unavailable; summing "
                  f"{len(h['elec_parts'])} scope meters instead", file=sys.stderr)
        for name in HVAC_METERS:
            handle = api.exchange.get_meter_handle(s, name)
            if handle != -1:
                h["hvac"][name] = handle

        bad = []
        if h["outdoor"] == -1:
            bad.append("outdoor temp")
        # A site meter that silently reads zero is worse than a crash: it
        # produces a complete, plausible log whose energy column is all
        # zeroes. Fail loudly instead.
        if h["site"].get("elec", -1) == -1 and not h.get("elec_parts"):
            bad.append("site electricity meter")
        if not h["hvac"]:
            bad.append("all HVAC meters")
        for z, hh in h["zones"].items():
            bad += [f"{z}.{k}" for k, v in hh.items() if v == -1]
        if bad:
            print(f"\n[{mode}] could not resolve handles: {', '.join(bad)}\n"
                  f"       zones requested: {zones}\n"
                  f"       run `--mode discover` to list valid zone names\n",
                  file=sys.stderr)
            return False
        print(f"[{mode}] handles resolved for {len(zones)} zones "
              f"({', '.join(zones)}); {len(h['hvac'])} HVAC meters; controlling...")
        return True

    def on_begin_timestep(s):
        """Read state, choose a strategy, write setpoints for this timestep."""
        if not usable(s):
            counters["skipped"] += 1
            return
        if not h["resolved"]:
            if not resolve(s):
                api.runtime.stop_simulation(s)
                return
            h["resolved"] = True

        minute_of_day = int(round(api.exchange.current_time(s) * 60)) % (24 * 60)
        day_of_week = api.exchange.day_of_week(s)
        outdoor = api.exchange.get_variable_value(s, h["outdoor"])

        zone_state = {}
        for z, hh in h["zones"].items():
            temp = api.exchange.get_variable_value(s, hh["temp"])
            mrt = api.exchange.get_variable_value(s, hh["mrt"])
            rh = api.exchange.get_variable_value(s, hh["rh"])
            occ = api.exchange.get_variable_value(s, hh["people"]) > 0
            zone_state[z] = {"temp": temp, "mrt": mrt, "rh": rh, "occupied": occ}

        occupied_zones = sum(1 for v in zone_state.values() if v["occupied"])
        mins_to_occ = _minutes_to_occupancy_change(minute_of_day, day_of_week)
        mins_to_peak = grid.minutes_until_peak(minute_of_day)
        features = building_features(occupied_zones, mins_to_occ, mins_to_peak)
        ref_mode = reference_mode(features)

        agent_ran = False
        if mode == "ai":
            # The LLM sets strategy on a slow cadence; the supervisor below
            # re-derives per-zone setpoints from live state every timestep.
            # That split is what makes a slow cadence safe - see
            # supervisor.apply_mode().
            agent_ran = counters["steps"] % agent_every_n_steps == 0
            if agent_ran:
                agent_tools.update({
                    **features.to_dict(),
                    "occupied_zones": occupied_zones,
                    "total_zones": len(zones),
                    "min_zone_temp_c": round(min(v["temp"] for v in zone_state.values()), 2),
                    "mean_zone_temp_c": round(
                        sum(v["temp"] for v in zone_state.values()) / len(zone_state), 2),
                    "outdoor_temp_c": round(outdoor, 2),
                    "comfort_band_c": [COMFORT_MIN_C, COMFORT_MAX_C],
                    "minute_of_day": minute_of_day,
                })
                decision = agent.decide()
                counters["agent_calls"] += 1
                # A None mode means the model returned something outside the
                # allowed set despite the grammar. Falling back to the
                # reference policy keeps the building safe, and the
                # substitution is counted rather than hidden.
                chosen = decision.mode or ref_mode
                held.update(mode=chosen, reason=decision.reason,
                            latency=decision.latency_s,
                            state_injected=decision.state_injected)
                stats.record_decision(chosen, ref_mode)
            active_mode = held["mode"] or ref_mode
        elif mode == "rules":
            active_mode = ref_mode
        else:
            active_mode = ref_mode  # recorded for comparison only

        enforced = 0
        setpoints = {}
        for z, v in zone_state.items():
            if mode == "native":
                # Observe only - never write the actuator, so the model runs
                # on its own thermostat schedules.
                setpoints[z] = (float("nan"), False)
                continue
            if mode == "baseline":
                sp, was_enforced = rule_based_setpoint(v["occupied"]), False
            else:
                d = apply_mode(active_mode, v["temp"], v["occupied"],
                               radiant_temp_c=v["mrt"], rh_pct=v["rh"])
                stats.record_zone(d)
                sp, was_enforced = d.setpoint_c, d.comfort_enforced
            if was_enforced:
                enforced += 1
            api.exchange.set_actuator_value(s, h["zones"][z]["heat_sp"], sp)
            setpoints[z] = (sp, was_enforced)

        pending.clear()
        pending.update(
            minute_of_day=minute_of_day, day_of_week=day_of_week, outdoor=outdoor,
            zone_state=zone_state, setpoints=setpoints, occupied_zones=occupied_zones,
            active_mode=active_mode, ref_mode=ref_mode, agent_ran=agent_ran,
            enforced=enforced, day=api.exchange.day_of_year(s),
        )

    def on_end_timestep(s):
        """Read the meters for the timestep just simulated and log it."""
        if not usable(s) or not pending or not h["resolved"]:
            return

        if h["site"].get("elec", -1) != -1:
            elec_j = max(0.0, api.exchange.get_meter_value(s, h["site"]["elec"]))
        else:
            elec_j = sum(max(0.0, api.exchange.get_meter_value(s, hd))
                         for hd in h.get("elec_parts", []))
        gas_j = max(0.0, api.exchange.get_meter_value(s, h["site"]["gas"])) \
            if h["site"].get("gas", -1) != -1 else 0.0
        hvac_j = sum(max(0.0, api.exchange.get_meter_value(s, hd))
                     for hd in h["hvac"].values())

        elec_kwh = elec_j / JOULES_PER_KWH
        gas_kwh = gas_j / JOULES_PER_KWH
        hvac_kwh = hvac_j / JOULES_PER_KWH
        minute = pending["minute_of_day"]

        # Electricity carries the time-of-use tariff and the varying carbon
        # intensity; gas is flat on both. Blending them into one average
        # would erase exactly the signal the controller is optimising against.
        cost = elec_kwh * grid.price_per_kwh(minute) + gas_kwh * grid.GAS_PRICE_PER_KWH
        carbon = (elec_kwh * grid.carbon_intensity(minute)
                  + gas_kwh * grid.GAS_CARBON_G_PER_KWH)

        step = counters["steps"]
        pmvs, band_viol, pmv_viol = [], 0, 0
        for z, v in pending["zone_state"].items():
            sp, enforced = pending["setpoints"][z]
            pmv = comfort.fanger_pmv(v["temp"], v["mrt"], v["rh"]) if v["occupied"] else 0.0
            if v["occupied"]:
                pmvs.append(pmv)
                if not comfort.is_comfortable(pmv):
                    pmv_viol += 1
                if not (COMFORT_MIN_C <= v["temp"] <= COMFORT_MAX_C):
                    band_viol += 1
            zwriter.writerow({
                "step": step, "day": pending["day"], "minute_of_day": minute,
                "zone": z, "zone_temp_c": round(v["temp"], 3),
                "radiant_temp_c": round(v["mrt"], 3), "rh_pct": round(v["rh"], 2),
                "setpoint_c": sp, "occupied": v["occupied"],
                "pmv": round(pmv, 3), "ppd": round(comfort.ppd(pmv), 2),
                "comfort_enforced": enforced,
            })

        temps = [v["temp"] for v in pending["zone_state"].values()]
        bwriter.writerow({
            "step": step, "day": pending["day"], "day_of_week": pending["day_of_week"],
            "minute_of_day": minute, "outdoor_temp_c": round(pending["outdoor"], 2),
            "site_kwh_step": round(elec_kwh + gas_kwh, 5),
            "hvac_kwh_step": round(hvac_kwh, 5),
            "elec_kwh_step": round(elec_kwh, 5), "gas_kwh_step": round(gas_kwh, 5),
            "cost_step": round(cost, 5), "carbon_g_step": round(carbon, 2),
            "price_per_kwh": grid.price_per_kwh(minute), "is_peak": grid.is_peak(minute),
            "occupied_zones": pending["occupied_zones"],
            "mean_zone_temp_c": round(sum(temps) / len(temps), 3),
            "min_zone_temp_c": round(min(temps), 3),
            "worst_pmv": round(max(pmvs, key=abs), 3) if pmvs else 0.0,
            "zones_pmv_violation": pmv_viol, "zones_band_violation": band_viol,
            "mode": pending["active_mode"] if mode not in ("native", "baseline") else mode,
            "reference_mode": pending["ref_mode"],
            "mode_agrees": pending["active_mode"] == pending["ref_mode"],
            "agent_ran": pending["agent_ran"], "latency_s": held["latency"],
            "state_injected": held["state_injected"],
            "comfort_enforced_zones": pending["enforced"],
            "reason": held["reason"] if mode == "ai" else "",
        })
        counters["steps"] += 1
        pending.clear()

    def on_progress(_pct):
        pass

    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, on_begin_timestep)
    api.runtime.callback_end_zone_timestep_after_zone_reporting(state, on_end_timestep)

    print(f"[{mode}] running EnergyPlus: {idf_path}")
    if mode == "ai":
        print(f"[{mode}] model={model}, one LLM decision every "
              f"{agent_every_n_steps} timesteps")
    try:
        api.runtime.run_energyplus(state, ["-w", epw_path, "-d", f"out/{mode}", idf_path])
    finally:
        bf.close()
        zf.close()

    if counters["steps"] == 0:
        print(f"[{mode}] WARNING: no timesteps were logged.\n"
              f"         Handles never resolved, or the run period never started.\n"
              f"         Check zone names with --mode discover.", file=sys.stderr)
    else:
        print(f"[{mode}] {counters['steps']} timesteps logged to {bpath}")
        print(f"[{mode}] per-zone detail in {zpath} "
              f"({counters['steps'] * len(zones)} zone-steps)")
        print(f"[{mode}] {counters['skipped']} sizing/warmup callbacks correctly skipped")
        if mode == "ai":
            print(f"[{mode}] {counters['agent_calls']} LLM decisions")
            print(f"[{mode}] {stats.summary()}")
        elif mode == "rules":
            print(f"[{mode}] {stats.summary()}")
    return bpath


def main():
    parser = argparse.ArgumentParser(description="Eco-Loop closed-loop control runner")
    parser.add_argument(
        "--mode", required=True,
        choices=["mock", "mock-rules", "mock-ai", "discover",
                 "native", "baseline", "rules", "ai"],
        help="mock/mock-rules/mock-ai use the lightweight simulator; "
             "discover lists the zone names in an IDF; "
             "native/baseline/rules/ai use real EnergyPlus "
             "(native = the model's own schedules untouched, "
             "rules = the strategy policy with no LLM, the control arm)",
    )
    parser.add_argument("--hours", type=float, default=24.0, help="mock horizon in hours")
    parser.add_argument("--step-minutes", type=int, default=15, help="mock step size")
    parser.add_argument("--agent-every-n-steps", type=int, default=4,
                        help="call the LLM every N timesteps; the supervisor still "
                             "re-evaluates comfort on every timestep in between")
    parser.add_argument("--model", default="qwen2.5:3b-instruct", help="Ollama model name")
    parser.add_argument("--no-supervisor", action="store_true",
                        help="mock-ai only: apply the agent's raw setpoints without the "
                             "deterministic override layer")
    parser.add_argument("--idf", help="path to .idf (required for discover/real modes)")
    parser.add_argument("--epw", help="path to .epw weather file (required for real modes)")
    parser.add_argument("--zones", default="all",
                        help='zones to control: "all" (every thermostat-controlled zone '
                             'in the IDF) or a comma-separated list')
    args = parser.parse_args()

    if args.mode == "mock":
        run_mock(args.hours, args.step_minutes)
    elif args.mode == "mock-rules":
        run_mock_rules(args.hours, args.step_minutes)
    elif args.mode == "mock-ai":
        run_mock_ai(args.hours, args.step_minutes, args.model, args.agent_every_n_steps,
                    use_supervisor=not args.no_supervisor)
    else:
        if not args.idf or not args.epw:
            parser.error(f"--idf and --epw are required for {args.mode} mode")
        if args.mode == "discover":
            discover_energyplus_handles(args.idf, args.epw)
        else:
            if args.zones == "all":
                zones = _zones_from_idf(args.idf)
                if not zones:
                    parser.error(f"no thermostat-controlled zones found in {args.idf}; "
                                 f"run --mode discover and pass --zones explicitly")
            else:
                zones = [z.strip() for z in args.zones.split(",") if z.strip()]
            run_real_energyplus(args.mode, args.idf, args.epw, args.model, zones,
                                agent_every_n_steps=args.agent_every_n_steps)


if __name__ == "__main__":
    main()
