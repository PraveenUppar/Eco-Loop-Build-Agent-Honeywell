"""
Orchestrates the closed-loop run. Four modes:

  mock      - rule-based fixed schedule driving the mock thermal model.
              This is the "baseline" for the mock track.
  mock-ai   - LLM agent driving the mock thermal model. Cheap/fast way to
              prove the tool-calling loop works before touching EnergyPlus.
  baseline  - rule-based fixed schedule driving *real* EnergyPlus via the
              PyEnergyPlus API. Requires EnergyPlus installed (see README).
  ai        - LLM agent driving real EnergyPlus, injecting setpoints back
              into the running simulation each step.

Every mode writes a CSV to logs/<mode>.csv with the same columns, so
dashboard/generate_report.py can compare any two runs directly.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import grid
from tools import COMFORT_MAX_C, COMFORT_MIN_C, MockBuilding, ToolExecutor

LOG_COLUMNS = [
    "step", "day", "minute_of_day", "zone_temp_c", "outdoor_temp_c",
    "setpoint_c", "occupied", "pmv", "energy_kwh_step", "cost_step",
    "carbon_g_step", "latency_s",
    "agent_ran", "agent_proposed_c", "overridden", "override_reason", "reasoning",
]


def rule_based_setpoint(occupied: bool) -> float:
    """Fixed schedule baseline: 21C occupied, 18C setback unoccupied."""
    return 21.0 if occupied else 18.0


def open_log(mode: str):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"{mode}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
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
            '    $env:PYTHONPATH = "C:\\EnergyPlusV24-1-0"\n'
            "  Linux/macOS:\n"
            "    export PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-24-1-0\n\n"
            "Then re-run. Use --mode discover first to find your zone names.",
            file=sys.stderr,
        )
        sys.exit(1)


def discover_energyplus_handles(idf_path: str, epw_path: str) -> None:
    """
    Print the zone names and setpoint actuators an IDF actually exposes.

    Zone names are not guessable and differ between example files, so
    rather than reading them out of the IDF by hand, run a short
    simulation and ask the API what exists. Writes
    `out/discover/eplusout.edd`, EnergyPlus's full actuator list, and
    prints the thermostat actuators that matter here.
    """
    EnergyPlusAPI = _import_energyplus()
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    reported = {"done": False}

    def on_ready(s):
        if reported["done"] or not api.exchange.api_data_fully_ready(s):
            return
        reported["done"] = True
        print("\n=== Zones in this model ===")
        for name in api.exchange.get_object_names(s, "Zone"):
            print(f"  {name}")
        print("\nFull actuator list written to out/discover/eplusout.edd")
        print("Look for lines containing 'Zone Temperature Control'. Use the")
        print("zone name from that file with --zone.\n")
        api.runtime.stop_simulation(s)

    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(state, on_ready)
    # -r writes the .edd actuator listing; -D runs design days only, which
    # is enough to enumerate objects without simulating a whole year.
    api.runtime.run_energyplus(
        state, ["-w", epw_path, "-d", "out/discover", "-r", "-D", idf_path]
    )


def run_real_energyplus(mode: str, idf_path: str, epw_path: str, model: str,
                        zone: str, use_supervisor: bool = True) -> str:
    """
    Real EnergyPlus integration via the PyEnergyPlus API.

    Same control logic as the mock modes - only the source of state and the
    destination of setpoints change. `baseline` drives the fixed schedule,
    `ai` drives the LLM through the same supervisor used in `mock-ai`.

    Two EnergyPlus-specific details this gets right, both of which are
    common ways this integration silently produces nothing:

    1. Output variables must be requested BEFORE the run starts, or the
       handle lookup returns -1 at runtime.
    2. Handles cannot be resolved until `api_data_fully_ready()` is true -
       they are invalid during warmup - so resolution happens lazily inside
       the callback rather than up front.
    """
    EnergyPlusAPI = _import_energyplus()
    from supervisor import OverrideStats, supervise

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    stats = OverrideStats()

    agent = None
    if mode == "ai":
        from llm_agent import HVACAgent
        # ToolExecutor needs a building for its clamp/state plumbing; the
        # EnergyPlus values below are what actually drive the decision.
        agent = HVACAgent(ToolExecutor(MockBuilding()), model=model)

    # Must be requested before run_energyplus(), not inside the callback.
    api.exchange.request_variable(state, "Zone Mean Air Temperature", zone)
    api.exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")

    f, writer, path = open_log(mode)
    h = {"zone_temp": None, "outdoor": None, "heat_sp": None, "resolved": False}
    counters = {"steps": 0, "actuated": 0}

    def on_timestep(s):
        if not api.exchange.api_data_fully_ready(s):
            return

        if not h["resolved"]:
            h["zone_temp"] = api.exchange.get_variable_handle(
                s, "Zone Mean Air Temperature", zone)
            h["outdoor"] = api.exchange.get_variable_handle(
                s, "Site Outdoor Air Drybulb Temperature", "Environment")
            h["heat_sp"] = api.exchange.get_actuator_handle(
                s, "Zone Temperature Control", "Heating Setpoint", zone)
            h["resolved"] = True
            bad = [k for k in ("zone_temp", "outdoor", "heat_sp") if h[k] == -1]
            if bad:
                print(f"\n[{mode}] could not resolve handles: {', '.join(bad)}\n"
                      f"       zone name used: {zone!r}\n"
                      f"       run `--mode discover` to list valid zone names\n",
                      file=sys.stderr)
                api.runtime.stop_simulation(s)
                return
            print(f"[{mode}] handles resolved for zone {zone!r}, controlling...")

        zone_temp = api.exchange.get_variable_value(s, h["zone_temp"])
        outdoor = api.exchange.get_variable_value(s, h["outdoor"])
        minute_of_day = int(api.exchange.hour(s) * 60 + api.exchange.minutes(s)) % (24 * 60)
        day = api.exchange.day_of_year(s)
        occupied = 8 * 60 <= minute_of_day < 18 * 60

        est = {
            "zone_temp_c": zone_temp,
            "outdoor_temp_c": outdoor,
            "occupied": occupied,
            "minutes_until_occupancy_change": (
                (18 * 60) - minute_of_day if occupied
                else ((8 * 60) - minute_of_day if minute_of_day < 8 * 60
                      else (8 * 60 + 24 * 60) - minute_of_day)
            ),
            "minutes_until_price_peak": grid.minutes_until_peak(minute_of_day),
            "comfort_band_c": [COMFORT_MIN_C, COMFORT_MAX_C],
        }

        latency, proposed, overridden, reason = 0.0, None, False, ""
        if mode == "baseline":
            setpoint = rule_based_setpoint(occupied)
        else:
            result = agent.run_step()
            proposed, latency = result.final_setpoint_c, result.latency_s
            if use_supervisor:
                decision = supervise(proposed, est)
                stats.record(decision)
                setpoint, overridden, reason = (
                    decision.setpoint_c, decision.overridden, decision.reason)
            else:
                setpoint = proposed if proposed is not None else rule_based_setpoint(occupied)

        api.exchange.set_actuator_value(s, h["heat_sp"], setpoint)
        counters["actuated"] += 1

        kwh = api.exchange.get_meter_value(s, meter["h"]) / 3_600_000 if meter["h"] != -1 else 0.0
        writer.writerow({
            "step": counters["steps"], "day": day, "minute_of_day": minute_of_day,
            "zone_temp_c": round(zone_temp, 3), "outdoor_temp_c": round(outdoor, 2),
            "setpoint_c": setpoint, "occupied": occupied,
            "pmv": round((zone_temp - 22.0) / 2.0, 2) if occupied else 0.0,
            "energy_kwh_step": round(kwh, 4),
            "cost_step": round(kwh * grid.price_per_kwh(minute_of_day), 5),
            "carbon_g_step": round(kwh * grid.carbon_intensity(minute_of_day), 2),
            "latency_s": latency, "agent_ran": mode == "ai",
            "agent_proposed_c": proposed, "overridden": overridden,
            "override_reason": reason, "reasoning": "",
        })
        counters["steps"] += 1

    meter = {"h": -1}

    def on_meters_ready(s):
        if meter["h"] == -1 and api.exchange.api_data_fully_ready(s):
            meter["h"] = api.exchange.get_meter_handle(s, "Heating:EnergyTransfer")

    api.runtime.callback_begin_new_environment(state, on_meters_ready)
    api.runtime.callback_end_of_zone_timestep_after_zone_reporting(state, on_timestep)

    print(f"[{mode}] running EnergyPlus: {idf_path}")
    try:
        api.runtime.run_energyplus(state, ["-w", epw_path, "-d", f"out/{mode}", idf_path])
    finally:
        f.close()

    if counters["actuated"] == 0:
        print(f"[{mode}] WARNING: no timesteps were actuated - the log is empty.\n"
              f"         Handles never resolved. Check the zone name with --mode discover.",
              file=sys.stderr)
    else:
        print(f"[{mode}] {counters['steps']} timesteps logged to {path}")
        if mode == "ai" and use_supervisor:
            print(f"[{mode}] supervisor: {stats.summary()}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Eco-Loop closed-loop control runner")
    parser.add_argument(
        "--mode", required=True,
        choices=["mock", "mock-rules", "mock-ai", "discover", "baseline", "ai"],
        help="mock/mock-rules/mock-ai use the lightweight simulator "
             "(mock-rules = supervisor rules with no LLM, the control arm); "
             "discover lists the zone names in an IDF; "
             "baseline/ai use real EnergyPlus",
    )
    parser.add_argument("--hours", type=float, default=24.0, help="mock horizon in hours")
    parser.add_argument("--step-minutes", type=int, default=15, help="mock step size")
    parser.add_argument("--agent-every-n-steps", type=int, default=1,
                         help="call the LLM every N steps in mock-ai mode (reduces latency cost)")
    parser.add_argument("--model", default="qwen2.5-coder:1.5b", help="Ollama model name")
    parser.add_argument("--no-supervisor", action="store_true",
                         help="apply the agent's raw setpoints without the deterministic "
                              "override layer - use this to measure unassisted agent quality")
    parser.add_argument("--idf", help="path to .idf (required for discover/baseline/ai)")
    parser.add_argument("--epw", help="path to .epw weather file (required for discover/baseline/ai)")
    parser.add_argument("--zone", help="zone name to control (required for baseline/ai); "
                                        "find it with --mode discover")
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
            if not args.zone:
                parser.error("--zone is required for baseline/ai modes; "
                             "find valid zone names with --mode discover")
            run_real_energyplus(args.mode, args.idf, args.epw, args.model,
                                args.zone, use_supervisor=not args.no_supervisor)


if __name__ == "__main__":
    main()
