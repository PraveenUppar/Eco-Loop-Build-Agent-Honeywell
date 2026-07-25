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

from tools import MockBuilding, ToolExecutor

LOG_COLUMNS = [
    "step", "day", "minute_of_day", "zone_temp_c", "outdoor_temp_c",
    "setpoint_c", "occupied", "pmv", "energy_kwh_step", "latency_s", "reasoning",
]


def rule_based_setpoint(occupied: bool) -> float:
    """Fixed schedule baseline: 21C occupied, 18C setback unoccupied."""
    return 21.0 if occupied else 18.0


def open_log(mode: str):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"{mode}.csv")
    f = open(path, "w", newline="")
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
                "energy_kwh_step": s.energy_kwh_step, "latency_s": 0.0, "reasoning": "",
            })
    finally:
        f.close()
    print(f"[mock] {steps} steps logged to {path}")
    return path


def run_mock_ai(hours: int, step_minutes: int, model: str, agent_every_n_steps: int) -> str:
    from llm_agent import HVACAgent  # deferred import: needs ollama installed

    building = MockBuilding(step_minutes=step_minutes)
    executor = ToolExecutor(building)
    agent = HVACAgent(executor, model=model)
    steps = int(hours * 60 / step_minutes)

    f, writer, path = open_log("mock-ai")
    current_setpoint = building.state.setpoint_c
    reasoning = ""
    latency = 0.0
    try:
        for i in range(steps):
            if i % agent_every_n_steps == 0:
                result = agent.run_step()
                current_setpoint = result.final_setpoint_c
                reasoning = result.reasoning
                latency = result.latency_s
                print(f"  step {i}: setpoint -> {current_setpoint}C "
                      f"({latency}s) | {reasoning[:80]}")

            s = building.step(current_setpoint)
            writer.writerow({
                "step": i, "day": s.day, "minute_of_day": s.minute_of_day,
                "zone_temp_c": s.zone_temp_c, "outdoor_temp_c": s.outdoor_temp_c,
                "setpoint_c": s.setpoint_c, "occupied": s.occupied, "pmv": s.pmv,
                "energy_kwh_step": s.energy_kwh_step, "latency_s": latency,
                "reasoning": reasoning,
            })
    finally:
        f.close()
    print(f"[mock-ai] {steps} steps logged to {path}")
    return path


def run_real_energyplus(mode: str, idf_path: str, epw_path: str, model: str) -> str:
    """
    Real EnergyPlus integration via the PyEnergyPlus API.

    Requires EnergyPlus installed and its `pyenergyplus` package importable
    (add the EnergyPlus install dir to PYTHONPATH - see README). Not wired
    yet on this machine since EnergyPlus isn't installed here; this is the
    Hour 7-13 task from PLAN.md.
    """
    try:
        from pyenergyplus.api import EnergyPlusAPI
    except ImportError:
        print(
            "pyenergyplus not importable. Install EnergyPlus and add its "
            "install directory to PYTHONPATH, e.g.:\n"
            "  export PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-24-1-0\n"
            "See README.md for the full setup step.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    f, writer, path = open_log(mode)
    step_counter = {"i": 0}

    building = MockBuilding()  # only used to hold ToolExecutor's clamp logic for `ai` mode
    executor = ToolExecutor(building)
    agent = None
    if mode == "ai":
        from llm_agent import HVACAgent
        agent = HVACAgent(executor, model=model)

    # TODO (Hour 7-13, PLAN.md): resolve real zone/actuator handles.
    #   zone_temp_handle = api.exchange.get_variable_handle(
    #       state, "Zone Mean Air Temperature", "<ZONE NAME>")
    #   setpoint_actuator_handle = api.exchange.get_actuator_handle(
    #       state, "Schedule:Compact", "Schedule Value", "<SETPOINT SCHEDULE NAME>")
    # Zone/schedule names come from opening the IDF and checking the Zone
    # and Schedule:Compact objects - see PLAN.md Hour 7-13.
    zone_temp_handle = None
    setpoint_actuator_handle = None

    def on_end_of_zone_timestep(_state):
        nonlocal zone_temp_handle, setpoint_actuator_handle
        if zone_temp_handle is None:
            return  # handles not resolved yet - fill in the TODO above first
        # Placeholder read/act cycle - fill in once handles resolve:
        # zone_temp = api.exchange.get_variable_value(_state, zone_temp_handle)
        # setpoint = rule_based_setpoint(...) if mode == "baseline" else agent.run_step().final_setpoint_c
        # api.exchange.set_actuator_value(_state, setpoint_actuator_handle, setpoint)
        step_counter["i"] += 1

    api.runtime.callback_end_of_zone_timestep_after_zone_reporting(
        state, on_end_of_zone_timestep
    )

    print(f"[{mode}] running EnergyPlus on {idf_path} / {epw_path} ...")
    api.runtime.run_energyplus(state, ["-w", epw_path, "-d", "out", idf_path])
    f.close()
    print(f"[{mode}] logged to {path} ({step_counter['i']} timesteps observed)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Eco-Loop closed-loop control runner")
    parser.add_argument(
        "--mode", required=True,
        choices=["mock", "mock-ai", "baseline", "ai"],
        help="mock/mock-ai use the lightweight simulator; baseline/ai use real EnergyPlus",
    )
    parser.add_argument("--hours", type=float, default=24.0, help="mock horizon in hours")
    parser.add_argument("--step-minutes", type=int, default=15, help="mock step size")
    parser.add_argument("--agent-every-n-steps", type=int, default=1,
                         help="call the LLM every N steps in mock-ai mode (reduces latency cost)")
    parser.add_argument("--model", default="qwen2.5-coder:1.5b", help="Ollama model name")
    parser.add_argument("--idf", help="path to .idf (required for baseline/ai)")
    parser.add_argument("--epw", help="path to .epw weather file (required for baseline/ai)")
    args = parser.parse_args()

    if args.mode == "mock":
        run_mock(args.hours, args.step_minutes)
    elif args.mode == "mock-ai":
        run_mock_ai(args.hours, args.step_minutes, args.model, args.agent_every_n_steps)
    else:
        if not args.idf or not args.epw:
            parser.error("--idf and --epw are required for baseline/ai modes")
        run_real_energyplus(args.mode, args.idf, args.epw, args.model)


if __name__ == "__main__":
    main()
