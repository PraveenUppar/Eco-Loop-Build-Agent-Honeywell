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
        choices=["mock", "mock-rules", "mock-ai", "baseline", "ai"],
        help="mock/mock-rules/mock-ai use the lightweight simulator "
             "(mock-rules = supervisor rules with no LLM, the control arm); "
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
    parser.add_argument("--idf", help="path to .idf (required for baseline/ai)")
    parser.add_argument("--epw", help="path to .epw weather file (required for baseline/ai)")
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
            parser.error("--idf and --epw are required for baseline/ai modes")
        run_real_energyplus(args.mode, args.idf, args.epw, args.model)


if __name__ == "__main__":
    main()
