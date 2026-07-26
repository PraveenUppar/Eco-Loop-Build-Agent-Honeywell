"""Run one or all control modes and write comparable metrics.

Identical IDF, identical weather, identical run period. The only thing that
differs between modes is the controller.

    python src/run_experiment.py --mode baseline
    python src/run_experiment.py --mode all
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import metrics
import shared_state
from controllers import RuleBasedController
from eplus_runner import BaselineController, EnergyPlusRunner

MODES = ("baseline", "rulebased", "agent")


def _results_dir(mode: str) -> Path:
    path = cfg.RESULTS / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def _baseline_series() -> list[float] | None:
    """Per-timestep cumulative baseline kWh, for the live vs-baseline tool."""
    csv = _results_dir("baseline") / "eplusout.csv"
    if csv.exists():
        return metrics.cumulative_kwh_series(csv)
    return None


def run_mode(mode: str, verbose: bool = True,
             demo: bool = False) -> dict[str, Any]:
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; choose from {MODES}")

    outdir = _results_dir(mode)
    shared_state.StateStore.clear()

    agent = None
    bridge = None

    if mode == "baseline":
        controller = BaselineController()
    elif mode == "rulebased":
        controller = RuleBasedController()
    else:
        from agent import AgentController
        from mcp_bridge import MCPBridge
        bridge = MCPBridge()
        if verbose:
            print(f"[experiment] MCP tools available: {bridge.tool_names}")
        agent = AgentController(bridge=bridge, outdir=outdir,
                                verbose=verbose, trace=demo)
        controller = agent

    started = time.time()
    try:
        runner = EnergyPlusRunner(
            controller=controller,
            outdir=outdir,
            verbose=verbose,
            baseline_series=None if mode == "baseline" else _baseline_series(),
        )
        exit_code = runner.run()
    finally:
        if agent:
            agent.close()
        if bridge:
            bridge.close()

    wall_clock = time.time() - started
    if exit_code != 0:
        raise SystemExit(f"{mode}: EnergyPlus exited {exit_code}")

    csv_path = outdir / "eplusout.csv"
    if not csv_path.exists():
        raise SystemExit(f"{mode}: no eplusout.csv produced in {outdir}")

    baseline = None
    if mode != "baseline":
        base_json = _results_dir("baseline") / "metrics.json"
        if base_json.exists():
            baseline = json.loads(base_json.read_text(encoding="utf-8"))

    result = metrics.compute(csv_path, mode, baseline=baseline)
    result["wall_clock_s"] = round(wall_clock, 1)
    result["clamp_violations"] = len(runner.violations)
    result["envelope_engaged_steps"] = runner.operating_envelope_engaged

    if agent:
        result["agent"] = agent.summary()
        # Wall clock minus LLM time shows what the supervisor actually costs.
        result["sim_only_s"] = round(
            wall_clock - agent.summary()["total_llm_seconds"], 1)

    (outdir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    if runner.violations:
        (outdir / "violations.txt").write_text(
            "\n".join(runner.violations), encoding="utf-8")

    # Keep a tidy timeseries next to the raw output for the dashboard.
    metrics.timeseries(csv_path).to_csv(outdir / "timeseries.csv", index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all",
                        choices=(*MODES, "all"),
                        help="which controller to run")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="print each stage of the control loop "
                             "(SENSE -> TOOL -> LLM -> VALID -> ACT) for the "
                             "demonstration recording")
    args = parser.parse_args()

    if not cfg.SIM_IDF.exists():
        raise SystemExit("models/simulation.idf missing -- run "
                         "`python src/prepare_model.py` first")

    modes = MODES if args.mode == "all" else (args.mode,)
    results: dict[str, dict[str, Any]] = {}

    for mode in modes:
        print(f"\n{'=' * 70}\n  {mode.upper()}\n{'=' * 70}")
        results[mode] = run_mode(mode, verbose=not args.quiet, demo=args.demo)

    # Modes run individually still get compared against whatever is on disk.
    for mode in MODES:
        if mode not in results:
            path = _results_dir(mode) / "metrics.json"
            if path.exists():
                results[mode] = json.loads(path.read_text(encoding="utf-8"))

    ordered = {m: results[m] for m in MODES if m in results}
    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    print(metrics.comparison_table(ordered))

    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS / "summary.json").write_text(
        json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"\nwrote {cfg.RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
