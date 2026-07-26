"""Extended-horizon endurance test of the closed loop.

Evaluation criterion 1 asks how robustly the pipeline executes over an extended
simulation time horizon. The reported three-week experiment does not answer
that on its own, so this runs the same loop over a full cooling season and
records whether anything degrades: crashes, handle failures, dropped state
writes, clamp violations, comfort breaches, or a rising fallback rate.

Writes to results/endurance/ and never touches the reported results.

    python src/endurance_run.py                      # 1 Jun - 31 Aug
    python src/endurance_run.py --start 5 1 --end 9 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

OUTDIR = cfg.RESULTS / "endurance"


def build_idf(start: tuple[int, int], end: tuple[int, int]) -> Path:
    """Build a long-period model, reusing the normal preparation path."""
    from eppy.modeleditor import IDF
    import prepare_model

    cfg.RUN_START, cfg.RUN_END = start, end
    prepare_model.prepare()                     # writes cfg.SIM_IDF

    IDF.setiddname(str(cfg.IDD_PATH))
    idf = IDF(str(cfg.SIM_IDF))
    target = OUTDIR / "endurance.idf"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(target))
    return target


def run(mode: str, idf: Path, baseline_series=None) -> dict[str, Any]:
    import metrics
    import shared_state
    from controllers import RuleBasedController
    from eplus_runner import BaselineController, EnergyPlusRunner

    shared_state.StateStore.clear()
    outdir = OUTDIR / mode
    outdir.mkdir(parents=True, exist_ok=True)

    agent = bridge = None
    if mode == "baseline":
        controller = BaselineController()
    elif mode == "rulebased":
        controller = RuleBasedController()
    else:
        from agent import AgentController
        from mcp_bridge import MCPBridge
        bridge = MCPBridge()
        agent = AgentController(bridge=bridge, outdir=outdir, verbose=False)
        controller = agent

    started = time.time()
    try:
        runner = EnergyPlusRunner(controller=controller, outdir=outdir,
                                  idf=idf, verbose=False,
                                  baseline_series=baseline_series)
        exit_code = runner.run()
    finally:
        if agent:
            agent.close()
        if bridge:
            bridge.close()

    elapsed = time.time() - started
    csv = outdir / "eplusout.csv"
    result: dict[str, Any] = {
        "mode": mode,
        "exit_code": exit_code,
        "completed": exit_code == 0 and csv.exists(),
        "timesteps_logged": runner.step,
        "clamp_violations": len(runner.violations),
        "envelope_engaged_steps": runner.operating_envelope_engaged,
        "wall_clock_s": round(elapsed, 1),
    }
    if agent:
        result["agent"] = agent.summary()
    if csv.exists():
        result["metrics"] = metrics.compute(csv, mode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", nargs=2, type=int, default=[6, 1],
                        metavar=("MONTH", "DAY"))
    parser.add_argument("--end", nargs=2, type=int, default=[8, 31],
                        metavar=("MONTH", "DAY"))
    args = parser.parse_args()

    start, end = tuple(args.start), tuple(args.end)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"ENDURANCE RUN  {start[0]}/{start[1]} -> {end[0]}/{end[1]}")
    idf = build_idf(start, end)
    print(f"model: {idf}\n")

    import metrics

    results = {}
    for mode in ("baseline", "agent"):
        print(f"--- {mode} ---", flush=True)
        series = None
        if mode != "baseline":
            base_csv = OUTDIR / "baseline" / "eplusout.csv"
            if base_csv.exists():
                series = metrics.cumulative_kwh_series(base_csv)
        results[mode] = run(mode, idf, baseline_series=series)
        r = results[mode]
        print(f"  completed={r['completed']}  steps={r['timesteps_logged']}  "
              f"wall={r['wall_clock_s']}s  clamp_violations={r['clamp_violations']}")
        if "metrics" in r:
            m = r["metrics"]
            print(f"  {m['total_kwh']:.1f} kWh   comfort violations "
                  f"{m['comfort_violation_pct']:.2f}%")
        if "agent" in r:
            print(f"  agent: {json.dumps(r['agent'])}")
        print(flush=True)

    base = results["baseline"].get("metrics", {})
    agent = results["agent"].get("metrics", {})
    if base and agent:
        saved = (base["total_kwh"] - agent["total_kwh"]) / base["total_kwh"] * 100
        hvac = (base["hvac_kwh"] - agent["hvac_kwh"]) / base["hvac_kwh"] * 100
        results["comparison"] = {
            "baseline_kwh": base["total_kwh"],
            "agent_kwh": agent["total_kwh"],
            "savings_pct": round(saved, 2),
            "hvac_savings_pct": round(hvac, 2),
            "baseline_comfort_violation_pct": base["comfort_violation_pct"],
            "agent_comfort_violation_pct": agent["comfort_violation_pct"],
        }

    both_ok = all(results[m]["completed"] for m in ("baseline", "agent"))
    total_violations = sum(results[m]["clamp_violations"] for m in ("baseline", "agent"))
    results["verdict"] = {
        "ran_to_completion": both_ok,
        "total_clamp_violations": total_violations,
        "timesteps": results["agent"]["timesteps_logged"],
    }

    (OUTDIR / "endurance.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"  ran to completion : {both_ok}")
    print(f"  timesteps         : {results['agent']['timesteps_logged']}")
    print(f"  clamp violations  : {total_violations}")
    if "comparison" in results:
        c = results["comparison"]
        print(f"  savings           : {c['savings_pct']:+.2f}%  "
              f"(HVAC {c['hvac_savings_pct']:+.2f}%)")
        print(f"  comfort violations: baseline "
              f"{c['baseline_comfort_violation_pct']:.2f}%  agent "
              f"{c['agent_comfort_violation_pct']:.2f}%")
    print(f"\nwrote {OUTDIR / 'endurance.json'}")


if __name__ == "__main__":
    main()
