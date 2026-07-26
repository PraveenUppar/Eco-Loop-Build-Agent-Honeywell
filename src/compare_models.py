"""Isolate model capability as a variable: same loop, different LLM.

This is a side experiment. It does NOT touch config.py, the three reported
modes, or anything under results/{baseline,rulebased,agent}. It reuses the
identical agent, MCP tool layer, safety envelope and building model, and
changes exactly one thing -- which model answers.

The question it answers: how much of the agent's performance is limited by
the 3B local model rather than by the control design?

    python src/compare_models.py --model qwen3.5:397b-cloud
    python src/compare_models.py --model gemma4:31b-cloud --model glm-5.1:cloud
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama

import config as cfg
import metrics
import shared_state
from agent import POLICY_SCHEMA, SYSTEM_PROMPT, AgentController
from eplus_runner import EnergyPlusRunner
from mcp_bridge import MCPBridge

COMPARE_ROOT = cfg.RESULTS / "model_compare"


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def extract_json(raw: str) -> dict[str, Any] | None:
    """Parse a reply that may be wrapped in a markdown fence.

    Ollama's cloud-hosted models ignore the `format` parameter that constrains
    local models, and return ```json ... ``` instead of bare JSON. Rather than
    change the production agent, the tolerance lives here: take the outermost
    braces and parse those.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


class FenceTolerantAgent(AgentController):
    """Identical to the reported agent, but tolerant of fenced JSON replies."""

    def _ask_model(self, messages):
        started = time.time()
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=POLICY_SCHEMA,
                options={"temperature": 0.2, "num_predict": 160},
            )
            raw = response["message"]["content"]
        except Exception as exc:                          # noqa: BLE001
            return None, time.time() - started, f"<error: {exc}>"
        return extract_json(raw), time.time() - started, raw


def preflight(model: str, attempts: int = 2) -> tuple[bool, str]:
    """Confirm the model answers usefully under production conditions.

    Uses the real system prompt, not a toy one, and allows the same kind of
    second chance the agent's self-correction loop would give it.
    """
    client = ollama.Client(timeout=cfg.LLM_TIMEOUT_S)
    last = "no attempt made"
    for attempt in range(attempts):
        try:
            started = time.time()
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": "TIME 07-05 02:00 (UNOCCUPIED)\nOUTDOOR 24.0 C\n"
                                "ZONES avg 22.0 C\n\nChoose setpoints for the "
                                "next 4 hours."},
                ],
                format=POLICY_SCHEMA,
                options={"temperature": 0.2, "num_predict": 160},
            )
            latency = time.time() - started
            content = response["message"]["content"]
            parsed = extract_json(content)
            if not parsed:
                last = f"unparseable reply: {content[:110]!r}"
                continue
            if not {"heating_c", "cooling_c"} <= parsed.keys():
                last = f"missing keys in {parsed}"
                continue
            return True, f"ok, {latency:.1f}s, sample {parsed}"
        except Exception as exc:                          # noqa: BLE001
            return False, str(exc)[:200]
    return False, last


def run_with_model(model: str, verbose: bool = True) -> dict[str, Any]:
    outdir = COMPARE_ROOT / safe_name(model)
    outdir.mkdir(parents=True, exist_ok=True)
    shared_state.StateStore.clear()

    baseline_json = cfg.RESULTS / "baseline" / "metrics.json"
    if not baseline_json.exists():
        raise SystemExit("run `python src/run_experiment.py --mode baseline` first")
    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    baseline_series = metrics.cumulative_kwh_series(
        cfg.RESULTS / "baseline" / "eplusout.csv")

    bridge = MCPBridge()
    agent = FenceTolerantAgent(bridge=bridge, outdir=outdir, model=model,
                               verbose=verbose)
    started = time.time()
    try:
        runner = EnergyPlusRunner(
            controller=agent, outdir=outdir, verbose=False,
            baseline_series=baseline_series)
        exit_code = runner.run()
    finally:
        agent.close()
        bridge.close()

    if exit_code != 0:
        raise SystemExit(f"{model}: EnergyPlus exited {exit_code}")

    result = metrics.compute(outdir / "eplusout.csv", model, baseline=baseline)
    result["wall_clock_s"] = round(time.time() - started, 1)
    result["agent"] = agent.summary()
    result["clamp_violations"] = len(runner.violations)
    (outdir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        help="model to test; repeat for several")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    for model in args.model:
        print(f"\n{'=' * 72}\n  {model}\n{'=' * 72}")
        ok, detail = preflight(model)
        print(f"  preflight: {detail}")
        if not ok:
            print(f"  SKIPPED -- {model} unusable")
            continue
        results[model] = run_with_model(model, verbose=not args.quiet)

    # Show the reported local-model run alongside, for reference.
    local = cfg.RESULTS / "agent" / "metrics.json"
    reference = json.loads(local.read_text(encoding="utf-8")) if local.exists() else None
    baseline = json.loads(
        (cfg.RESULTS / "baseline" / "metrics.json").read_text(encoding="utf-8"))
    rulebased_path = cfg.RESULTS / "rulebased" / "metrics.json"
    rulebased = (json.loads(rulebased_path.read_text(encoding="utf-8"))
                 if rulebased_path.exists() else None)

    print(f"\n{'=' * 88}\n  MODEL COMPARISON (identical loop, identical building)\n{'=' * 88}")
    header = (f"{'model':<26} {'kWh':>9} {'saved %':>9} {'HVAC kWh':>10} "
              f"{'comfort %':>10} {'retries':>8} {'fallbacks':>10} {'median s':>9}")
    print(header)
    print("-" * len(header))

    def row(label: str, r: dict[str, Any]) -> str:
        a = r.get("agent") or {}
        saved = f"{r.get('savings_pct', 0):+.2f}%" if "savings_pct" in r else "--"
        return (f"{label:<26} {r['total_kwh']:>9.2f} {saved:>9} "
                f"{r['hvac_kwh']:>10.2f} {r['comfort_violation_pct']:>10.2f} "
                f"{a.get('retries', '-'):>8} {a.get('fallbacks', '-'):>10} "
                f"{a.get('median_latency_s', '-'):>9}")

    print(row("baseline (no control)", baseline))
    if rulebased:
        print(row("rule-based (no LLM)", rulebased))
    if reference:
        print(row(f"{reference.get('agent', {}).get('model', 'local')} (reported)",
                  reference))
    for model, r in results.items():
        print(row(model, r))

    out = COMPARE_ROOT / "comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
