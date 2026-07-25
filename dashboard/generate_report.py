"""
Compares a baseline run log against an AI run log and produces:
  - a printed summary (kWh totals, % savings, comfort violation counts)
  - dashboard/report.html with charts, for the submission deliverable

Usage:
    python dashboard/generate_report.py \\
        --baseline logs/mock.csv --ai logs/mock-ai.csv \\
        --out dashboard/report.html
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grid  # noqa: E402
from tools import COMFORT_MAX_C, COMFORT_MIN_C  # noqa: E402


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["occupied"] = df["occupied"].astype(str).str.lower().isin(["true", "1"])
    return df


def comfort_violations(df: pd.DataFrame) -> int:
    occupied = df[df["occupied"]]
    out_of_band = occupied[
        (occupied["zone_temp_c"] < COMFORT_MIN_C) | (occupied["zone_temp_c"] > COMFORT_MAX_C)
    ]
    return len(out_of_band)


def _peak_kwh(df: pd.DataFrame) -> float:
    """Energy drawn during the on-peak price window."""
    peak = df[(df["minute_of_day"] >= grid.PEAK_START_HOUR * 60)
              & (df["minute_of_day"] < grid.PEAK_END_HOUR * 60)]
    return peak["energy_kwh_step"].sum()


def _pct(before: float, after: float) -> float:
    return 0.0 if before == 0 else round((before - after) / before * 100, 2)


def summarize(baseline: pd.DataFrame, ai: pd.DataFrame) -> dict:
    baseline_kwh = baseline["energy_kwh_step"].sum()
    ai_kwh = ai["energy_kwh_step"].sum()
    savings_pct = 0.0 if baseline_kwh == 0 else (baseline_kwh - ai_kwh) / baseline_kwh * 100

    # Cost and carbon are the metrics that actually move here. Total kWh
    # barely changes - the strategy shifts *when* energy is drawn rather
    # than reducing how much, so reporting kWh alone would hide the result.
    has_grid = "cost_step" in baseline.columns and "cost_step" in ai.columns
    if has_grid:
        grid_stats = {
            "baseline_cost": round(baseline["cost_step"].sum(), 2),
            "ai_cost": round(ai["cost_step"].sum(), 2),
            "cost_savings_pct": _pct(baseline["cost_step"].sum(), ai["cost_step"].sum()),
            "baseline_kg_co2": round(baseline["carbon_g_step"].sum() / 1000, 2),
            "ai_kg_co2": round(ai["carbon_g_step"].sum() / 1000, 2),
            "carbon_savings_pct": _pct(baseline["carbon_g_step"].sum(),
                                       ai["carbon_g_step"].sum()),
            "baseline_peak_kwh": round(_peak_kwh(baseline), 2),
            "ai_peak_kwh": round(_peak_kwh(ai), 2),
            "peak_shift_pct": _pct(_peak_kwh(baseline), _peak_kwh(ai)),
        }
    else:
        grid_stats = {}

    # Share of agent decisions the deterministic supervisor had to correct.
    # Reported prominently and deliberately: a high rate means the headline
    # savings reflect the override rules more than the LLM's own judgment,
    # and hiding that would misrepresent what the agent achieved.
    if {"overridden", "agent_ran"} <= set(ai.columns):
        # Count one decision per agent invocation, not per simulation step -
        # the agent runs every N steps and its decision is held in between.
        # (Selecting on "setpoint changed" would badly overstate this, since
        # an override is exactly what usually changes the setpoint.)
        ran = ai["agent_ran"].astype(str).str.lower().isin(["true", "1"])
        decisions = ai[ran]
        flags = decisions["overridden"].astype(str).str.lower().isin(["true", "1"])
        override_pct = round(flags.mean() * 100, 1) if len(decisions) else 0.0
    else:
        override_pct = None

    return {
        "baseline_kwh": round(baseline_kwh, 3),
        "ai_kwh": round(ai_kwh, 3),
        "savings_pct": round(savings_pct, 2),
        "baseline_comfort_violations": comfort_violations(baseline),
        "ai_comfort_violations": comfort_violations(ai),
        "baseline_avg_latency_s": round(baseline["latency_s"].mean(), 3),
        "ai_avg_latency_s": round(ai["latency_s"].mean(), 3),
        "supervisor_override_pct": override_pct,
        **grid_stats,
    }


def make_charts(baseline: pd.DataFrame, ai: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    # Cumulative energy comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(baseline.index, baseline["energy_kwh_step"].cumsum(), label="Baseline")
    ax.plot(ai.index, ai["energy_kwh_step"].cumsum(), label="AI-driven")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative energy (kWh)")
    ax.set_title("Cumulative energy: baseline vs AI-driven")
    ax.legend()
    fig.tight_layout()
    energy_path = os.path.join(out_dir, "energy_comparison.png")
    fig.savefig(energy_path)
    plt.close(fig)
    paths["energy"] = os.path.basename(energy_path)

    # Zone temp vs comfort band
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ai.index, ai["zone_temp_c"], label="AI zone temp", color="tab:orange")
    ax.axhspan(COMFORT_MIN_C, COMFORT_MAX_C, color="green", alpha=0.15, label="Comfort band")
    ax.set_xlabel("Step")
    ax.set_ylabel("Zone temp (C)")
    ax.set_title("AI-driven zone temperature vs comfort band")
    ax.legend()
    fig.tight_layout()
    comfort_path = os.path.join(out_dir, "comfort_band.png")
    fig.savefig(comfort_path)
    plt.close(fig)
    paths["comfort"] = os.path.basename(comfort_path)

    # Load shifting is the actual result, so show it directly: energy drawn
    # per step against the price curve. The visible signature is the AI
    # trace rising before the shaded peak and dropping inside it.
    if "cost_step" in ai.columns:
        # Hourly totals, not per-step values. Per-step energy is spiky
        # (every setpoint change shows as a transient) and dominated by
        # standing heat loss, which buries the actual difference. Hourly
        # bars are what the claim is about: how much energy lands inside
        # the expensive window.
        #
        # Grouping by hour also sidesteps a plotting artifact: minute_of_day
        # wraps from 1425 back to 0 at the end of each day, so a line plot
        # drew a spurious horizontal streak from hour 23.75 back to hour 0.
        def hourly(df):
            h = (df["minute_of_day"] // 60).astype(int)
            days = max(1, df["day"].nunique())
            return df.groupby(h)["energy_kwh_step"].sum().reindex(range(24), fill_value=0) / days

        base_h, ai_h = hourly(baseline), hourly(ai)
        fig, ax = plt.subplots(figsize=(9, 4))
        width = 0.42
        ax.bar([h - width / 2 for h in range(24)], base_h, width,
               label="Baseline", color="tab:blue")
        ax.bar([h + width / 2 for h in range(24)], ai_h, width,
               label="AI-driven", color="tab:orange")
        ax.axvspan(grid.PEAK_START_HOUR - 0.5, grid.PEAK_END_HOUR - 0.5,
                   color="red", alpha=0.12, label="Expensive hours (peak price + carbon)")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Energy used that hour (kWh, daily average)")
        ax.set_title("Load shifting: the agent buys energy before the peak, not during it")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlim(-1, 24)
        ax.legend()
        fig.tight_layout()
        shift_path = os.path.join(out_dir, "load_shift.png")
        fig.savefig(shift_path)
        plt.close(fig)
        paths["shift"] = os.path.basename(shift_path)

    return paths


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eco-Loop Savings Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; background: #0e1117; color: #e6e6e6; }}
  h1 {{ font-size: 1.4rem; }}
  .stats {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1.5rem 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.5rem; min-width: 160px; }}
  .stat .value {{ font-size: 1.6rem; font-weight: 600; }}
  .stat .label {{ font-size: 0.8rem; color: #9198a1; text-transform: uppercase; letter-spacing: 0.04em; }}
  .savings {{ color: #3fb950; }}
  .warn {{ color: #f85149; }}
  .note {{ font-size: 0.85rem; color: #9198a1; max-width: 60ch; line-height: 1.5; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0; border: 1px solid #30363d; }}
</style>
</head>
<body>
<h1>Eco-Loop Building Agent &mdash; Savings Dashboard</h1>
<p>Baseline: <code>{baseline_path}</code> &nbsp;|&nbsp; AI-driven: <code>{ai_path}</code></p>

<div class="stats">
  <div class="stat"><div class="value savings">{cost_savings_pct}%</div><div class="label">Cost reduction</div></div>
  <div class="stat"><div class="value savings">{carbon_savings_pct}%</div><div class="label">CO2 reduction</div></div>
  <div class="stat"><div class="value savings">{peak_shift_pct}%</div><div class="label">Peak-hour energy cut</div></div>
  <div class="stat"><div class="value {comfort_class}">{ai_comfort_violations}</div><div class="label">Comfort violations</div></div>
</div>

<p class="note">Total energy is roughly unchanged ({baseline_kwh} &rarr; {ai_kwh} kWh,
{savings_pct}%). That is expected and is the point: the agent shifts <em>when</em>
energy is drawn rather than reducing how much, heating while power is cheap and
clean and coasting on stored heat through the expensive carbon-heavy peak. A fixed
setback schedule cannot do this &mdash; it has no notion of what hour it is in price
terms.</p>

<div class="stats">
  <div class="stat"><div class="value">{baseline_cost} &rarr; {ai_cost}</div><div class="label">Cost (baseline &rarr; AI)</div></div>
  <div class="stat"><div class="value">{baseline_kg_co2} &rarr; {ai_kg_co2}</div><div class="label">kg CO2</div></div>
  <div class="stat"><div class="value">{baseline_peak_kwh} &rarr; {ai_peak_kwh}</div><div class="label">Peak-window kWh</div></div>
  <div class="stat"><div class="value">{ai_avg_latency_s}s</div><div class="label">Avg agent latency</div></div>
  <div class="stat"><div class="value">{supervisor_override_pct}%</div><div class="label">Supervisor overrides</div></div>
</div>

<p class="note">The supervisor override rate is the share of LLM setpoint decisions
that the deterministic safety layer had to correct (see <code>supervisor.py</code>).
The higher this number, the more the results above reflect the override rules
rather than the language model's own control judgment. Baseline comfort violations:
{baseline_comfort_violations}.</p>

<h2>Load shifting against the price peak</h2>
<img src="{shift_chart}" alt="Energy drawn vs price peak">

<h2>Cumulative energy</h2>
<img src="{energy_chart}" alt="Cumulative energy comparison">

<h2>Comfort band adherence (AI-driven run)</h2>
<img src="{comfort_chart}" alt="Comfort band chart">

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate Eco-Loop savings dashboard")
    parser.add_argument("--baseline", default="logs/mock.csv")
    parser.add_argument("--ai", default="logs/mock-ai.csv")
    parser.add_argument("--out", default="dashboard/report.html")
    args = parser.parse_args()

    baseline = load(args.baseline)
    ai = load(args.ai)
    stats = summarize(baseline, ai)

    print("=== Eco-Loop Savings Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    charts = make_charts(baseline, ai, out_dir)

    html = HTML_TEMPLATE.format(
        baseline_path=args.baseline,
        ai_path=args.ai,
        comfort_class="warn" if stats["ai_comfort_violations"] > 0 else "savings",
        energy_chart=charts["energy"],
        comfort_chart=charts["comfort"],
        shift_chart=charts.get("shift", ""),
        **stats,
    )
    with open(args.out, "w") as f:
        f.write(html)
    print(f"\nDashboard written to {args.out}")


if __name__ == "__main__":
    main()
