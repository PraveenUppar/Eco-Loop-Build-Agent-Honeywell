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


def summarize(baseline: pd.DataFrame, ai: pd.DataFrame) -> dict:
    baseline_kwh = baseline["energy_kwh_step"].sum()
    ai_kwh = ai["energy_kwh_step"].sum()
    savings_pct = 0.0 if baseline_kwh == 0 else (baseline_kwh - ai_kwh) / baseline_kwh * 100

    return {
        "baseline_kwh": round(baseline_kwh, 3),
        "ai_kwh": round(ai_kwh, 3),
        "savings_pct": round(savings_pct, 2),
        "baseline_comfort_violations": comfort_violations(baseline),
        "ai_comfort_violations": comfort_violations(ai),
        "baseline_avg_latency_s": round(baseline["latency_s"].mean(), 3),
        "ai_avg_latency_s": round(ai["latency_s"].mean(), 3),
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
  img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0; border: 1px solid #30363d; }}
</style>
</head>
<body>
<h1>Eco-Loop Building Agent &mdash; Savings Dashboard</h1>
<p>Baseline: <code>{baseline_path}</code> &nbsp;|&nbsp; AI-driven: <code>{ai_path}</code></p>

<div class="stats">
  <div class="stat"><div class="value">{baseline_kwh} kWh</div><div class="label">Baseline energy</div></div>
  <div class="stat"><div class="value">{ai_kwh} kWh</div><div class="label">AI-driven energy</div></div>
  <div class="stat"><div class="value savings">{savings_pct}%</div><div class="label">Energy reduction</div></div>
  <div class="stat"><div class="value {comfort_class}">{ai_comfort_violations}</div><div class="label">AI comfort violations</div></div>
  <div class="stat"><div class="value">{ai_avg_latency_s}s</div><div class="label">Avg agent latency/step</div></div>
</div>

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
        **stats,
    )
    with open(args.out, "w") as f:
        f.write(html)
    print(f"\nDashboard written to {args.out}")


if __name__ == "__main__":
    main()
