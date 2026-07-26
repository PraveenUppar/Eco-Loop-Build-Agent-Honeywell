"""Metrics computed from eplusout.csv -- the authoritative simulation output.

Deliberately read from the CSV rather than the live meter readings: the CSV is
what EnergyPlus itself reports, so the headline numbers cannot be an artefact
of how the runner accumulates meters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

FACILITY = "Electricity:Facility [J](TimeStep)"
COOLING = "Cooling:Electricity [J](TimeStep)"
FANS = "Fans:Electricity [J](TimeStep)"
OUTDOOR = "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"
OCCUPANCY = "OCCUPY-1:Schedule Value [](TimeStep)"

HOURS_PER_STEP = 1.0 / cfg.TIMESTEPS_PER_HOUR


def zone_temp_col(zone: str) -> str:
    return f"{zone}:Zone Mean Air Temperature [C](TimeStep)"


def load_run(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def compute(csv_path: str | Path, label: str,
            baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Energy, peak demand and comfort for one run."""
    df = load_run(csv_path)

    kwh = df[FACILITY].sum() * cfg.J_TO_KWH
    cooling_kwh = df[COOLING].sum() * cfg.J_TO_KWH if COOLING in df else 0.0
    fans_kwh = df[FANS].sum() * cfg.J_TO_KWH if FANS in df else 0.0

    # kW for a timestep = energy in that step / step duration in hours.
    demand_kw = df[FACILITY] * cfg.J_TO_KWH / HOURS_PER_STEP
    peak_kw = float(demand_kw.max())

    occupied = df[OCCUPANCY] > 0.05
    zone_cols = [zone_temp_col(z) for z in cfg.ZONES]

    # Comfort is only meaningful when people are present. Count each
    # zone-timestep as one observation.
    occ_temps = df.loc[occupied, zone_cols]
    total_obs = int(occ_temps.size)
    outside = ((occ_temps < cfg.COMFORT_LOW) | (occ_temps > cfg.COMFORT_HIGH))
    outside_obs = int(outside.to_numpy().sum())
    comfort_pct = (100.0 * outside_obs / total_obs) if total_obs else 0.0

    result: dict[str, Any] = {
        "label": label,
        "timesteps": int(len(df)),
        "total_kwh": round(float(kwh), 2),
        "cooling_kwh": round(float(cooling_kwh), 2),
        "fans_kwh": round(float(fans_kwh), 2),
        "hvac_kwh": round(float(cooling_kwh + fans_kwh), 2),
        "peak_kw": round(peak_kw, 3),
        "mean_outdoor_c": round(float(df[OUTDOOR].mean()), 2),
        "occupied_zone_observations": total_obs,
        "comfort_violation_pct": round(comfort_pct, 2),
        "mean_occupied_zone_c": round(float(occ_temps.mean().mean()), 2)
        if total_obs else None,
        "max_occupied_zone_c": round(float(occ_temps.max().max()), 2)
        if total_obs else None,
        "min_occupied_zone_c": round(float(occ_temps.min().min()), 2)
        if total_obs else None,
    }

    if baseline:
        base_kwh = baseline["total_kwh"]
        base_hvac = baseline["hvac_kwh"]
        result["baseline_kwh"] = base_kwh
        result["savings_kwh"] = round(base_kwh - result["total_kwh"], 2)
        result["savings_pct"] = round(
            (base_kwh - result["total_kwh"]) / base_kwh * 100, 2)
        result["hvac_savings_pct"] = round(
            (base_hvac - result["hvac_kwh"]) / base_hvac * 100, 2) if base_hvac else None
        result["peak_reduction_pct"] = round(
            (baseline["peak_kw"] - result["peak_kw"]) / baseline["peak_kw"] * 100, 2)
        result["comfort_delta_pct"] = round(
            result["comfort_violation_pct"] - baseline["comfort_violation_pct"], 2)

    return result


def cumulative_kwh_series(csv_path: str | Path) -> list[float]:
    """Per-timestep cumulative kWh, used for the vs-baseline live comparison."""
    df = load_run(csv_path)
    return (df[FACILITY].cumsum() * cfg.J_TO_KWH).round(5).tolist()


def timeseries(csv_path: str | Path) -> pd.DataFrame:
    """Tidy frame for the dashboard."""
    df = load_run(csv_path)
    out = pd.DataFrame({
        "datetime_raw": df["Date/Time"],
        "outdoor_c": df[OUTDOOR],
        "occupied": df[OCCUPANCY] > 0.05,
        "facility_j": df[FACILITY],
        "kwh": df[FACILITY] * cfg.J_TO_KWH,
    })
    out["cumulative_kwh"] = out["kwh"].cumsum()
    out["demand_kw"] = out["kwh"] / HOURS_PER_STEP
    for zone in cfg.ZONES:
        out[zone] = df[zone_temp_col(zone)]
    out["mean_zone_c"] = out[cfg.ZONES].mean(axis=1)
    out["heating_sp"] = df[f"{cfg.ZONES[0]}:Zone Thermostat Heating "
                           f"Setpoint Temperature [C](TimeStep)"]
    out["cooling_sp"] = df[f"{cfg.ZONES[0]}:Zone Thermostat Cooling "
                           f"Setpoint Temperature [C](TimeStep)"]
    out["step"] = range(1, len(out) + 1)
    return out


def comparison_table(results: dict[str, dict[str, Any]]) -> str:
    rows = [
        f"{'mode':<12} {'kWh':>9} {'saved %':>9} {'HVAC kWh':>10} "
        f"{'peak kW':>9} {'comfort viol %':>15}",
        "-" * 70,
    ]
    for name, r in results.items():
        savings = f"{r.get('savings_pct', 0):+.2f}%" if "savings_pct" in r else "  --"
        rows.append(
            f"{name:<12} {r['total_kwh']:>9.2f} {savings:>9} "
            f"{r['hvac_kwh']:>10.2f} {r['peak_kw']:>9.2f} "
            f"{r['comfort_violation_pct']:>15.2f}")
    return "\n".join(rows)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "out/baseline_cli/eplusout.csv"
    print(json.dumps(compute(target, "adhoc"), indent=2))
