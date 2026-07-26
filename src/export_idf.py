"""Bake the agent's learned policy into a standalone EnergyPlus model.

Deliverable 2 asks for a modified IDF produced during runtime. Actuating the
API leaves no file behind -- the setpoints exist only in memory while the
simulation runs -- so this reconstructs them into real Schedule:Compact
objects and writes models/agent_optimized.idf.

The result is a genuine artefact: it runs on its own, with no LLM, no MCP
server and no Python in the loop, and reproduces the agent's policy. That
makes the agent's output portable to any EnergyPlus installation.

Method: read the per-timestep setpoints the agent actually applied, split them
by day type (weekday vs weekend) and hour, and take the median of each bucket.
The median is deliberate -- it discards the occasional outlier decision from a
3B model rather than letting it distort the exported schedule.

    python src/export_idf.py            # write and verify
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppy.modeleditor import IDF

import config as cfg

# Design-day values preserved from the original model so autosizing is
# unaffected by the exported occupied/unoccupied policy.
DESIGN_DAY = {
    cfg.HEATING_SCHEDULE: {"SummerDesignDay": 16.7, "WinterDesignDay": 22.2},
    cfg.COOLING_SCHEDULE: {"SummerDesignDay": 23.9, "WinterDesignDay": 29.4},
}


def _day_type(day_of_month: int) -> str:
    """The run period starts on a Monday, so day 1 is Monday."""
    return "WeekDays" if (day_of_month - 1) % 7 < 5 else "WeekEnds"


def load_applied(run_log: Path) -> list[dict[str, Any]]:
    rows = []
    with open(run_log, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{run_log} is empty -- run the agent first")
    return rows


def build_profiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """Median applied setpoint for each (day type, hour) bucket."""
    buckets: dict[tuple[str, str, int], list[float]] = defaultdict(list)

    for row in rows:
        # sim_time is "MM-DD HH:MM", stamped at the END of the timestep, so
        # 01:00 closes the hour that starts at 00:00 -> bucket it to hour 0.
        date_part, time_part = row["sim_time"].split()
        day = int(date_part.split("-")[1])
        hour, minute = (int(x) for x in time_part.split(":"))
        hour_index = (hour - 1) % 24 if minute == 0 else hour

        day_type = _day_type(day)
        buckets[("heating", day_type, hour_index)].append(row["applied_heating"])
        buckets[("cooling", day_type, hour_index)].append(row["applied_cooling"])

    profiles: dict[str, dict[str, list[float]]] = {
        "heating": {"WeekDays": [], "WeekEnds": []},
        "cooling": {"WeekDays": [], "WeekEnds": []},
    }
    for kind in ("heating", "cooling"):
        for day_type in ("WeekDays", "WeekEnds"):
            hourly = []
            for hour in range(24):
                values = buckets.get((kind, day_type, hour))
                if values:
                    # Round to 0.1 C: readable in the IDF, no meaningful loss.
                    hourly.append(round(statistics.median(values), 1))
                else:
                    hourly.append(cfg.FALLBACK_HEATING if kind == "heating"
                                  else cfg.FALLBACK_COOLING)
            profiles[kind][day_type] = hourly
    return profiles


def _compact_fields(schedule_name: str,
                    hourly: dict[str, list[float]]) -> list[str]:
    """Build the Schedule:Compact field list.

    In the IDD, "Until: 07:00" and its value are two separate fields even
    though EnergyPlus writes them on one line.
    """
    fields: list[str] = ["Through: 12/31"]

    for day_type, value in DESIGN_DAY[schedule_name].items():
        fields += [f"For: {day_type}", "Until: 24:00", str(value)]

    for day_type, label in (("WeekDays", "For: WeekDays"),
                            ("WeekEnds", "For: WeekEnds Holiday")):
        fields.append(label)
        values = hourly[day_type]
        # Collapse consecutive equal hours into a single Until entry.
        start = 0
        for hour in range(1, 25):
            if hour == 24 or values[hour] != values[start]:
                fields += [f"Until: {hour:02d}:00", str(values[start])]
                start = hour
                if hour == 24:
                    break

    fields += ["For: AllOtherDays", "Until: 24:00",
               str(hourly["WeekEnds"][0])]
    return fields


def write_idf(profiles: dict[str, dict[str, list[float]]]) -> Path:
    IDF.setiddname(str(cfg.IDD_PATH))
    idf = IDF(str(cfg.SIM_IDF))

    targets = {
        cfg.HEATING_SCHEDULE: profiles["heating"],
        cfg.COOLING_SCHEDULE: profiles["cooling"],
    }

    for name, hourly in targets.items():
        for obj in list(idf.idfobjects["SCHEDULE:COMPACT"]):
            if obj.Name.lower() == name.lower():
                idf.removeidfobject(obj)

        new = idf.newidfobject("SCHEDULE:COMPACT")
        new.Name = name
        new.Schedule_Type_Limits_Name = "Temperature"
        for i, value in enumerate(_compact_fields(name, hourly), start=1):
            setattr(new, f"Field_{i}", value)

    idf.saveas(str(cfg.AGENT_IDF))
    return cfg.AGENT_IDF


def verify(idf_path: Path) -> dict[str, Any] | None:
    """Run the exported model standalone to prove it is valid and works."""
    outdir = cfg.OUT / "agent_optimized_check"
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(cfg.ENERGYPLUS_DIR / "energyplus.exe"),
         "-w", str(cfg.WEATHER), "-d", str(outdir), "-r", str(idf_path)],
        capture_output=True, text=True, check=False,
    )
    csv = outdir / "eplusout.csv"
    if not csv.exists():
        err = (outdir / "eplusout.err")
        print("  verification FAILED -- no CSV produced")
        if err.exists():
            print(err.read_text(encoding="utf-8", errors="replace")[-800:])
        return None

    import metrics
    baseline_json = cfg.RESULTS / "baseline" / "metrics.json"
    baseline = (json.loads(baseline_json.read_text(encoding="utf-8"))
                if baseline_json.exists() else None)
    return metrics.compute(csv, "agent_optimized", baseline=baseline)


def main() -> None:
    run_log = cfg.RESULTS / "agent" / "run_log.jsonl"
    if not run_log.exists():
        raise SystemExit("no agent run found -- run "
                         "`python src/run_experiment.py --mode agent` first")

    rows = load_applied(run_log)
    profiles = build_profiles(rows)

    print(f"derived from {len(rows)} timesteps of applied setpoints\n")
    for kind in ("heating", "cooling"):
        for day_type in ("WeekDays", "WeekEnds"):
            values = profiles[kind][day_type]
            print(f"  {kind:8} {day_type:9} " +
                  " ".join(f"{v:.0f}" for v in values))
    print()

    path = write_idf(profiles)
    print(f"wrote {path}")

    result = verify(path)
    if result:
        print(f"\nverified: the exported IDF runs standalone")
        print(f"  {result['total_kwh']:.2f} kWh over {result['timesteps']} timesteps")
        if "savings_pct" in result:
            print(f"  {result['savings_pct']:+.2f}% vs baseline, "
                  f"comfort violations {result['comfort_violation_pct']:.2f}%")
        (cfg.RESULTS / "agent_optimized_metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
