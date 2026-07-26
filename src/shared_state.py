"""Shared state between the EnergyPlus process and the MCP server process.

The MCP server runs as a separate stdio subprocess, so it cannot see the
simulation's Python objects. The exchange is therefore two small JSON files,
written atomically (write to .tmp, then os.replace, which is atomic on NTFS)
so a reader never observes a half-written file:

    runtime_state.json   simulation -> tools   (sensors, energy, policy)
    pending_policy.json  tools -> simulation   (setpoint writes from the agent)

A threading lock guards the in-process side, because the EnergyPlus callback
and any tool call in the same process can touch the store concurrently.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

STATE_FILE = cfg.OUT / "runtime_state.json"
POLICY_FILE = cfg.OUT / "pending_policy.json"

HISTORY_STEPS = 96          # 24 h at a 15-minute timestep
SECONDS_PER_STEP = 3600 / cfg.TIMESTEPS_PER_HOUR


def _atomic_write(path: Path, payload: dict[str, Any], attempts: int = 6) -> bool:
    """Atomically replace `path`, retrying around Windows sharing violations.

    os.replace is atomic, but on Windows it raises WinError 5 if the
    destination is open in another process -- which happens routinely here
    because the MCP server reads this file while the simulation writes it.
    State is telemetry, so a dropped write is preferable to killing the run:
    retry briefly, then give up and let the next timestep republish.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        for attempt in range(attempts):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.002 * (attempt + 1))
        return False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _safe_read(path: Path, attempts: int = 5) -> dict[str, Any] | None:
    """Read JSON, retrying past a concurrent replace on the writer side."""
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            time.sleep(0.002 * (attempt + 1))
    return None


class StateStore:
    """Simulation-side writer. One instance per run."""

    def __init__(self, mode: str, baseline_series: list[float] | None = None):
        self._lock = threading.RLock()
        self.mode = mode
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_STEPS)
        self.baseline_series = baseline_series or []
        self.latest: dict[str, Any] = {}

    def publish(self, state, policy) -> None:
        """Called from the EnergyPlus callback once per timestep."""
        with self._lock:
            self.history.append({
                "sim_time": state.sim_time,
                "kwh": round(state.elec_j * cfg.J_TO_KWH, 5),
                "hvac_kwh": round(state.hvac_j * cfg.J_TO_KWH, 5),
                "mean_temp": round(
                    sum(state.zone_temps.values()) / len(state.zone_temps), 2),
                "outdoor": state.outdoor_temp,
                "occupied": state.occupied,
                "heating_sp": policy.heating_sp,
                "cooling_sp": policy.cooling_sp,
            })

            baseline_to_date = None
            if self.baseline_series and state.step <= len(self.baseline_series):
                baseline_to_date = round(self.baseline_series[state.step - 1], 4)

            self.latest = {
                "mode": self.mode,
                "step": state.step,
                "sim_time": state.sim_time,
                "hour": state.hour,
                "day_of_week": state.day_of_week,
                "zone_temps": state.zone_temps,
                "zone_rh": state.zone_rh,
                "outdoor_temp": state.outdoor_temp,
                "occupancy": state.occupancy,
                "occupied": state.occupied,
                "cumulative_kwh": round(state.cumulative_kwh, 4),
                "cumulative_hvac_kwh": round(state.cumulative_hvac_kwh, 4),
                "baseline_cumulative_kwh": baseline_to_date,
                "current_policy": policy.to_dict(),
                "recent": list(self.history),
            }
            _atomic_write(STATE_FILE, self.latest)

    def take_pending_policy(self) -> dict[str, Any] | None:
        """Consume a setpoint write left by the set_setpoints tool."""
        with self._lock:
            payload = _safe_read(POLICY_FILE)
            if payload:
                try:
                    POLICY_FILE.unlink()
                except OSError:
                    pass
            return payload

    @staticmethod
    def clear() -> None:
        for path in (STATE_FILE, POLICY_FILE):
            try:
                path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Tool-side readers (used by the MCP server process)
# ---------------------------------------------------------------------------
def read_state() -> dict[str, Any] | None:
    return _safe_read(STATE_FILE)


def write_pending_policy(payload: dict[str, Any]) -> None:
    _atomic_write(POLICY_FILE, payload)


def summarise_energy(state: dict[str, Any], window_hours: float) -> dict[str, Any]:
    """kWh, peak demand and vs-baseline delta over a trailing window."""
    recent = state.get("recent") or []
    steps = max(1, int(window_hours * cfg.TIMESTEPS_PER_HOUR))
    window = recent[-steps:]

    if not window:
        return {"window_hours": window_hours, "samples": 0,
                "kwh": 0.0, "hvac_kwh": 0.0, "peak_kw": 0.0}

    kwh = sum(r["kwh"] for r in window)
    hvac_kwh = sum(r["hvac_kwh"] for r in window)
    # Each sample is energy over one timestep; kW = kWh / hours_per_step.
    hours_per_step = 1.0 / cfg.TIMESTEPS_PER_HOUR
    peak_kw = max(r["kwh"] for r in window) / hours_per_step

    out = {
        "window_hours": window_hours,
        "samples": len(window),
        "kwh": round(kwh, 3),
        "hvac_kwh": round(hvac_kwh, 3),
        "peak_kw": round(peak_kw, 2),
        "mean_outdoor_c": round(
            sum(r["outdoor"] for r in window) / len(window), 2),
        "mean_zone_c": round(
            sum(r["mean_temp"] for r in window) / len(window), 2),
        "cumulative_kwh": state.get("cumulative_kwh"),
    }

    baseline = state.get("baseline_cumulative_kwh")
    if baseline is not None and state.get("cumulative_kwh") is not None:
        delta = state["cumulative_kwh"] - baseline
        out["baseline_cumulative_kwh"] = baseline
        out["delta_vs_baseline_kwh"] = round(delta, 3)
        out["pct_vs_baseline"] = round(delta / baseline * 100, 2) if baseline else None
    return out
