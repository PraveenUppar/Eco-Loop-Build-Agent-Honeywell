"""In-process EnergyPlus runner: reads sensors and writes setpoints live.

Two callbacks, deliberately separated:

  begin_system_timestep_before_predictor -> ACTUATE
      The correct hook for setpoint control: it fires before the zone predictor
      computes loads, so a setpoint written here affects this timestep.

  end_zone_timestep_after_zone_reporting -> SENSE + DECIDE
      Fires exactly once per zone timestep. Reading the meter here avoids
      double-counting energy when EnergyPlus shortens the system timestep
      below the zone timestep during difficult HVAC iterations.

The controller returns a Policy; the runner clamps it and applies it. A
controller can never write an actuator directly.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

# TMY3 weather has no meaningful year; fix one so timestamps sort and plot.
SIM_YEAR = 2017

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(r"C:\EnergyPlusV26-1-0")))

import config as cfg
from policy import Policy, clamp_policy, operating_envelope, fallback_policy
from shared_state import StateStore

from pyenergyplus.api import EnergyPlusAPI


@dataclass
class BuildingState:
    """Snapshot handed to the controller once per zone timestep."""

    month: int
    day: int
    hour: int
    minute: int
    day_of_week: int
    sim_time: str
    zone_temps: dict[str, float]
    zone_rh: dict[str, float]
    outdoor_temp: float
    occupancy: float          # OCCUPY-1 schedule fraction, 0..1
    occupied: bool
    elec_j: float             # facility electricity this timestep, Joules
    hvac_j: float             # setpoint-controllable end uses, Joules
    cumulative_kwh: float
    cumulative_hvac_kwh: float
    step: int
    current_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


class Controller(Protocol):
    name: str

    def decide(self, state: BuildingState) -> Policy | None:
        """Return a new policy, or None to keep the current one."""


class BaselineController:
    """Does nothing. The IDF's own schedules run untouched."""

    name = "baseline"
    actuates = False

    def decide(self, state: BuildingState) -> Policy | None:
        return None


class EnergyPlusRunner:
    def __init__(self, controller: Controller, outdir: Path,
                 idf: Path | None = None, epw: Path | None = None,
                 dump_api_data: bool = False, verbose: bool = True,
                 baseline_series: list[float] | None = None,
                 enforce_comfort: bool = True):
        self.controller = controller
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.idf = Path(idf or cfg.SIM_IDF)
        self.epw = Path(epw or cfg.WEATHER)
        self.dump_api_data = dump_api_data
        self.verbose = verbose

        self.actuates = getattr(controller, "actuates", True)
        # Diagnostic controllers opt out so the actuation test can prove the
        # full setpoint range really moves energy.
        self.enforce_comfort = enforce_comfort and getattr(
            controller, "enforce_comfort", True)
        self.operating_envelope_engaged = 0
        self.applied_setpoints = (cfg.FALLBACK_HEATING, cfg.FALLBACK_COOLING)

        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()

        self._handles: dict[str, int] = {}
        self._handles_ready = False
        self._dumped = False

        self.current_policy: Policy = fallback_policy("initial")
        self.violations: list[str] = []
        self.step = 0
        self.cumulative_j = 0.0
        self.cumulative_hvac_j = 0.0
        self.rows: list[dict[str, Any]] = []
        self.log_path = self.outdir / "run_log.jsonl"
        self._log_file = None

        # Publishes sensor state for the MCP server process to read, and
        # receives setpoint writes made through the set_setpoints tool.
        self.store = StateStore(mode=controller.name,
                                baseline_series=baseline_series)

    # ------------------------------------------------------------------
    # handles
    # ------------------------------------------------------------------
    def _resolve_handles(self) -> None:
        """Fetch every handle once, and fail loudly naming what is missing."""
        ex, st = self.api.exchange, self.state
        h: dict[str, int] = {}
        missing: list[str] = []

        for zone in cfg.ZONES:
            h[f"temp:{zone}"] = ex.get_variable_handle(
                st, "Zone Mean Air Temperature", zone)
            h[f"rh:{zone}"] = ex.get_variable_handle(
                st, "Zone Air Relative Humidity", zone)

        h["outdoor"] = ex.get_variable_handle(
            st, "Site Outdoor Air Drybulb Temperature", "Environment")
        h["occupancy"] = ex.get_variable_handle(
            st, "Schedule Value", cfg.OCCUPANCY_SCHEDULE)

        for meter in cfg.FACILITY_METERS + cfg.HVAC_METERS:
            h[f"meter:{meter}"] = ex.get_meter_handle(st, meter)

        # Actuating a Schedule:Compact overrides its value for the timestep.
        h["act:heating"] = ex.get_actuator_handle(
            st, "Schedule:Compact", "Schedule Value", cfg.HEATING_SCHEDULE)
        h["act:cooling"] = ex.get_actuator_handle(
            st, "Schedule:Compact", "Schedule Value", cfg.COOLING_SCHEDULE)

        for name, handle in h.items():
            if handle == -1:
                missing.append(name)

        if missing:
            raise RuntimeError(
                "EnergyPlus handle resolution failed for: "
                + ", ".join(missing)
                + "\nCheck available_api_data.csv for the exact spelling."
            )

        self._handles = h
        self._handles_ready = True
        if self.verbose:
            print(f"[runner] resolved {len(h)} handles, none missing")

    def _guards_pass(self) -> bool:
        ex, st = self.api.exchange, self.state
        if not ex.api_data_fully_ready(st):
            return False
        if ex.warmup_flag(st):
            return False
        # Only act inside the configured weather run period. This filters out
        # the sizing design days (1/21 and 7/21) whose callbacks also fire.
        month, day = ex.month(st), ex.day_of_month(st)
        if (month, day) < cfg.RUN_START or (month, day) > cfg.RUN_END:
            return False
        return True

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _on_begin_timestep(self, state) -> None:
        """ACTUATE: push the current policy onto the setpoint schedules."""
        try:
            if not self._guards_pass():
                return
            if not self._handles_ready:
                self._maybe_dump_api_data()
                self._resolve_handles()
            if not self.actuates:
                return

            ex = self.api.exchange
            heating = self.current_policy.heating_sp
            cooling = self.current_policy.cooling_sp

            # Inner-loop comfort guard, applied every timestep regardless of
            # what the supervisor asked for 4 hours ago.
            if self.enforce_comfort:
                occupied = ex.get_variable_value(
                    state, self._handles["occupancy"]) > 0.05
                heating, cooling, adjusted = operating_envelope(
                    heating, cooling, occupied)
                if adjusted:
                    self.operating_envelope_engaged += 1

            self.applied_setpoints = (heating, cooling)
            ex.set_actuator_value(state, self._handles["act:heating"], heating)
            ex.set_actuator_value(state, self._handles["act:cooling"], cooling)
        except Exception as exc:            # never let a callback kill the sim
            print(f"[runner] actuation error: {exc}", file=sys.stderr)

    def _on_end_timestep(self, state) -> None:
        """SENSE: read everything, log it, and let the controller re-plan."""
        try:
            if not self._guards_pass() or not self._handles_ready:
                return

            ex = self.api.exchange
            h = self._handles

            zone_temps = {z: round(ex.get_variable_value(state, h[f"temp:{z}"]), 3)
                          for z in cfg.ZONES}
            zone_rh = {z: round(ex.get_variable_value(state, h[f"rh:{z}"]), 2)
                       for z in cfg.ZONES}
            outdoor = round(ex.get_variable_value(state, h["outdoor"]), 3)
            occupancy = round(ex.get_variable_value(state, h["occupancy"]), 3)

            elec_j = sum(ex.get_meter_value(state, h[f"meter:{m}"])
                         for m in cfg.FACILITY_METERS)
            hvac_j = sum(ex.get_meter_value(state, h[f"meter:{m}"])
                         for m in cfg.HVAC_METERS)

            self.cumulative_j += elec_j
            self.cumulative_hvac_j += hvac_j
            self.step += 1

            # EnergyPlus reports the END of the timestep, so minutes runs 1..60
            # and hour 0..23. Rolling that through a timedelta normalises
            # "23:60" into midnight of the following day.
            month, day = ex.month(state), ex.day_of_month(state)
            stamp = (datetime(SIM_YEAR, month, day)
                     + timedelta(hours=ex.hour(state), minutes=ex.minutes(state)))
            hour, minute = stamp.hour, stamp.minute
            sim_time = stamp.strftime("%m-%d %H:%M")

            bstate = BuildingState(
                month=month, day=day, hour=hour, minute=minute,
                day_of_week=ex.day_of_week(state),
                sim_time=sim_time,
                zone_temps=zone_temps,
                zone_rh=zone_rh,
                outdoor_temp=outdoor,
                occupancy=occupancy,
                occupied=occupancy > 0.05,
                elec_j=elec_j,
                hvac_j=hvac_j,
                cumulative_kwh=self.cumulative_j * cfg.J_TO_KWH,
                cumulative_hvac_kwh=self.cumulative_hvac_j * cfg.J_TO_KWH,
                step=self.step,
                current_policy=self.current_policy.to_dict(),
            )

            row = {
                "step": self.step,
                "sim_time": sim_time,
                "outdoor_temp": outdoor,
                "zone_temps": zone_temps,
                "zone_rh": zone_rh,
                "occupancy": occupancy,
                "elec_j": elec_j,
                "hvac_j": hvac_j,
                "cumulative_kwh": round(bstate.cumulative_kwh, 4),
                "cumulative_hvac_kwh": round(bstate.cumulative_hvac_kwh, 4),
                "policy_heating": self.current_policy.heating_sp,
                "policy_cooling": self.current_policy.cooling_sp,
                "applied_heating": self.applied_setpoints[0],
                "applied_cooling": self.applied_setpoints[1],
                "policy_source": self.current_policy.source,
                "actuating": self.actuates,
            }

            # Publish before deciding so any tool call the controller makes
            # during decide() sees this timestep's readings.
            self.store.publish(bstate, self.current_policy)

            new_policy = self.controller.decide(bstate)

            # A controller may instead have written setpoints through the MCP
            # set_setpoints tool; pick those up here.
            if new_policy is None:
                pending = self.store.take_pending_policy()
                if pending:
                    new_policy = Policy(
                        heating_sp=pending.get("heating_c"),
                        cooling_sp=pending.get("cooling_c"),
                        source="mcp",
                        reason=pending.get("reason", ""),
                    )

            if new_policy is not None:
                clamped, violations = clamp_policy(new_policy)
                clamped.valid_from = sim_time
                self.current_policy = clamped
                if violations:
                    self.violations.extend(
                        f"{sim_time} {v}" for v in violations)
                row["decision"] = {
                    "heating": clamped.heating_sp,
                    "cooling": clamped.cooling_sp,
                    "source": clamped.source,
                    "reason": clamped.reason,
                    "violations": violations,
                }

            self.rows.append(row)
            if self._log_file:
                self._log_file.write(json.dumps(row) + "\n")

            if self.verbose and self.step % 96 == 0:
                print(f"[runner] {sim_time}  outdoor={outdoor:5.1f}C  "
                      f"mean_zone={sum(zone_temps.values()) / len(zone_temps):5.1f}C  "
                      f"cum={bstate.cumulative_kwh:7.1f} kWh")

        except Exception as exc:
            print(f"[runner] sensing error: {exc}", file=sys.stderr)

    def _maybe_dump_api_data(self) -> None:
        """Dump the full API catalogue once -- the ground truth for names."""
        if self._dumped or not self.dump_api_data:
            return
        data = self.api.exchange.list_available_api_data_csv(self.state)
        path = self.outdir / "available_api_data.csv"
        path.write_bytes(data)
        self._dumped = True
        if self.verbose:
            print(f"[runner] wrote API catalogue -> {path} ({len(data)} bytes)")

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self) -> int:
        api, st = self.api, self.state

        # Requesting variables before the run guarantees they exist in the
        # exchange even if the IDF's Output:Variable set changes.
        for zone in cfg.ZONES:
            api.exchange.request_variable(st, "Zone Mean Air Temperature", zone)
            api.exchange.request_variable(st, "Zone Air Relative Humidity", zone)
        api.exchange.request_variable(
            st, "Site Outdoor Air Drybulb Temperature", "Environment")
        api.exchange.request_variable(st, "Schedule Value", cfg.OCCUPANCY_SCHEDULE)

        api.runtime.callback_begin_system_timestep_before_predictor(
            st, self._on_begin_timestep)
        api.runtime.callback_end_zone_timestep_after_zone_reporting(
            st, self._on_end_timestep)

        self._log_file = open(self.log_path, "w", encoding="utf-8")
        started = time.time()
        try:
            exit_code = api.runtime.run_energyplus(st, [
                "-w", str(self.epw),
                "-d", str(self.outdir),
                "-r",
                str(self.idf),
            ])
        finally:
            self._log_file.close()
            self._log_file = None
            api.state_manager.reset_state(st)

        self.wall_clock = time.time() - started
        if self.verbose:
            print(f"[runner] mode={self.controller.name} exit={exit_code} "
                  f"steps={self.step} "
                  f"total={self.cumulative_j * cfg.J_TO_KWH:.1f} kWh "
                  f"wall={self.wall_clock:.1f}s "
                  f"violations={len(self.violations)}")
        return exit_code


if __name__ == "__main__":
    # Smoke test: baseline read-only pass that also dumps the API catalogue.
    runner = EnergyPlusRunner(
        controller=BaselineController(),
        outdir=cfg.OUT / "sensor_check",
        dump_api_data=True,
    )
    code = runner.run()
    print(f"exit={code} rows={len(runner.rows)}")
    if runner.rows:
        first, last = runner.rows[0], runner.rows[-1]
        print("first:", json.dumps(first["zone_temps"]))
        print("last :", json.dumps(last["zone_temps"]))
    sys.exit(code)
