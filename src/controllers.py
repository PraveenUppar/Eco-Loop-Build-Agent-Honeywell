"""Deterministic controllers: the inner loop and the safety net.

These run without any LLM involvement. `RuleBasedController` is both a
comparison arm in the experiments and the fallback the agent degrades to when
the model is unreachable or keeps producing invalid policies.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from policy import Policy


class FixedController:
    """Holds one setpoint pair for the whole run.

    Used to prove the actuators are genuinely connected: a loose setpoint must
    lower energy and a tight one must raise it, relative to baseline.
    """

    actuates = True
    # Opt out of the comfort guard: this controller exists to prove the full
    # actuator range moves energy, which the guard would mask.
    enforce_comfort = False

    def __init__(self, heating: float, cooling: float, label: str = "fixed"):
        self.heating = heating
        self.cooling = cooling
        self.name = label
        self._sent = False

    def decide(self, state) -> Policy | None:
        if self._sent:
            return None
        self._sent = True
        return Policy(
            heating_sp=self.heating,
            cooling_sp=self.cooling,
            source=self.name,
            reason=f"fixed {self.heating}/{self.cooling} for entire run",
        )


class RuleBasedController:
    """Occupancy-scheduled deadband with night/weekend setback.

    Baseline occupied deadband is 22.2-23.9 C, only 1.7 C wide, which makes the
    VAV system work hard to hold a narrow band. Widening it to 20.0-24.5 C stays
    inside the reported comfort band [20, 25] while cutting cooling demand.
    """

    name = "rulebased"
    actuates = True

    OCCUPIED = (20.0, 24.5)
    UNOCCUPIED = (18.0, 27.0)

    def __init__(self):
        self._last: tuple[float, float] | None = None

    def target_for(self, state) -> tuple[float, float]:
        """The setpoints this policy wants right now, always.

        Kept separate from `decide` because `decide` suppresses unchanged
        policies; callers using this as a fallback need the actual target, not
        None.
        """
        return self.OCCUPIED if state.occupied else self.UNOCCUPIED

    def decide(self, state) -> Policy | None:
        target = self.target_for(state)
        if target == self._last:
            return None                      # no change, avoid log churn
        self._last = target

        heating, cooling = target
        mode = "occupied" if state.occupied else "setback"
        return Policy(
            heating_sp=heating,
            cooling_sp=cooling,
            source="rulebased",
            reason=f"{mode} deadband {heating}-{cooling} C",
        )
