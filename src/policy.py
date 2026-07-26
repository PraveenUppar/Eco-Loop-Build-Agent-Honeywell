"""Setpoint policy + the safety clamp that every policy must pass through.

Nothing reaches an EnergyPlus actuator without going through `clamp_policy`.
That is deliberate: the supervisory LLM is a 3B local model, so the validation
layer is load-bearing rather than decorative. A malformed or unsafe suggestion
degrades to a clamped or fallback policy instead of cooking the occupants.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


@dataclass
class Policy:
    """A setpoint policy that holds until the supervisor replaces it."""

    heating_sp: float
    cooling_sp: float
    # Optional per-zone overrides: {zone_name: (heating, cooling)}
    per_zone: dict[str, tuple[float, float]] = field(default_factory=dict)
    source: str = "fallback"     # baseline | rulebased | llm | fallback | clamped
    reason: str = ""
    valid_from: str = ""         # sim timestamp the policy took effect

    def for_zone(self, zone: str) -> tuple[float, float]:
        return self.per_zone.get(zone, (self.heating_sp, self.cooling_sp))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fallback_policy(reason: str = "fallback") -> Policy:
    return Policy(
        heating_sp=cfg.FALLBACK_HEATING,
        cooling_sp=cfg.FALLBACK_COOLING,
        source="fallback",
        reason=reason,
    )


def _coerce(value: Any) -> float | None:
    """Accept int/float/numeric-string; reject NaN, inf, None and junk."""
    if isinstance(value, bool):        # bool is an int subclass -- reject it
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _clip(value: float, lo: float, hi: float) -> tuple[float, bool]:
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


def clamp_pair(heating: Any, cooling: Any,
               label: str = "") -> tuple[float, float, list[str]]:
    """Clamp one heating/cooling pair into the safety envelope.

    Returns (heating, cooling, violations). Any input that cannot be coerced to
    a finite number falls back to the configured safe setpoint.
    """
    violations: list[str] = []
    prefix = f"{label}: " if label else ""

    h = _coerce(heating)
    if h is None:
        violations.append(f"{prefix}heating_sp not numeric ({heating!r}) -> fallback")
        h = cfg.FALLBACK_HEATING

    c = _coerce(cooling)
    if c is None:
        violations.append(f"{prefix}cooling_sp not numeric ({cooling!r}) -> fallback")
        c = cfg.FALLBACK_COOLING

    h, clipped = _clip(h, cfg.HEATING_MIN, cfg.HEATING_MAX)
    if clipped:
        violations.append(
            f"{prefix}heating_sp {heating} outside "
            f"[{cfg.HEATING_MIN}, {cfg.HEATING_MAX}] -> {h}")

    c, clipped = _clip(c, cfg.COOLING_MIN, cfg.COOLING_MAX)
    if clipped:
        violations.append(
            f"{prefix}cooling_sp {cooling} outside "
            f"[{cfg.COOLING_MIN}, {cfg.COOLING_MAX}] -> {c}")

    # Enforce the minimum deadband by widening upward -- raising the cooling
    # setpoint is the energy-saving direction, and HEATING_MAX (22) plus the
    # deadband (1.5) never exceeds COOLING_MAX (27), so this always resolves.
    if c - h < cfg.MIN_DEADBAND:
        widened = min(h + cfg.MIN_DEADBAND, cfg.COOLING_MAX)
        if widened - h < cfg.MIN_DEADBAND:          # defensive, unreachable today
            h = widened - cfg.MIN_DEADBAND
        violations.append(
            f"{prefix}deadband {c - h:.2f} < {cfg.MIN_DEADBAND} -> cooling {widened}")
        c = widened

    return round(h, 2), round(c, 2), violations


def clamp_policy(policy: Policy) -> tuple[Policy, list[str]]:
    """Clamp a whole policy, including any per-zone overrides."""
    h, c, violations = clamp_pair(policy.heating_sp, policy.cooling_sp)

    per_zone: dict[str, tuple[float, float]] = {}
    for zone, pair in (policy.per_zone or {}).items():
        if zone not in cfg.ZONES:
            violations.append(f"unknown zone {zone!r} dropped")
            continue
        try:
            zh, zc = pair
        except (TypeError, ValueError):
            violations.append(f"zone {zone} override malformed ({pair!r}) dropped")
            continue
        zh, zc, zv = clamp_pair(zh, zc, label=zone)
        violations.extend(zv)
        per_zone[zone] = (zh, zc)

    clamped = Policy(
        heating_sp=h,
        cooling_sp=c,
        per_zone=per_zone,
        source=policy.source if not violations else f"{policy.source}+clamped",
        reason=policy.reason,
        valid_from=policy.valid_from,
    )
    return clamped, violations


def operating_envelope(heating: float, cooling: float,
                       occupied: bool) -> tuple[float, float, bool]:
    """Hold setpoints inside the envelope appropriate to current occupancy.

    This is the deterministic interlock a real BMS keeps underneath its
    supervisor, and it runs every timestep regardless of what the supervisor
    asked for four hours ago. It is symmetric:

      OCCUPIED -- pull setpoints into the comfort band. The safety clamp allows
        cooling up to 27 C, which is right for an empty building at night but
        would leave occupants too warm. Without this, a setback chosen at 04:00
        would still be in force when people arrive at 06:00.

      EMPTY -- push setpoints out to the setback envelope. Nobody can be
        uncomfortable, so any equipment running is pure waste; this guarantees
        the deadband is wide enough that neither heating nor cooling engages.
    """
    if occupied:
        guarded_h = min(max(heating, cfg.COMFORT_LOW), cfg.HEATING_MAX)
        guarded_c = min(cooling, cfg.COMFORT_HIGH)
        if guarded_c - guarded_h < cfg.MIN_DEADBAND:
            guarded_h = guarded_c - cfg.MIN_DEADBAND
    else:
        guarded_h = min(heating, cfg.SETBACK_HEATING_MAX)
        guarded_c = max(cooling, cfg.SETBACK_COOLING_MIN)

    adjusted = (guarded_h != heating) or (guarded_c != cooling)
    return round(guarded_h, 2), round(guarded_c, 2), adjusted


def policy_from_llm(payload: dict[str, Any]) -> tuple[Policy, list[str]]:
    """Build a Policy from a parsed LLM JSON response, then clamp it."""
    per_zone_raw = payload.get("per_zone_overrides") or {}
    per_zone: dict[str, tuple[float, float]] = {}
    if isinstance(per_zone_raw, dict):
        for zone, override in per_zone_raw.items():
            if isinstance(override, dict):
                per_zone[zone] = (override.get("heating_c"),
                                  override.get("cooling_c"))
            elif isinstance(override, (list, tuple)) and len(override) == 2:
                per_zone[zone] = (override[0], override[1])

    raw = Policy(
        heating_sp=payload.get("heating_c"),
        cooling_sp=payload.get("cooling_c"),
        per_zone=per_zone,
        source="llm",
        reason=str(payload.get("reason", ""))[:300],
    )
    return clamp_policy(raw)
