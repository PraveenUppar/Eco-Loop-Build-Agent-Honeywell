"""Unit tests for the safety clamp. Run: python src/test_clamp.py

The clamp is the only thing standing between a 3B model's output and a real
actuator, so it is tested against deliberately hostile input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from policy import Policy, clamp_pair, clamp_policy, policy_from_llm

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(f"{label}: {detail}")
    print(f"  [{status}] {label}" + (f"  -- {detail}" if detail else ""))


def in_envelope(h, c):
    return (cfg.HEATING_MIN <= h <= cfg.HEATING_MAX
            and cfg.COOLING_MIN <= c <= cfg.COOLING_MAX
            and c - h >= cfg.MIN_DEADBAND - 1e-9)


print("=== garbage input -> always lands inside the safety envelope ===")

GARBAGE = [
    ("wildly out of range", 99.0, -40.0),
    ("strings", "not-a-number", "23.5"),
    ("None", None, None),
    ("inverted setpoints", 26.0, 19.0),
    ("deadband too narrow", 22.0, 23.0),
    ("NaN / inf", float("nan"), float("inf")),
    ("booleans", True, False),
]

for label, h_in, c_in in GARBAGE:
    h, c, violations = clamp_pair(h_in, c_in)
    check(label, in_envelope(h, c), f"({h_in!r}, {c_in!r}) -> {h}/{c}, "
                                    f"{len(violations)} violation(s)")

print("\n=== valid input passes through untouched ===")
h, c, violations = clamp_pair(21.0, 25.0)
check("valid pair unchanged", (h, c) == (21.0, 25.0) and not violations,
      f"-> {h}/{c}, violations={violations}")

print("\n=== per-zone overrides ===")
p = Policy(heating_sp=21.0, cooling_sp=25.0,
           per_zone={"SPACE1-1": (30.0, 10.0), "NOT-A-ZONE": (21.0, 24.0)})
clamped, violations = clamp_policy(p)
check("bad zone override clamped",
      in_envelope(*clamped.per_zone["SPACE1-1"]),
      f"SPACE1-1 -> {clamped.per_zone['SPACE1-1']}")
check("unknown zone dropped", "NOT-A-ZONE" not in clamped.per_zone,
      f"per_zone keys = {list(clamped.per_zone)}")

print("\n=== malformed LLM payloads ===")
PAYLOADS = [
    ("empty dict", {}),
    ("wrong keys", {"temperature": 22, "mode": "eco"}),
    ("nested junk", {"heating_c": {"value": 21}, "cooling_c": [25]}),
    ("plausible", {"heating_c": 20.5, "cooling_c": 24.5, "reason": "mild day"}),
]
for label, payload in PAYLOADS:
    pol, violations = policy_from_llm(payload)
    check(label, in_envelope(pol.heating_sp, pol.cooling_sp),
          f"-> {pol.heating_sp}/{pol.cooling_sp}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("all clamp tests passed")
