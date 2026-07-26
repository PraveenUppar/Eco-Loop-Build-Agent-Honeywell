"""Prove the actuators are really connected. Run: python src/test_actuation.py

If setpoints are wired correctly, a loose deadband must LOWER facility energy
and a tight one must RAISE it, both relative to the untouched baseline. A
result that moves in only one direction usually means the actuator is silently
not being applied.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from controllers import FixedController, RuleBasedController
from eplus_runner import BaselineController, EnergyPlusRunner

CASES = [
    ("baseline", BaselineController()),
    ("loose  18.0/27.0", FixedController(18.0, 27.0, "loose")),
    ("tight  22.0/23.0", FixedController(22.0, 23.0, "tight")),
    ("rulebased", RuleBasedController()),
]

results = {}
for label, controller in CASES:
    runner = EnergyPlusRunner(
        controller=controller,
        outdir=cfg.OUT / f"actuation_{controller.name}",
        verbose=False,
    )
    code = runner.run()
    kwh = runner.cumulative_j * cfg.J_TO_KWH
    hvac = runner.cumulative_hvac_j * cfg.J_TO_KWH
    results[label] = (kwh, hvac, code, runner.step)
    print(f"{label:20} exit={code} steps={runner.step:4d} "
          f"facility={kwh:8.2f} kWh   hvac={hvac:7.2f} kWh")

base_kwh = results["baseline"][0]
print(f"\nbaseline = {base_kwh:.2f} kWh")
for label, (kwh, hvac, _, _) in results.items():
    if label == "baseline":
        continue
    delta = (kwh - base_kwh) / base_kwh * 100
    print(f"  {label:20} {kwh:8.2f} kWh   {delta:+6.2f}% vs baseline")

loose = results["loose  18.0/27.0"][0]
tight = results["tight  22.0/23.0"][0]

print()
ok_loose = loose < base_kwh
ok_tight = tight > base_kwh
print(f"[{'PASS' if ok_loose else 'FAIL'}] loose deadband reduces energy "
      f"({loose:.1f} < {base_kwh:.1f})")
print(f"[{'PASS' if ok_tight else 'FAIL'}] tight deadband increases energy "
      f"({tight:.1f} > {base_kwh:.1f})")

if not (ok_loose and ok_tight):
    print("\nActuation is NOT bidirectional -- do not proceed to the agent.")
    sys.exit(1)
print("\nactuation confirmed in both directions")
