"""
Fanger Predicted Mean Vote (PMV) and Predicted Percentage Dissatisfied (PPD).

The brief asks for PMV specifically, and the earlier code reported a linear
stand-in - `(zone_temp - 22) / 2` - which is not PMV in any sense. It ignores
radiant temperature, humidity, clothing and metabolic rate, all four of which
move real PMV substantially, and it cannot distinguish a 21C room with cold
walls from a 21C room without them. Reporting that number as "PMV" against a
+-0.5 ASHRAE threshold would be claiming a comfort result the model never
computed.

This is the actual ISO 7730 / ASHRAE 55 Fanger model, implemented directly
rather than pulled in as a dependency - it is one well-specified function and
the iteration below is the standard one from the ISO 7730 appendix.

Requires two extra EnergyPlus output variables per zone (mean radiant
temperature and relative humidity), which `run_loop.py` requests up front.
"""

from __future__ import annotations

import math

# Occupied-hours assumptions for this building. Stated here rather than
# buried at the call site because PMV is only interpretable alongside them.
#   met  - metabolic rate, 1.2 met = seated light office work (ISO 7730 A.2)
#   clo  - clothing insulation, 1.0 clo = typical winter indoor dress
#   vel  - relative air velocity, m/s, still air in a mixed-mode office
DEFAULT_MET = 1.2
DEFAULT_CLO = 1.0
DEFAULT_AIR_VELOCITY = 0.1

# ASHRAE 55 recommends |PMV| < 0.5 for a general comfort target, which
# corresponds to PPD < 10%. This is the threshold the dashboard judges
# occupied timesteps against.
PMV_COMFORT_LIMIT = 0.5


def fanger_pmv(air_temp_c: float, radiant_temp_c: float, rel_humidity_pct: float,
               air_velocity: float = DEFAULT_AIR_VELOCITY,
               met: float = DEFAULT_MET, clo: float = DEFAULT_CLO) -> float:
    """
    Predicted Mean Vote on the ASHRAE thermal sensation scale.

    -3 cold, -2 cool, -1 slightly cool, 0 neutral, +1 slightly warm,
    +2 warm, +3 hot.
    """
    # Water vapour partial pressure, Pa.
    pa = rel_humidity_pct * 10.0 * math.exp(16.6536 - 4030.183 / (air_temp_c + 235.0))

    icl = 0.155 * clo          # clothing insulation, m2K/W
    m = met * 58.15            # metabolic rate, W/m2
    mw = m                     # external work is zero for office activity

    # Clothing area factor.
    fcl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl

    hcf = 12.1 * math.sqrt(air_velocity)   # forced convection coefficient
    taa = air_temp_c + 273.0
    tra = radiant_temp_c + 273.0

    # Clothing surface temperature, solved iteratively (ISO 7730 appendix D).
    tcla = taa + (35.5 - air_temp_c) / (3.05 * 0.001 * (5733.0 - 6.99 * mw - pa))
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4

    xn = tcla / 100.0
    xf = xn
    hc = hcf
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25   # natural convection
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= 0.00015:
            break
    tcl = 100.0 * xn - 273.0

    # Heat loss components, W/m2.
    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)          # skin diffusion
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0        # sweating
    hl3 = 1.7 * 0.00001 * m * (5867.0 - pa)                 # latent respiration
    hl4 = 0.0014 * m * (34.0 - air_temp_c)                  # dry respiration
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)       # radiation
    hl6 = fcl * hc * (tcl - air_temp_c)                     # convection

    # Thermal sensation transfer coefficient.
    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    return ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)


def temp_for_pmv(target_pmv: float, radiant_temp_c: float, rel_humidity_pct: float,
                 air_velocity: float = DEFAULT_AIR_VELOCITY,
                 met: float = DEFAULT_MET, clo: float = DEFAULT_CLO) -> float:
    """
    The air temperature that delivers `target_pmv` under current conditions.

    This is what lets the controller aim at a comfort *outcome* instead of a
    fixed thermostat number, and in this building that distinction is worth
    real energy. PMV depends on humidity, and this model's winter indoor
    relative humidity ranges from 4% to 41%. At 24% RH a 21.0C room sits at
    PMV -0.27; in drier air the same 21.0C is materially colder to occupants.
    A fixed setpoint has to be chosen for the worst case and then overheats
    the building the rest of the time - which is exactly what the model's own
    22.2C schedule does.

    Solved by bisection because PMV is monotonically increasing in air
    temperature but has no closed-form inverse (the clothing surface
    temperature is itself an iterative solve).

    Radiant temperature is held at its currently measured value rather than
    predicted forward. That is an approximation - surfaces do warm as the air
    warms - but the loop re-solves this every timestep, so the residual is
    corrected on the next step rather than accumulating.
    """
    lo, hi = 10.0, 32.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if fanger_pmv(mid, radiant_temp_c, rel_humidity_pct,
                      air_velocity, met, clo) < target_pmv:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ppd(pmv: float) -> float:
    """Predicted Percentage Dissatisfied, %. Bottoms out at 5% by construction."""
    return 100.0 - 95.0 * math.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)


def is_comfortable(pmv: float, limit: float = PMV_COMFORT_LIMIT) -> bool:
    return abs(pmv) <= limit


if __name__ == "__main__":
    # Sanity anchors against ISO 7730 Table D.1-style expectations: a neutral
    # office should land near 0, and the sign should follow temperature.
    for ta in (18.0, 20.0, 21.0, 22.0, 23.0, 24.0, 26.0):
        p = fanger_pmv(ta, ta, 40.0)
        print(f"  ta={ta:>4.1f}C  tr={ta:>4.1f}C  rh=40%  ->  PMV={p:+.2f}  PPD={ppd(p):5.1f}%")
