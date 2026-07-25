"""
Time-varying electricity price and grid carbon intensity.

Why this exists: with a flat price and flat carbon factor, the optimal
control strategy is static, so a fixed setback schedule is already
near-optimal and there is nothing for an agent to reason about. Measured
directly - supervisor rules landed within +-1% of the fixed baseline
across every climate tested (see BUILD_LOG.md).

A varying signal changes the problem. The cheapest and cleanest kWh is
no longer "the one you don't use" but "the one you use at 3am instead of
6pm". That creates load shifting: heat the building while power is cheap
and clean, bank it in the thermal mass, then coast through the expensive
carbon-heavy peak. Comfort is unchanged; cost and emissions drop. A fixed
schedule cannot do this by construction - it has no notion of what hour
it is in price terms.

The brief asks for exactly this: "peak demand thresholds, and local
carbon grid intensity", and "the LLM calculates optimal Energy
Conservation Measures".

Curve shapes are representative of a grid with significant solar and
gas peaking plant, not measured data from any specific utility - they
are stated here rather than presented as real tariff data.
"""

from __future__ import annotations

# Time-of-use tariff, currency units per kWh.
#   off-peak overnight, mid-tier daytime, expensive evening peak
OFF_PEAK_PRICE = 0.08
MID_PEAK_PRICE = 0.15
ON_PEAK_PRICE = 0.32

# Grid carbon intensity, gCO2 per kWh.
#   dips midday when solar is abundant, spikes in the evening when solar
#   drops off and gas peakers come online
OVERNIGHT_CARBON = 250.0
SOLAR_CARBON = 120.0
SHOULDER_CARBON = 300.0
PEAK_CARBON = 480.0

# Evening peak window - the hours worth steering load away from.
PEAK_START_HOUR = 16
PEAK_END_HOUR = 21


def _hour(minute_of_day: int) -> float:
    return (minute_of_day % (24 * 60)) / 60.0


def price_per_kwh(minute_of_day: int) -> float:
    """Time-of-use electricity price at the given time of day."""
    hour = _hour(minute_of_day)
    if hour < 7 or hour >= 22:
        return OFF_PEAK_PRICE
    if PEAK_START_HOUR <= hour < PEAK_END_HOUR:
        return ON_PEAK_PRICE
    return MID_PEAK_PRICE


def carbon_intensity(minute_of_day: int) -> float:
    """Grid carbon intensity (gCO2/kWh) at the given time of day."""
    hour = _hour(minute_of_day)
    if PEAK_START_HOUR <= hour < PEAK_END_HOUR:
        return PEAK_CARBON
    if 10 <= hour < 15:
        return SOLAR_CARBON
    if hour < 6 or hour >= 22:
        return OVERNIGHT_CARBON
    return SHOULDER_CARBON


def is_peak(minute_of_day: int) -> bool:
    return PEAK_START_HOUR <= _hour(minute_of_day) < PEAK_END_HOUR


def minutes_until_peak(minute_of_day: int) -> int:
    """
    Minutes until the next on-peak window starts. Zero while already in
    it. This is the signal that makes pre-charging possible: it tells the
    controller how long it has to bank heat before power gets expensive.
    """
    if is_peak(minute_of_day):
        return 0
    minute_of_day %= 24 * 60
    peak_start = PEAK_START_HOUR * 60
    if minute_of_day >= peak_start:
        peak_start += 24 * 60
    return peak_start - minute_of_day


def forecast(minute_of_day: int, hours: int = 6, step_minutes: int = 60) -> list[dict]:
    """
    Upcoming price and carbon intensity, for an agent deciding whether to
    run now or wait. Returned as plain dicts so it serializes straight
    into a tool response.
    """
    out = []
    for i in range(hours):
        t = minute_of_day + i * step_minutes
        out.append({
            "in_minutes": i * step_minutes,
            "hour_of_day": int(_hour(t)),
            "price_per_kwh": price_per_kwh(t),
            "carbon_g_per_kwh": carbon_intensity(t),
            "is_peak": is_peak(t),
        })
    return out
