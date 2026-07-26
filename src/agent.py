"""Supervisory LLM agent: sets setpoint policy every few simulated hours.

The model is a *supervisor*, not a per-timestep controller. It re-plans every
LLM_INTERVAL_HOURS of simulated time; the deterministic inner loop applies the
resulting policy at every 15-minute timestep. That is the only way to keep a
local model in the loop without the latency dominating the run, and it mirrors
how supervisory setpoint reset works in a real BMS.

The loop is: read state through MCP tools -> prompt the model -> write the
setpoints back through the set_setpoints tool, which validates them. A
rejection is fed back to the model as a correction prompt. If the model still
cannot produce a usable policy, the rule-based controller takes over for that
interval, so the building is never left without a policy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama

import config as cfg
from controllers import RuleBasedController
from mcp_bridge import MCPBridge
from policy import Policy

# A JSON schema handed to Ollama as a structured-output constraint. This does
# far more for reliability on a 3B model than asking politely in the prompt.
POLICY_SCHEMA = {
    "type": "object",
    "properties": {
        "heating_c": {"type": "number"},
        "cooling_c": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["heating_c", "cooling_c", "reason"],
}

SYSTEM_PROMPT = f"""You supervise the heating and cooling of a 5-zone office
building. Every 4 hours you choose two numbers. First understand what those
numbers actually do.

--- WHAT A SETPOINT IS ---
A setpoint is a threshold, not a target. Each zone has two of them:
  heating_c : if the room temperature falls BELOW this, the heater switches on.
  cooling_c : if the room temperature rises ABOVE this, the air conditioner
              switches on.
Between those two numbers, NOTHING runs. The room temperature simply floats
wherever the weather and the people in it put it. That in-between region is
called the DEADBAND, and it is free: no electricity is spent while the room
sits inside it.

--- WHICH DIRECTION SAVES ELECTRICITY ---
Because equipment only runs outside the deadband:
  RAISING cooling_c means the room must get hotter before the AC starts,
      so the AC runs FEWER hours. Higher cooling_c = LESS electricity.
  LOWERING heating_c means the room must get colder before the heater starts,
      so the heater runs FEWER hours. Lower heating_c = LESS electricity.
  A WIDER deadband (big gap between the two) = LESS electricity, always.
  A NARROW deadband forces the equipment to fight to hold a tight range, and
      heating and cooling can even work against each other. That is the most
      expensive situation possible.

--- THE MISTAKE TO AVOID ---
When it is hot outside, it feels natural to LOWER cooling_c to "fight" the
heat. This is backwards. Lowering cooling_c does not remove the heat for free,
it just makes the AC start earlier and run longer, which COSTS more
electricity. The correct response to hot weather is to RAISE cooling_c to the
highest value still allowed, so the building tolerates a little more warmth
instead of paying to fight it.

--- OCCUPIED VS EMPTY ---
The building is OCCUPIED on weekdays roughly 06:00-20:00, and EMPTY at night
and at weekends.
  EMPTY: nobody can be uncomfortable, so comfort does not constrain you. Widen
      the deadband as far as the limits allow ({cfg.HEATING_MIN} and
      {cfg.COOLING_MAX}). This is where the largest, safest savings are.
  OCCUPIED: people are present, so the room must stay inside the comfort band
      {cfg.COMFORT_LOW}-{cfg.COMFORT_HIGH} C. Within that band you still want
      the widest deadband you can get, which means pushing cooling_c UP toward
      {cfg.COMFORT_HIGH}, not down.

--- WHY PEAK MATTERS ---
Sudden large drops in cooling_c make the AC start at full power all at once,
creating an expensive spike in demand. Prefer smooth, gradual changes over
sharp swings.

--- HARD LIMITS (a reply outside these is rejected) ---
  heating_c between {cfg.HEATING_MIN} and {cfg.HEATING_MAX}
  cooling_c between {cfg.COOLING_MIN} and {cfg.COOLING_MAX}
  cooling_c minus heating_c at least {cfg.MIN_DEADBAND}

Your goal: the least electricity possible, without pushing an occupied room
outside {cfg.COMFORT_LOW}-{cfg.COMFORT_HIGH} C.

Reply with JSON only: {{"heating_c": <number>, "cooling_c": <number>, "reason": "<15 words max>"}}"""


class AgentController:
    """LLM supervisor with tool-validated self-correction and a safe fallback."""

    name = "agent"
    actuates = True

    def __init__(self, bridge: MCPBridge, outdir: Path,
                 model: str = cfg.OLLAMA_MODEL,
                 interval_hours: float = cfg.LLM_INTERVAL_HOURS,
                 use_cache: bool = True, verbose: bool = True,
                 trace: bool = False):
        self.bridge = bridge
        self.model = model
        # Prints every stage of one control cycle. Off by default; the demo
        # recording needs to show data leaving EnergyPlus and control actions
        # coming back, which the ordinary one-line log does not make visible.
        self.trace = trace
        self.interval_steps = max(1, int(interval_hours * cfg.TIMESTEPS_PER_HOUR))
        self.use_cache = use_cache
        self.verbose = verbose

        self.client = ollama.Client(timeout=cfg.LLM_TIMEOUT_S)
        self.fallback = RuleBasedController()

        self._next_step = 1
        self._cache: dict[tuple, dict[str, float]] = {}
        self._history: list[dict[str, Any]] = []

        outdir.mkdir(parents=True, exist_ok=True)
        self.log_path = outdir / "llm_calls.jsonl"
        self._log = open(self.log_path, "w", encoding="utf-8")

        # Telemetry reported in the dashboard and the write-up.
        self.stats = {
            "calls": 0, "cache_hits": 0, "retries": 0,
            "invalid": 0, "fallbacks": 0, "tool_rejections": 0,
            "latencies": [],
        }

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._log:
            self._log.close()
            self._log = None

    def _record(self, entry: dict[str, Any]) -> None:
        if self._log:
            self._log.write(json.dumps(entry) + "\n")
            self._log.flush()

    # ------------------------------------------------------------------
    def _trace_sense(self, state) -> None:
        """Show the sensor readings arriving from the running simulation."""
        temps = "  ".join(f"{z.split('-')[0]} {v:5.1f}"
                          for z, v in state.zone_temps.items())
        print(f"\n{'=' * 74}")
        print(f"[SENSE] EnergyPlus -> agent   sim time {state.sim_time}")
        print(f"        {temps}  (degC)")
        print(f"        outdoor {state.outdoor_temp:5.1f} C | "
              f"{'OCCUPIED' if state.occupied else 'EMPTY':8} | "
              f"{state.cumulative_kwh:8.1f} kWh used so far", flush=True)

    def _cache_key(self, state) -> tuple:
        """Bucket the state so near-identical situations reuse a decision."""
        mean_temp = sum(state.zone_temps.values()) / len(state.zone_temps)
        return (
            state.hour // 4,                 # 4-hour block of the day
            state.occupied,
            round(state.outdoor_temp),        # nearest degree
            round(mean_temp),
        )

    def _build_prompt(self, state, energy: dict[str, Any]) -> str:
        temps = state.zone_temps
        hottest = max(temps, key=temps.get)
        coolest = min(temps, key=temps.get)
        mean_temp = sum(temps.values()) / len(temps)
        pol = state.current_policy

        # State the range that is actually valid right now. The occupied
        # comfort limit is the constraint a small model most often violates,
        # and repeating it in the user turn cuts rejected replies sharply.
        if state.occupied:
            allowed = (f"ALLOWED NOW (people present): "
                       f"heating_c {cfg.COMFORT_LOW}-{cfg.HEATING_MAX}, "
                       f"cooling_c {cfg.COOLING_MIN}-{cfg.COMFORT_HIGH}. "
                       f"The top of the cooling range is the cheap end; "
                       f"the bottom is the expensive end. Above "
                       f"{cfg.COMFORT_HIGH} breaks comfort and is rejected.")
        else:
            allowed = (f"ALLOWED NOW (building empty): "
                       f"heating_c {cfg.HEATING_MIN}-{cfg.HEATING_MAX}, "
                       f"cooling_c {cfg.COOLING_MIN}-{cfg.COOLING_MAX}. "
                       f"Nobody can be uncomfortable, so the widest deadband "
                       f"is also the cheapest.")

        lines = [
            f"TIME {state.sim_time} ({'OCCUPIED' if state.occupied else 'UNOCCUPIED'})",
            allowed,
            f"OUTDOOR {state.outdoor_temp:.1f} C",
            f"ZONES avg {mean_temp:.1f} C "
            f"(coolest {coolest} {temps[coolest]:.1f}, "
            f"hottest {hottest} {temps[hottest]:.1f})",
            f"CURRENT SETPOINTS heating {pol.get('heating_sp')} / "
            f"cooling {pol.get('cooling_sp')}",
        ]

        if energy.get("ok"):
            lines.append(
                f"LAST {energy.get('window_hours')}H {energy.get('kwh')} kWh "
                f"(peak {energy.get('peak_kw')} kW)")
            if energy.get("pct_vs_baseline") is not None:
                lines.append(
                    f"VS BASELINE {energy['pct_vs_baseline']:+.2f}% so far")

        if self._history:
            lines.append("")
            lines.append("RECENT DECISIONS (most recent last)")
            lines.append("  time        htg   clg   kWh/4h  avgC")
            for h in self._history[-6:]:
                lines.append(
                    f"  {h['sim_time']}  {h['heating']:<5.1f} {h['cooling']:<5.1f} "
                    f"{h['kwh']:<7.1f} {h['mean_temp']:.1f}")

        lines.append("")
        lines.append("Choose setpoints for the next 4 hours.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _ask_model(self, messages: list[dict[str, str]]) -> tuple[dict | None, float, str]:
        """One model call. Returns (parsed_json_or_None, latency_s, raw_text)."""
        started = time.time()
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=POLICY_SCHEMA,
                options={"temperature": 0.2, "num_predict": 160},
            )
            raw = response["message"]["content"]
        except Exception as exc:                        # noqa: BLE001
            return None, time.time() - started, f"<error: {exc}>"

        latency = time.time() - started
        try:
            return json.loads(raw), latency, raw
        except json.JSONDecodeError:
            return None, latency, raw

    def _apply_via_tool(self, parsed: dict, state) -> dict[str, Any]:
        """Write setpoints through MCP so the tool layer validates them."""
        return self.bridge.call("set_setpoints", {
            "heating_c": parsed.get("heating_c"),
            "cooling_c": parsed.get("cooling_c"),
            "zone": "all",
            "reason": str(parsed.get("reason", ""))[:200],
        })

    # ------------------------------------------------------------------
    def decide(self, state) -> Policy | None:
        if state.step < self._next_step:
            return None
        self._next_step = state.step + self.interval_steps

        key = self._cache_key(state)
        if self.use_cache and key in self._cache:
            cached = self._cache[key]
            self.stats["cache_hits"] += 1
            self._record({
                "sim_time": state.sim_time, "step": state.step,
                "cache_hit": True, "key": list(key), "applied": cached,
                "latency_s": 0.0,
            })
            self._remember(state, cached["heating_c"], cached["cooling_c"])
            if self.trace:
                self._trace_sense(state)
                print(f"[CACHE] state already seen -> reuse "
                      f"{cached['heating_c']}/{cached['cooling_c']} C, "
                      f"no LLM call", flush=True)
            return Policy(
                heating_sp=cached["heating_c"], cooling_sp=cached["cooling_c"],
                source="llm-cache", reason=cached.get("reason", "cached decision"),
            )

        if self.trace:
            self._trace_sense(state)

        energy = self.bridge.call("get_energy_summary",
                                  {"window_hours": cfg.LLM_INTERVAL_HOURS})
        if self.trace:
            print(f"[TOOL ] MCP get_energy_summary -> {energy.get('kwh')} kWh "
                  f"over {cfg.LLM_INTERVAL_HOURS} h, peak "
                  f"{energy.get('peak_kw')} kW", flush=True)

        prompt = self._build_prompt(state, energy)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        retries = 0
        total_latency = 0.0
        raw_last = ""

        for attempt in range(cfg.LLM_MAX_RETRIES + 1):
            parsed, latency, raw = self._ask_model(messages)
            total_latency += latency
            raw_last = raw
            self.stats["calls"] += 1
            self.stats["latencies"].append(round(latency, 3))

            problem = None
            tool_result: dict[str, Any] = {}

            if self.trace:
                shown = (json.dumps(parsed) if parsed is not None
                         else raw[:90].replace("\n", " "))
                print(f"[LLM  ] {self.model} -> {shown}  ({latency:.1f}s"
                      + (f", retry {attempt}" if attempt else "") + ")",
                      flush=True)

            if parsed is None:
                problem = "response was not valid JSON"
                self.stats["invalid"] += 1
            else:
                tool_result = self._apply_via_tool(parsed, state)
                if not tool_result.get("ok"):
                    problem = "; ".join(tool_result.get("violations")
                                        or [str(tool_result.get("error"))])
                    self.stats["tool_rejections"] += 1

            if self.trace:
                if problem:
                    print(f"[VALID] REJECTED by tool layer: {problem[:100]}",
                          flush=True)
                else:
                    a = tool_result["applied"]
                    print(f"[VALID] accepted -> heating {a['heating_c']} C / "
                          f"cooling {a['cooling_c']} C", flush=True)

            if problem is None:
                applied = tool_result["applied"]
                self._record({
                    "sim_time": state.sim_time, "step": state.step,
                    "cache_hit": False, "retries": retries,
                    "latency_s": round(total_latency, 3),
                    "prompt_chars": len(prompt) + len(SYSTEM_PROMPT),
                    "approx_prompt_tokens": (len(prompt) + len(SYSTEM_PROMPT)) // 4,
                    "response": parsed, "applied": applied, "valid": True,
                })
                if self.use_cache:
                    self._cache[key] = {
                        "heating_c": applied["heating_c"],
                        "cooling_c": applied["cooling_c"],
                        "reason": str(parsed.get("reason", ""))[:120],
                    }
                self._remember(state, applied["heating_c"], applied["cooling_c"])
                if self.trace:
                    print(f"[ACT  ] agent -> EnergyPlus: setpoint schedules "
                          f"overwritten, applied every 15 min until next "
                          f"decision", flush=True)
                elif self.verbose:
                    print(f"[agent] {state.sim_time} -> "
                          f"{applied['heating_c']}/{applied['cooling_c']} "
                          f"({latency:.1f}s, retries={retries}) "
                          f"{str(parsed.get('reason',''))[:50]}")
                # The tool already wrote the pending policy; returning None
                # lets the runner consume it, keeping MCP the single path.
                return None

            # Self-correction: hand the exact failure back to the model.
            if attempt < cfg.LLM_MAX_RETRIES:
                retries += 1
                self.stats["retries"] += 1
                messages.append({"role": "assistant", "content": raw[:400]})
                messages.append({
                    "role": "user",
                    "content": (f"That was rejected: {problem}. "
                                f"Reply with corrected JSON only, obeying "
                                f"heating_c {cfg.HEATING_MIN}-{cfg.HEATING_MAX}, "
                                f"cooling_c {cfg.COOLING_MIN}-{cfg.COOLING_MAX}, "
                                f"gap >= {cfg.MIN_DEADBAND}."),
                })

        # Both attempts failed -- hand this interval to the rule-based policy.
        # Use target_for rather than decide(): decide() suppresses unchanged
        # policies and would silently degrade the fallback to a constant.
        self.stats["fallbacks"] += 1
        heating, cooling = self.fallback.target_for(state)
        fallback = Policy(
            heating_sp=heating, cooling_sp=cooling,
            source="fallback", reason="LLM produced no usable policy")
        self._record({
            "sim_time": state.sim_time, "step": state.step,
            "cache_hit": False, "retries": retries, "valid": False,
            "latency_s": round(total_latency, 3), "raw": raw_last[:400],
            "applied": {"heating_c": fallback.heating_sp,
                        "cooling_c": fallback.cooling_sp},
            "note": "fell back to rule-based policy",
        })
        if self.verbose:
            print(f"[agent] {state.sim_time} FALLBACK after {retries} retries")
        self._remember(state, fallback.heating_sp, fallback.cooling_sp)
        return fallback

    def _remember(self, state, heating: float, cooling: float) -> None:
        mean_temp = sum(state.zone_temps.values()) / len(state.zone_temps)
        self._history.append({
            "sim_time": state.sim_time,
            "heating": heating,
            "cooling": cooling,
            "kwh": round(state.cumulative_kwh, 1),
            "mean_temp": round(mean_temp, 1),
        })

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        lat = sorted(self.stats["latencies"])
        median = lat[len(lat) // 2] if lat else 0.0
        # "calls" counts every model invocation, retries included, so it is not
        # the decision count. One decision = one cache hit, or one call plus
        # however many retries it needed.
        decisions = (self.stats["calls"] - self.stats["retries"]
                     + self.stats["cache_hits"])
        return {
            "model": self.model,
            "supervisory_decisions": decisions,
            "llm_calls": self.stats["calls"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate_pct": round(
                100 * self.stats["cache_hits"] / max(1, decisions), 1),
            "retries": self.stats["retries"],
            "invalid_json": self.stats["invalid"],
            "tool_rejections": self.stats["tool_rejections"],
            "fallbacks": self.stats["fallbacks"],
            "median_latency_s": round(median, 2),
            "total_llm_seconds": round(sum(self.stats["latencies"]), 1),
        }
