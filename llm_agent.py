"""
LLM agent orchestration: turns a zone state snapshot into a tool-calling
conversation with a local Ollama model, and executes whatever tools the
model chooses to call against a ToolExecutor.

Kept deliberately simple (direct tool-calling loop, not a full MCP server -
see PLAN.md's Hour 3-7 notes on why: fewer moving parts to debug under a
deadline). If you outgrow this, TOOL_SCHEMAS in tools.py is already in the
OpenAI/MCP-compatible function-calling shape, so swapping the transport
later is mostly a matter of changing how dispatch() gets called.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import ollama

from supervisor import BUILDING_MODES
from tools import BUILDING_TOOL_SCHEMAS, TOOL_SCHEMAS, ToolExecutor

DEFAULT_MODEL = "qwen2.5-coder:1.5b"

_TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}


def _coerce_args(raw_args) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return {}


def _find_json_objects(text: str) -> list[str]:
    """Scan text for top-level balanced-brace JSON object substrings."""
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:i + 1])
                    start = None
    return objects


def _parse_fallback_tool_calls(content: str) -> list[dict]:
    """
    Best-effort recovery for models that don't populate Ollama's structured
    tool_calls field but dump JSON-ish tool-call text into the message
    content instead - observed with qwen2.5-coder:1.5b, whose chat template
    expects <tool_call> tags the model doesn't reliably emit (see
    BUILD_LOG.md). Scans for {"name": ..., "arguments": ...} objects and
    normalizes arguments that arrive as null or a JSON-encoded string.
    """
    calls = []
    text = content.replace("```json", "").replace("```", "")
    for candidate in _find_json_objects(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if name not in _TOOL_NAMES:
            continue
        calls.append({"name": name, "arguments": _coerce_args(obj.get("arguments"))})
    return calls

SYSTEM_PROMPT = """You are the supervisory control agent for a building HVAC zone.

Your job each cycle:
1. Call get_zone_state. It returns occupied, minutes_until_occupancy_change
   (minutes until occupancy flips) and minutes_until_price_peak (minutes
   until electricity becomes expensive and carbon-heavy; 0 means now).
2. Work down this list. Stop at the FIRST line that is true and use its
   number. Do not read any further lines after one matches.

   A. occupied is false and minutes_until_occupancy_change is more
      than 120                                    -> 18.0
   B. minutes_until_price_peak is between 1 and 120  -> 23.5
   C. anything else                                  -> 20.5

   Line A means the building is empty and stays empty for a while, so let
   it go cold. Line B means expensive electricity is coming soon, so heat
   up now while it is still cheap and let the building coast through the
   peak. Line C is the normal comfortable setting.

   The answer is always exactly 18.0, 23.5 or 20.5 - never any other
   number, and never a value copied from zone_temp_c. The same line will
   match many cycles in a row; that is correct, just call
   set_zone_setpoint again with the same number.
3. Call set_zone_setpoint with that number. Call it exactly once.
4. Give one short sentence of reasoning for your decision.

Only use the tools provided. Do not invent tools or parameters. Make at
most one set_zone_setpoint call per cycle.
"""


# --------------------------------------------------------------------------
# Strategy agent (real EnergyPlus path).
#
# Two changes from the single-zone agent below, both forced by measurement
# rather than taste:
#
# 1. The action is a *named strategy* from a closed set, not a float. Asked
#    for a number, qwen2.5-coder:1.5b produced a usable one in 10 of 156
#    calls during a real EnergyPlus run - the other 146 cycles fell through
#    to a supervisor fallback, so the "agent" was contributing nothing while
#    appearing to run.
#
# 2. The final decision turn is emitted under Ollama's structured-output
#    grammar (`format=<json schema>`), which makes an unparseable or
#    out-of-set answer impossible rather than merely unlikely. Measured
#    across 75 synthetic building states: 75/75 responses valid, versus the
#    ~6% usable rate above.
#
# Information gathering still happens through ordinary tool calls, so the
# agent chooses what to look at; only the final action is constrained.
# --------------------------------------------------------------------------

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": list(BUILDING_MODES)},
        "reason": {"type": "string"},
    },
    "required": ["mode", "reason"],
}

STRATEGY_SYSTEM_PROMPT = """You are the supervisory control agent for a five-zone office building's heating.

First call get_building_state to see the building. You may also call
get_grid_forecast to see upcoming electricity prices and carbon intensity.

Then choose exactly one control strategy.

get_building_state gives you four conditions, each already worked out for
you as true or false: building_occupied, occupancy_imminent, peak_now,
peak_approaching. Read down this list and stop at the FIRST line whose
condition is true. Do not do any arithmetic.

1. peak_now is true AND building_occupied is true
      -> COAST
2. peak_approaching is true AND (building_occupied is true
   OR occupancy_imminent is true)
      -> PRECHARGE
3. building_occupied is false AND occupancy_imminent is false
      -> SETBACK
4. building_occupied is false AND occupancy_imminent is true
      -> PREHEAT
5. none of the above matched
      -> COMFORT

What the strategies mean physically:
  SETBACK   - building empty for a while yet, let it go cold and save energy.
  PREHEAT   - people arrive soon, warm the building up before they get here.
  COMFORT   - people are here, hold a normal comfortable temperature.
  PRECHARGE - electricity gets expensive and dirty soon, so heat up NOW
              while it is still cheap and store that warmth in the building.
  COAST     - electricity is expensive right now, so stop buying it and let
              the stored warmth carry the building through.
"""


@dataclass
class StrategyDecision:
    mode: str | None
    reason: str = ""
    latency_s: float = 0.0
    tools_used: list = field(default_factory=list)
    state_injected: bool = False
    parse_failed: bool = False


class StrategyAgent:
    """
    Picks one building-level ECM per decision cycle.

    Deliberately does NOT talk to the simulation directly. Its output is a
    label; `supervisor.apply_mode()` turns that into per-zone setpoints and
    enforces comfort. That split is what lets the model run on a slow
    cadence without putting occupants at risk - see `apply_mode`'s docstring.
    """

    def __init__(self, executor, model: str = DEFAULT_MODEL,
                 client: "ollama.Client | None" = None, max_tool_rounds: int = 3):
        self.executor = executor
        self.model = model
        self.client = client or ollama.Client(timeout=90)
        self.max_tool_rounds = max_tool_rounds

    def decide(self) -> StrategyDecision:
        start = time.monotonic()
        self.executor.begin_cycle()

        messages = [
            {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
            {"role": "user", "content": "Assess the building and choose a strategy."},
        ]

        # Phase 1 - let the agent gather whatever context it wants.
        for _ in range(self.max_tool_rounds):
            response = self.client.chat(
                model=self.model, messages=messages, tools=BUILDING_TOOL_SCHEMAS,
                options={"num_predict": 200, "temperature": 0.0, "seed": 42},
            )
            msg = response["message"]
            messages.append(msg)

            calls = [
                {"name": c["function"]["name"],
                 "arguments": _coerce_args(c["function"]["arguments"])}
                for c in (msg.get("tool_calls") or [])
            ]
            if not calls:
                calls = _parse_fallback_tool_calls(msg.get("content") or "")
            if not calls:
                break

            for call in calls:
                try:
                    result = self.executor.dispatch(call["name"], call["arguments"])
                except (TypeError, ValueError) as exc:
                    result = {"error": str(exc)}
                messages.append({"role": "tool", "content": json.dumps(result)})

        # If the agent never looked at the building, hand it the state anyway
        # rather than letting it decide blind. Tracked as `state_injected` and
        # reported, because an agent that has to be spoon-fed its own context
        # is a weaker result than one that fetches it, and that distinction
        # should not be silently absorbed.
        state_injected = "get_building_state" not in self.executor.calls
        if state_injected:
            messages.append({
                "role": "user",
                "content": "Building state: "
                           + json.dumps(self.executor.get_building_state()),
            })

        # Phase 2 - constrained action. No tools here: the only legal output
        # is one of the five strategy labels plus a reason.
        messages.append({
            "role": "user",
            "content": "Now reply with the single chosen strategy and a one-sentence reason.",
        })
        response = self.client.chat(
            model=self.model, messages=messages, format=STRATEGY_SCHEMA,
            options={"num_predict": 150, "temperature": 0.0, "seed": 42},
        )

        mode, reason, parse_failed = None, "", False
        try:
            payload = json.loads(response["message"]["content"])
            candidate = str(payload.get("mode", "")).strip().upper()
            if candidate in BUILDING_MODES:
                mode = candidate
            else:
                parse_failed = True
            reason = str(payload.get("reason", ""))[:200]
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Should be unreachable while the grammar is applied, but a
            # silent None here would look identical to "the agent declined",
            # so it is flagged rather than swallowed.
            parse_failed = True

        return StrategyDecision(
            mode=mode,
            reason=reason,
            latency_s=round(time.monotonic() - start, 3),
            tools_used=list(self.executor.calls),
            state_injected=state_injected,
            parse_failed=parse_failed,
        )


@dataclass
class AgentStepResult:
    reasoning: str
    tool_calls: list = field(default_factory=list)
    final_setpoint_c: float = None
    latency_s: float = 0.0
    raw_messages: list = field(default_factory=list)


class HVACAgent:
    def __init__(self, executor: ToolExecutor, model: str = DEFAULT_MODEL,
                 client: "ollama.Client | None" = None, max_tool_rounds: int = 4):
        self.executor = executor
        self.model = model
        self.client = client or ollama.Client(timeout=60)
        self.max_tool_rounds = max_tool_rounds

    def run_step(self) -> AgentStepResult:
        start = time.monotonic()
        # Any setpoint reaching the simulation must come from THIS cycle's
        # reasoning, not linger from the last one.
        self.executor.begin_cycle()

        # The agent fetches state itself via get_zone_state rather than
        # having it injected. Injecting it was tried and made results
        # worse: the model then answered on every cycle but chose badly
        # (1 of 6 correct, defaulting to one value regardless of input),
        # whereas failing to answer is safe - `run_step` returns None and
        # the supervisor applies a situation-appropriate fallback. A
        # confident wrong setpoint is worse than no setpoint.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Evaluate the current zone state and act."},
        ]

        tool_calls_made = []
        reasoning_text = ""

        for _ in range(self.max_tool_rounds):
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                options={
                    # Small models occasionally fall into a degenerate
                    # repetition loop instead of emitting a short tool call -
                    # cap generation length so one bad turn can't stall the run.
                    "num_predict": 300,
                    # Control decisions must be reproducible: at Ollama's
                    # default sampling temperature, identical code produced
                    # wildly different 24h outcomes run to run (see
                    # BUILD_LOG.md). Greedy decoding trades away variety the
                    # agent has no use for.
                    "temperature": 0.0,
                    "seed": 42,
                },
            )
            msg = response["message"]
            messages.append(msg)

            calls = [
                {"name": c["function"]["name"], "arguments": _coerce_args(c["function"]["arguments"])}
                for c in (msg.get("tool_calls") or [])
            ]
            if not calls:
                calls = _parse_fallback_tool_calls(msg.get("content") or "")

            if not calls:
                reasoning_text = (msg.get("content") or "").strip()
                break

            for call in calls:
                name, args = call["name"], call["arguments"]

                try:
                    result = self.executor.dispatch(name, args)
                except (TypeError, ValueError) as exc:
                    # Malformed tool args from the model - feed the error
                    # back so it can self-correct instead of crashing the loop.
                    result = {"error": str(exc)}

                tool_calls_made.append({"name": name, "arguments": args, "result": result})
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })

        latency = time.monotonic() - start
        final_setpoint = self.executor.consume_pending_setpoint()

        return AgentStepResult(
            reasoning=reasoning_text,
            tool_calls=tool_calls_made,
            final_setpoint_c=final_setpoint,
            latency_s=round(latency, 3),
            raw_messages=messages,
        )
