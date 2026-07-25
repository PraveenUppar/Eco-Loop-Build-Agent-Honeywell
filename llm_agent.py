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

from tools import TOOL_SCHEMAS, ToolExecutor

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
1. Call get_zone_state to see the current temperature, outdoor temperature,
   occupancy, and PMV thermal comfort index.
2. Decide whether the setpoint should change, based on these priorities in order:
   a. Never let PMV drift outside [-1.0, 1.0] while the zone is occupied - this
      is the comfort boundary and takes precedence over energy savings.
   b. When occupied and comfort allows it, prefer setpoints that reduce the gap
      to outdoor temperature (less conditioning effort = less energy).
   c. When unoccupied, you may set back the setpoint toward outdoor temperature
      to save energy, since no one is present to feel discomfort.
3. Call set_zone_setpoint with your chosen value. The tool clamps to a safe
   hard range automatically, so it's fine to call it even near the edges.
4. Give one short sentence of reasoning for your decision before finishing.

Only use the two tools provided. Do not invent tools or parameters. Make at
most one set_zone_setpoint call per cycle.
"""


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
        self.client = client or ollama.Client()
        self.max_tool_rounds = max_tool_rounds

    def run_step(self) -> AgentStepResult:
        start = time.monotonic()
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
