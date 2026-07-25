# System Architecture — Eco-Loop Building Agent

> Fill in every `[bracketed]` section with actual numbers/decisions once
> you have real run data (Hour 17-20 in PLAN.md). This template mirrors
> the four things the brief's deliverable #4 explicitly asks for:
> tool-calling architecture, prompt engineering, latency management, and
> handling long simulation logs.

## 1. Overview

Eco-Loop pairs a physics-based simulator ([EnergyPlus] / a lightweight
mock thermal model, see below) with a locally-hosted open-source LLM
([qwen2.5-coder:7b] via Ollama) in a closed control loop:

```
EnergyPlus / MockBuilding  --state-->  ToolExecutor.get_zone_state()
        ^                                        |
        |                                        v
   set_zone_setpoint()  <----tool call----  HVACAgent (Ollama chat + tools)
```

Each cycle: the agent calls `get_zone_state`, reasons about comfort vs.
energy tradeoffs against the current PMV and occupancy, then calls
`set_zone_setpoint` with its decision. The setpoint is clamped to a hard
safety range in `tools.py` before being fed back into the simulator —
the LLM cannot bypass this regardless of what it requests.

## 2. Tool-calling architecture

- **Transport**: direct tool-calling via Ollama's OpenAI-compatible
  `chat(..., tools=...)` API (`llm_agent.py`), not a standalone MCP
  server — chosen to minimize moving parts under the build deadline.
  `tools.py::TOOL_SCHEMAS` is already in OpenAI/MCP-compatible function
  schema form, so migrating to a real MCP server later is a transport
  swap, not a redesign.
- **Tools exposed**: `get_zone_state` (read-only sensor snapshot),
  `set_zone_setpoint` (the only write path into the simulation).
- **Self-correction**: malformed tool arguments are caught and fed back
  to the model as a tool-result error rather than crashing the loop
  (`llm_agent.py::run_step`), giving the model a chance to retry with
  corrected arguments within the same cycle (max [4] tool rounds).
- **[Fill in: did you observe self-correction happen in practice? How
  often did the model call tools in the wrong order / with bad args?]**

## 3. Prompt engineering strategy

- System prompt (`llm_agent.py::SYSTEM_PROMPT`) encodes an explicit
  priority order: comfort boundary (PMV within [-1.0, 1.0] while
  occupied) takes precedence over energy savings, with unoccupied
  setback as the main lever for savings.
- **[Fill in: any prompt iterations needed — e.g. did the model
  oscillate setpoints every cycle, refuse to call tools, or ignore the
  comfort priority until you reworded the prompt?]**

## 4. Latency management

- Mock-mode loop supports `--agent-every-n-steps` to call the LLM less
  than once per simulation step, trading reasoning granularity for
  throughput over long horizons.
- Per-step latency is logged (`latency_s` column in `logs/*.csv`) and
  surfaced in the dashboard as average agent latency/step.
- **[Fill in: measured average/p95 latency per LLM call on your
  hardware, and what --agent-every-n-steps value you settled on for the
  multi-day run.]**

## 5. Handling long simulation logs

- CSV logs are written incrementally per step rather than buffered in
  memory, so multi-day horizons don't grow unbounded RAM usage.
- The dashboard (`dashboard/generate_report.py`) aggregates (cumulative
  sum, mean) rather than rendering every raw row, keeping the report
  readable regardless of run length.
- **[Fill in: longest horizon actually run, and any log-size or
  EnergyPlus stdout-volume issues encountered.]**

## 6. Results

- Baseline energy: **[X] kWh**
- AI-driven energy: **[Y] kWh**
- Net reduction: **[Z]%**
- Comfort violations (AI-driven, occupied steps outside [20, 24]°C): **[N]**
- **[Fill in: any case where the agent traded comfort for savings, and
  how you addressed it if so.]**

## 7. Known limitations / next steps

- **[Fill in: real EnergyPlus integration status, anything left as
  future work.]**
