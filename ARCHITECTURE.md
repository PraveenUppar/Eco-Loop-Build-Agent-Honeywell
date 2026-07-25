# System Architecture — Eco-Loop Building Agent

A closed-loop supervisory controller for building HVAC: a locally-hosted
open-source LLM ingests live zone state, reasons about comfort, price and
grid carbon intensity, and injects setpoints back into a running
simulation each cycle.

This document covers the four things the brief asks for — tool-calling
architecture, prompt engineering, latency management, and handling long
simulation logs — plus the measured results and an honest account of what
does and does not work.

---

## 1. Overview

```
      ┌──────────────────────────────────────────────┐
      │  Simulation  (MockBuilding / EnergyPlus)     │
      │  zone temp · outdoor temp · occupancy · PMV  │
      └───────────────┬──────────────────────────────┘
                      │ get_zone_state / get_grid_forecast
                      ▼
      ┌──────────────────────────────────────────────┐
      │  HVACAgent   (Ollama, qwen2.5-coder:1.5b)    │
      │  tool-calling loop, greedy decoding          │
      └───────────────┬──────────────────────────────┘
                      │ proposed setpoint
                      ▼
      ┌──────────────────────────────────────────────┐
      │  Supervisor  (deterministic override layer)  │
      │  comfort interlock · preheat · load shifting │
      └───────────────┬──────────────────────────────┘
                      │ set_zone_setpoint (clamped 18–26 °C)
                      ▼
              back into the simulation
```

Each cycle the agent reads state, decides, and calls a tool. Every
setpoint passes through two independent safety layers before reaching the
simulation: the supervisor (situational) and a hard clamp in `tools.py`
(absolute). The LLM cannot bypass either.

**Modules**

| File | Responsibility |
|---|---|
| `tools.py` | Thermal model, comfort/safety clamp, tool schemas and executor |
| `grid.py` | Time-of-use price and grid carbon intensity curves |
| `llm_agent.py` | Ollama tool-calling loop, prompt, fallback parser |
| `supervisor.py` | Deterministic override layer + intervention accounting |
| `run_loop.py` | Orchestration, four run modes, CSV logging |
| `dashboard/` | Baseline-vs-agent comparison and charts |

---

## 2. Tool-calling architecture

**Transport.** Direct tool-calling via Ollama's OpenAI-compatible
`chat(..., tools=...)` API rather than a standalone MCP server — fewer
moving parts under a deadline. `TOOL_SCHEMAS` in `tools.py` is already in
OpenAI/MCP-compatible function-schema form, so migrating to a real MCP
server is a transport swap, not a redesign.

**Tools exposed to the model**

| Tool | Purpose |
|---|---|
| `get_zone_state` | Temperature, outdoor temperature, occupancy, PMV, minutes until occupancy changes, current price/carbon, minutes until price peak |
| `get_grid_forecast` | Six-hour price and carbon forecast, for deciding *when* to heat |
| `set_zone_setpoint` | The only write path into the simulation |

**Safety is enforced in code, never in the prompt.** `clamp_setpoint()`
restricts every request to 18–26 °C regardless of what the model asks
for. The narrower 20–24 °C comfort band is what the dashboard audits.
Two tiers: a hard safety rail and a comfort target.

**Self-correction.** Malformed tool arguments are caught and returned to
the model as a tool-result error rather than crashing the loop, giving it
a chance to retry within the same cycle (max 4 tool rounds).

**Fallback tool-call parser.** `qwen2.5-coder:1.5b` does not reliably
emit the `<tool_call>` tags its own chat template requires, so Ollama
often returns tool calls as plain text in `message.content` with an empty
`tool_calls` field. Symptom: the setpoint never moved off its default for
an entire 24-hour run. `_parse_fallback_tool_calls()` scans message
content for balanced-brace JSON objects naming a known tool and
normalises the malformed argument shapes actually observed —
markdown-fenced JSON, `"arguments": null`, arguments as a JSON-encoded
string, and multiple concatenated objects.

---

## 3. Prompt engineering strategy

The prompt went through five revisions. What worked was not better
phrasing but **lower cognitive load**.

| Revision | Result |
|---|---|
| Priority hierarchy in prose ("comfort takes precedence over energy") | Setpoints stuck at default; then oscillating 21→18→24→23→20 |
| Explicit numeric setback ("hold 18.0 °C when empty") | Held steady, but energy 13.9% worse than baseline |
| Four-branch conditional with preheat | 40/40 occupied steps in comfort violation |
| Flat `condition → exact number` rule table | Model emitted `18.915` — a value from no rule |
| Same table, plus greedy decoding | Usable; 12.5% override rate (3 rules) |
| Five rules, after adding grid-awareness | 60.4% override rate |
| Cut back to three rules, three output values | 95.8% override rate; 44/48 cycles produced no setpoint |
| State injected into the prompt instead of fetched | Always answered, but 1/6 correct — reverted |

**Findings that generalise:**

- **Flat rule tables beat priority hierarchies.** A 1.5 B model cannot
  reliably evaluate "apply the first rule that matches" across four
  nested conditions. Each rule now maps a condition to one exact number.
- **Name the number, never describe it.** "Near the low end of the safe
  range" produced 23 °C. "Exactly 18.0" produced 18.0.
- **State stability is correct.** Without an explicit instruction that
  repeating the previous setpoint is expected, the model treated every cycle
  as needing a fresh answer and oscillated.
- **Determinism is non-negotiable for control.** At Ollama's default
  sampling temperature, identical code produced wildly different 24-hour
  outcomes, making every A/B comparison meaningless. Fixed with
  `temperature: 0.0` and `seed: 42`.

**The honest limit — a capability probe, not a phrasing problem.**
To establish whether any prompt could work, the task was stripped to the
simplest form it can take: two pre-computed booleans, a three-line lookup
table, no arithmetic, no thresholds, no multi-step tool flow.

```
building_empty_for_a_while is true   -> 18.0
price_peak_coming_soon is true       -> 23.5
otherwise                            -> 20.5
```

The model scored 3/6 — and the three it got right were exactly the three
whose answer was the `otherwise` line. It failed every case where a flag
was true. It is not reading the input; it emits a constant.

There is nothing below this to simplify. Anything further means writing
the answer into the prompt, at which point the model is not controlling
anything. Two related failures point the same way:

- **Numeric thresholds fail.** "more than 120", "between 1 and 120" —
  every miss in the three-rule version was a range comparison.
- **It copies numbers out of its input.** It answered `21.0` (the current
  setpoint, permitted by no rule) and earlier `18.915` (the current zone
  temperature). `setpoint_c` was removed from the state passed to it for
  this reason.

This is why the supervisor exists (§5), and why the model is the binding
constraint on this project rather than the prompt.

---

## 4. Latency management

| Metric | Value |
|---|---|
| Model | `qwen2.5-coder:1.5b` (986 MB) via Ollama |
| Mean latency per agent decision | ~2.0 s |
| Typical range | 0.3 – 5 s |
| Decision cadence | `--agent-every-n-steps` (4 = hourly at 15-min steps) |

Three controls:

1. **Decoupled decision cadence.** The agent runs every *N* simulation
   steps, holding its setpoint between calls. At `N=4` a 48-hour run costs
   48 LLM calls rather than 192 — the control problem does not change
   meaningfully every 15 minutes.
2. **Bounded generation.** `num_predict: 300` caps tokens per turn. One
   run hung for 10+ minutes when the model fell into a degenerate
   repetition loop with no cap.
3. **Client timeout.** `ollama.Client(timeout=60)` as a second safety
   net, so a stalled request cannot block the loop indefinitely.

Latency is logged per step (`latency_s`) and surfaced on the dashboard.

**Hardware note.** `qwen2.5-coder:7b` was the intended model but does not
run on the development machine (8 GB RAM, RTX 2050 with 4 GB VRAM) —
Ollama reports `model requires more system memory (1.8 GiB) than is
available` and the runner terminates. The model name is a single
constant (`DEFAULT_MODEL`) plus a `--model` flag.

---

## 5. The supervisory override layer

Because the LLM cannot be trusted to hold comfort on its own, a
deterministic layer sits between its proposal and the simulation. It
overrides only physically indefensible proposals; anything merely
suboptimal is left alone, so the agent retains real authority.

| Rule | Trigger | Action |
|---|---|---|
| 1 | Occupied, zone outside comfort band, proposal not targeting the band | Drive to comfort |
| 2 | Occupancy imminent, proposal too low to preheat in time | Preheat |
| 3 | Building empty, occupancy not imminent, proposal above setback | Setback |
| 4 | Occupants leaving soon, zone projected to stay comfortable | Coast on thermal mass |
| 5 | Price peak approaching, power still cheap | Pre-charge thermal mass |
| 6 | Peak active, stored heat available | Discharge instead of buying |

**Intervention rate is reported, not hidden.** `OverrideStats` tracks it,
every step logs `agent_proposed_c` / `overridden` / `override_reason`,
and the dashboard displays the percentage with a caption stating that a
high rate means the results reflect the rules rather than the model.
`--no-supervisor` measures unassisted agent quality, and a `mock-rules`
mode runs the rules with **no LLM at all** as an experimental control —
that arm is what makes the model's true contribution legible.

This mirrors real BMS practice, where supervisory optimisation sits under
a comfort interlock. The difference between honest and dishonest use of
that pattern is whether the intervention rate is published.

---

## 6. Handling long simulation logs

- **Streamed, not buffered.** CSV rows are written per step, so a
  multi-day horizon does not grow unbounded memory.
- **The agent never sees raw logs.** It receives a small structured state
  dict per cycle — roughly 8 fields — rather than accumulated history.
  Log length is therefore decoupled from prompt length, which is what
  keeps latency flat over long horizons.
- **Fresh conversation per decision.** Each cycle starts a new message
  list, so context cannot grow without bound across a run.
- **Aggregate reporting.** The dashboard summarises (sums, means,
  violation counts) rather than rendering every row.
- **UTF-8 explicitly.** Log files are opened with `encoding="utf-8"`;
  the Windows default (cp1252) silently corrupted files when the model's
  reasoning text contained a `°` character.

---

## 7. Results

48-hour horizon, winter profile (−1 °C to 9 °C), 15-minute steps.
Baseline is a fixed setback schedule: 21 °C occupied, 18 °C otherwise.

Three arms are reported. `mock-rules` — the supervisor rules with **no
LLM at all** — is the experimental control, and it is the arm that makes
the language model's true contribution legible.

| | Energy (kWh) | Cost | CO₂ (kg) | Peak-window kWh | Comfort violations | Overrides |
|---|---|---|---|---|---|---|
| Fixed schedule (baseline) | 57.38 | 8.18 | 15.49 | 7.57 | 8 / 80 | — |
| Rules only, no LLM | 58.11 | **7.84** | **15.14** | 5.73 | **0 / 80** | — |
| LLM + supervisor | 60.37 | 8.08 | 15.73 | 5.52 | **0 / 80** | 60.4 % |

Change versus baseline:

| | Cost | CO₂ | Peak-window energy |
|---|---|---|---|
| Rules only | **−4.2 %** | **−2.3 %** | **−24.3 %** |
| LLM + supervisor | −1.2 % | **+1.5 % (worse)** | −27.0 % |

**The load-shifting strategy works.** Peak-window draw falls by roughly a
quarter in both arms, and during the peak the measured HVAC draw reaches
exactly zero for several consecutive steps — the building coasts on heat
banked while power was cheap. Comfort violations are eliminated in both
arms. Total energy rises slightly, which is expected: this shifts *when*
energy is bought, not how much.

**But the LLM makes the result worse, not better.** On the metrics that
matter it is beaten by its own rule set: −1.2 % cost against −4.2 %, and
carbon actually 1.5 % *worse than baseline*. The override rate tells the
story — 60.4 %, up from 12.5 % before grid-awareness was added. Going
from a 3-rule to a 5-rule prompt pushed `qwen2.5-coder:1.5b` past its
capability, so the supervisor now corrects most of its decisions, and the
ones that survive are worse than what the rules would have chosen.

The headline savings in this project therefore belong to the
deterministic control logic, not to the language model. Attributing them
to the LLM would be false.

### Why not a larger headline energy number

With **flat** pricing, the supervisor rules landed within ±1 % of the
fixed baseline across every climate tested (outdoor base 14 / 8 / 2 /
−4 °C). A well-chosen setback schedule already captures nearly all
available savings in a single-zone building with static occupancy. Real
headroom appears only once something varies that a static schedule cannot
track — here, price and carbon intensity. Reporting a large "energy
saving" against this baseline would have required either a weaker
baseline or a broken energy model.

---

## 8. Verification and known defects found

Bugs found in this codebase during development, all fixed and recorded in
`BUILD_LOG.md` with reproduction detail:

- **Exploitable energy model.** Energy was charged as
  `abs(setpoint − zone_temp)`, making it nearly free to hold
  `setpoint == zone_temp`. The LLM found the exploit and began echoing
  the current zone temperature back as its setpoint. Replaced with a
  degree-day model including standing envelope loss.
- **Controller droop.** A plain proportional term meant a 21 °C setpoint
  settled at 19.2 °C, so the baseline's nominal comfort was fictitious.
  Now converges to setpoint (measured droop 0.00 °C).
- **Stale occupancy at period boundaries.** `occupied` was read from a
  flag refreshed only inside `step()`, so at exactly 08:00 the horizon
  reported 1440 minutes instead of 0 and ordered a setback as occupants
  arrived.
- **Stacked drift rules.** Optimal-stop and peak-discharge both fired at
  once, compounding their temperature drops through the comfort floor.
  Both are now gated on `projected_temp_after()`, validated against
  logged data (predicted 20.23 °C where the simulation produced
  20.223 °C).
- **Encoding.** Log files written in the Windows default codepage became
  unreadable when reasoning text contained `°`.

Supervisor logic has a 10-case test covering each override branch plus
the cases that must *not* override.

---

## 9. Limitations and next steps

- **EnergyPlus integration is scaffolded, not complete.** `run_loop.py`
  has the PyEnergyPlus callback structure with explicit `TODO`s for zone
  and actuator handles; EnergyPlus is not installed on the development
  machine. All results above come from the lightweight thermal model,
  which is a proxy — its parameters are plausible but not calibrated
  against a real building.
- **The LLM's contribution is currently negative.** The `mock-rules`
  control arm beats the LLM arm on both cost and carbon (§7). Model
  capability is the binding constraint: at 3 rules the override rate was
  12.5 %, at 5 rules it is 60.4 %. Running `qwen2.5-coder:7b` on adequate
  hardware is the single highest-value change and needs only a `--model`
  flag. Until then, the honest claim is that this project demonstrates a
  working closed-loop *architecture* with a model too small to drive it.
- **Single zone, static occupancy schedule.** Multi-zone control with
  stochastic occupancy is where LLM reasoning would plausibly beat a rule
  table, because the rule table stops being writable by hand.
- **Price and carbon curves are representative, not measured** — stated
  shapes for a grid with solar and gas peaking plant, not utility data.
